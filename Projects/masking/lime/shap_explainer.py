import random
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
import numpy as np
import shap

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# Paths to models
encoder_path = "model/epochs_500_latent_128/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128/classifier_final.pth"

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
classifier = Classifier().to(device)

encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

encoder.eval()
decoder.eval()
classifier.eval()

# Global list to store results for the CSV file
results = []

# Function to randomly select an image from the dataset directory
def select_random_image(dataset_dir):
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith(".png")]
    return os.path.join(dataset_dir, random.choice(image_files))

# 1. Read the CSV file to get the label for a specific image
def get_actual_label(csv_file, image_path):
    image_name = os.path.basename(image_path)
    df = pd.read_csv(csv_file)
    label_row = df[df["image_filename"] == image_name]
    if label_row.empty:
        raise ValueError(f"Image {image_name} not found in the CSV file.")
    label = label_row["label"].values[0]
    actual_label = 0 if label == "STOP" else 1
    return actual_label

# 2. Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((80, 160)),  # Assuming CARLA image size
        transforms.ToTensor(),
    ])
    transformed_image = transform(image)
    transformed_image_tensor = (
        transformed_image if isinstance(transformed_image, torch.Tensor) else torch.tensor(transformed_image)
    )
    return transformed_image_tensor.unsqueeze(0).to(device)

# Function to generate background latent vectors from the dataset
def get_background_latent_vectors(encoder, dataset_dir, num_samples=100):
    image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".png")]
    background_latent_vectors = []
    
    for i, image_path in enumerate(image_files):
        if i >= num_samples:  # Limit the number of samples
            break
        image_tensor = preprocess_image(image_path).to(device)
        latent_vector = encoder(image_tensor)[2]  # Get latent vector from encoder
        background_latent_vectors.append(latent_vector.cpu().detach().numpy())
    
    return np.array(background_latent_vectors)

# 3. Apply SHAP on the latent space
def apply_shap_on_latent_space(latent_vector, classifier, background_latent_vectors):
    # Background latent vectors need to be tensors
    background_latent_vectors = torch.tensor(background_latent_vectors, dtype=torch.float32).to(device)

    # SHAP Explainer
    explainer = shap.GradientExplainer(classifier, background_latent_vectors)

    # SHAP values for the latent vector
    shap_values = explainer.shap_values(latent_vector)

    # Summarize the most important features based on SHAP values
    important_features = np.argsort(np.abs(shap_values[0]))[-5:]  # Select top 5 most important features
    print(f"Important features identified by SHAP: {important_features}")

    # Also print SHAP values for the important features
    print(f"SHAP values for these important features: {shap_values[0][important_features]}")
    
    return important_features

# 4. Mask latent features with different methods
def mask_latent_features_using_shap(latent_vector, classifier, background_latent_vectors):
    important_features = apply_shap_on_latent_space(latent_vector, classifier, background_latent_vectors)
    
    # Proceed with the same masking logic as before using the important features identified by SHAP
    masked_latent_vector = mask_latent_features(latent_vector, important_features, "median")
    return masked_latent_vector

# Function to mask latent features
def mask_latent_features(latent_vector, important_features, method="median"):
    masked_latent_vector = latent_vector.clone()
    if method == "median":
        median_value = torch.median(latent_vector)
        for feature in important_features:
            masked_latent_vector[:, feature] = median_value
    elif method == "zero":
        for feature in important_features:
            masked_latent_vector[:, feature] = 0
    return masked_latent_vector

# 5. Function to save and visualize images
def save_images(
    original_image,
    reconstructed_image_before_masking,
    reconstructed_image_after_masking,
    reconstruction_loss_before,
    reconstruction_loss_after,
    method,
    actual_label,
    predicted_label_before,
    predicted_label_after,
    image_name,
):
    # Ensure the results directory exists
    results_dir = "plots/reconstruction"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Convert tensors to numpy arrays for visualization
    original_image_np = original_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_before_np = reconstructed_image_before_masking.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_after_np = reconstructed_image_after_masking.cpu().detach().squeeze().numpy().transpose(1, 2, 0)

    # Save images here...

# Function to test a single image with SHAP and masking
def test_single_image_with_shap(image_path, csv_file):
    actual_label = get_actual_label(csv_file, image_path)
    print(f"Image {image_path} is selected and its label is: {'STOP' if actual_label == 0 else 'GO'}")
    image = preprocess_image(image_path)

    # Step 1: Get the latent vector from the encoder
    with torch.no_grad():
        latent_vector = encoder(image)[2]
        print(f"Original Latent Vector:\n {latent_vector}")

    latent_vector.requires_grad = True  # Ensure requires_grad=True

    # **Reshape latent vector before passing into classifier**
    latent_vector = latent_vector.view(1, -1)  # Reshape to (batch_size, num_features)

    # Step 2: Get original prediction from the classifier
    original_prediction = F.softmax(classifier(latent_vector), dim=1)
    original_class = torch.argmax(original_prediction, dim=1).item()
    predicted_label = "STOP" if original_class == 0 else "GO"
    print(f"Original Prediction: {predicted_label}")

    # Step 3: Decode the latent vector to get the reconstructed image
    reconstructed_image = decoder(latent_vector)
    reconstructed_image_resized = F.interpolate(
        reconstructed_image,
        size=image.shape[2:],  # Ensure reconstruction matches the original image size
        mode="bilinear",
        align_corners=False,
    )

    # Compute reconstruction loss
    reconstruction_loss = F.mse_loss(reconstructed_image_resized, image).item()
    print(f"Reconstruction Loss (Without Masking): {reconstruction_loss}")

    # Step 4: Save original and reconstructed images
    save_images(
        image,
        reconstructed_image,
        reconstructed_image,
        reconstruction_loss,
        reconstruction_loss,
        "original",
        actual_label,
        predicted_label,
        predicted_label,
        os.path.basename(image_path),
    )

    # Step 5: Use background latent vectors for SHAP
    background_latent_vectors = get_background_latent_vectors(encoder, "dataset/town7_dataset/test")

    # Step 6: Apply SHAP to identify important features
    important_features = apply_shap_on_latent_space(latent_vector, classifier, background_latent_vectors)

    # Step 7: Mask the latent vector based on SHAP important features
    masked_latent_vector = mask_latent_features(latent_vector, important_features, method="median")

    # Step 8: Get prediction after masking
    masked_prediction = classifier(masked_latent_vector)
    masked_class = torch.argmax(masked_prediction, dim=1).item()
    masked_label_str = "STOP" if masked_class == 0 else "GO"
    print(f"Masked Prediction (median): {masked_label_str}")

    # Step 9: Decode the masked latent vector to get the masked reconstructed image
    masked_reconstructed_image = decoder(masked_latent_vector)
    masked_reconstructed_image_resized = F.interpolate(
        masked_reconstructed_image,
        size=image.shape[2:],  # Match the original image size
        mode="bilinear",
        align_corners=False,
    )
    masked_reconstruction_loss = F.mse_loss(masked_reconstructed_image_resized, image).item()
    print(f"Reconstruction Loss after Masking with (median) method: {masked_reconstruction_loss}")

    # Save the final images
    save_images(
        image,
        reconstructed_image,
        masked_reconstructed_image,
        reconstruction_loss,
        masked_reconstruction_loss,
        "median",
        actual_label,
        predicted_label,
        masked_label_str,
        os.path.basename(image_path),
    )

# Process multiple images with SHAP
def process_multiple_images_with_shap(image_dir, csv_file):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]

    for image_path in image_files:
        print(f"Processing image: {image_path}")
        try:
            test_single_image_with_shap(image_path, csv_file)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

# Save results to CSV
def print_and_save_results():
    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

# Run the process
process_multiple_images_with_shap("dataset/town7_dataset/test/", "dataset/town7_dataset/test/labeled_test_data_log.csv")
print_and_save_results()
