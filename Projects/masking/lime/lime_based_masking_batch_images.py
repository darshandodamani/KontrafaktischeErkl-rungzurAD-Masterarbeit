import random
from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lime.lime_tabular import LimeTabularExplainer
import sys
import numpy as np

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
    print(f"Input Image Original shape: {np.array(image).shape}")
    transform = transforms.Compose([
        transforms.Resize((80, 160)),  # Assuming CARLA image size
        transforms.ToTensor(),
    ])
    transformed_image = transform(image)
    transformed_image_tensor = (
        transformed_image if isinstance(transformed_image, torch.Tensor) else torch.tensor(transformed_image)
    )
    return transformed_image_tensor.unsqueeze(0).to(device)

# 3. Apply LIME on the latent space
def apply_lime_on_latent_space(latent_vector, classifier):
    latent_vector_np = latent_vector.cpu().numpy().flatten()
    print("--------------------")
    print(f"Latent vector shape: {latent_vector_np.shape}")

    # LIME Tabular Explainer
    explainer = LimeTabularExplainer(
        latent_vector_np.reshape(1, -1),
        feature_names=[f"latent_dim_{i}" for i in range(latent_vector_np.shape[0])],
        class_names=["STOP", "GO"],
        discretize_continuous=False,
    )
    # Explain the instance
    explanation = explainer.explain_instance(
        latent_vector_np,
        lambda x: F.softmax(
            classifier(torch.tensor(x).view(-1, 128).float().to(device)), dim=1
        )
        .cpu()
        .detach()
        .numpy(),
        num_features=5,  # Adjust number of features as needed
    )

    return explanation

# 4. Mask latent features with different methods
def mask_latent_features(latent_vector, important_features, method="zero"):
    masked_latent = latent_vector.clone()
    print("--------------------")
    # Ensure that important_features contains the correct indices
    print(f"Features being masked: {important_features}")

    if method == "zero":
        print("Masking features by setting them to 0...")
        masked_latent[:, important_features] = 0

    elif method == "median":
        median_val = torch.median(latent_vector).item()
        print(f"Median value: {median_val}")
        masked_latent[:, important_features] = median_val

    elif method == "random":
        random_vals = torch.randn_like(masked_latent[:, important_features])
        print("Masking features by setting them to random values...")
        masked_latent[:, important_features] = random_vals

    return masked_latent

# 5. Function to save and visualize images
def save_images_combined(
    original_image,
    reconstructed_image_before_masking,
    reconstructed_image_after_masking,
    reconstruction_loss_before,
    reconstruction_loss_after,
    method,
    actual_label,
    predicted_label_before,
    predicted_label_after,
    counterfactual_image=None,
    counterfactual_label=None,
    counterfactual_loss=None,
    image_name="combined"
):
    # Ensure the results directory exists
    results_dir = "plots/reconstruction"
    counterfactual_dir = os.path.join(results_dir, "counterfactual")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(counterfactual_dir):
        os.makedirs(counterfactual_dir)

    # Convert tensors to numpy arrays for visualization
    original_image_np = original_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_before_np = reconstructed_image_before_masking.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_after_np = reconstructed_image_after_masking.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    
    # Create the combined plot
    fig, axs = plt.subplots(1, 4 if counterfactual_image is not None else 3, figsize=(20, 5))

    # Plot original image
    axs[0].imshow(original_image_np)
    # axs[0].set_title(f"Original\nLabel: {'STOP' if actual_label == 0 else 'GO'}")
    axs[0].set_title(f"Original")

    # Plot reconstructed image before masking
    axs[1].imshow(reconstructed_before_np)
    axs[1].set_title(f"Before Masking\nLoss: {reconstruction_loss_before:.4f}\nPred: {predicted_label_before}")

    # Plot reconstructed image after masking
    axs[2].imshow(reconstructed_after_np)
    axs[2].set_title(f"After Masking ({method})\nLoss: {reconstruction_loss_after:.4f}\nPred: {predicted_label_after}")

    # If a counterfactual explanation is generated, plot it
    if counterfactual_image is not None:
        counterfactual_image_np = counterfactual_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
        axs[3].imshow(counterfactual_image_np)
        # axs[3].set_title(f"Counterfactual\nLoss: {counterfactual_loss:.4f}\nPred: {counterfactual_label}")
        axs[3].set_title(f"Counterfactual\nPred: {counterfactual_label}")

        # Save the counterfactual image separately
        plt.figure(figsize=(5, 5))
        plt.imshow(counterfactual_image_np)
        # plt.title(f"Counterfactual\nLoss: {counterfactual_loss:.4f}\nPred: {counterfactual_label}")
        plt.title(f"Counterfactual\nPred: {counterfactual_label}")
        plt.savefig(f"{counterfactual_dir}/counterfactual_{image_name}.png")
        plt.close()

        # Print confirmation
        print(f"Counterfactual explanation image saved at: {counterfactual_dir}/counterfactual_{image_name}.png")

    plt.savefig(f"{results_dir}/{image_name}_combined.png")
    plt.close()

# 6. Main function to test a single image with LIME and masking
def test_single_image(image_path, csv_file):
    actual_label = get_actual_label(csv_file, image_path)
    print(f"Image {image_path} is selected and its label is: {'STOP' if actual_label == 0 else 'GO'}")
    image = preprocess_image(image_path)

    with torch.no_grad():
        latent_vector = encoder(image)[2]
        print(f"Original Latent Vector:\n {latent_vector}")

        original_prediction = classifier(latent_vector)
        original_class = torch.argmax(original_prediction, dim=1).item()
        predicted_label = "STOP" if original_class == 0 else "GO"
        print(f"Original Prediction: {predicted_label}")

        reconstructed_image = decoder(latent_vector)
        reconstructed_image_resized = F.interpolate(
            reconstructed_image, size=image.shape[2:], mode="bilinear", align_corners=False
        )
        reconstruction_loss = F.mse_loss(reconstructed_image_resized, image).item()
        print(f"Reconstruction Loss (Without Masking): {reconstruction_loss}")

        explanation = apply_lime_on_latent_space(latent_vector, classifier)
        important_features = [int(feature.split("_")[-1]) for feature, _ in explanation.as_list()]
        print(f"Important features identified by LIME: {important_features}")

        for method in ["median"]:
            masked_latent_vector = mask_latent_features(latent_vector, important_features, method)
            masked_prediction = classifier(masked_latent_vector)
            masked_class = torch.argmax(masked_prediction, dim=1).item()
            masked_label_str = "STOP" if masked_class == 0 else "GO"
            print(f"Masked Prediction ({method}): {masked_label_str}")

            masked_reconstructed_image = decoder(masked_latent_vector)
            masked_reconstructed_image_resized = F.interpolate(
                masked_reconstructed_image, size=image.shape[2:], mode="bilinear", align_corners=False
            )
            masked_reconstruction_loss = F.mse_loss(masked_reconstructed_image_resized, image).item()
            print(f"Reconstruction Loss after Masking with ({method}) method: {masked_reconstruction_loss}")

            counterfactual_image, counterfactual_label, counterfactual_loss = None, None, None
            if masked_class != original_class:
                print(f"Prediction changed after masking with {method}. Generating counterfactual explanation.")
                re_encoded_latent_vector = encoder(masked_reconstructed_image_resized)[2]
                re_encoded_prediction = classifier(re_encoded_latent_vector)
                re_encoded_class = torch.argmax(re_encoded_prediction, dim=1).item()
                re_encoded_label = "STOP" if re_encoded_class == 0 else "GO"
                print(f"Re-encoded Prediction: {re_encoded_label}")

                if re_encoded_class != original_class:
                    print(f"Counterfactual explanation found!")
                    counterfactual_image = masked_reconstructed_image_resized
                    counterfactual_label = re_encoded_label
                    counterfactual_loss = F.mse_loss(counterfactual_image, image).item()

            # Save the combined image (original, before/after masking, and counterfactual)
            save_images_combined(
                image, reconstructed_image, masked_reconstructed_image, reconstruction_loss, 
                masked_reconstruction_loss, method, actual_label, predicted_label, masked_label_str, 
                counterfactual_image, counterfactual_label, counterfactual_loss, os.path.basename(image_path)
            )

            # Store result in the global list for CSV
            results.append({
                "Image": os.path.basename(image_path),
                "Method": method,
                "Reconstruction Loss Before": reconstruction_loss,
                "Reconstruction Loss After": masked_reconstruction_loss,
                "Original Prediction": predicted_label,
                "Masked Prediction": masked_label_str,
                "Counterfactual Generated": counterfactual_image is not None,
                "Counterfactual Prediction": counterfactual_label if counterfactual_image is not None else "N/A",
                "Counterfactual Loss": counterfactual_loss if counterfactual_image is not None else "N/A"
            })
            
            # Save results to CSV after each image is processed
            # df = pd.DataFrame(results)
            # df.to_csv("plots/reconstruction/masking_analysis_results.csv", index=False)
            # print(f"Results updated and saved to CSV at: plots/reconstruction/masking_analysis_results.csv")

            

# 7. Function to randomly select an image and wait for user input before processing the next one
def process_random_images(dataset_dir, csv_file):
    while True:
        input("Press Enter to process a new random image, or Ctrl+C to exit.")
        image_path = select_random_image(dataset_dir)
        test_single_image(image_path, csv_file)

# # After processing, save the results to a CSV
# def save_results_to_csv():
#     df = pd.DataFrame(results)
#     df.to_csv("plots/reconstruction/masking_analysis_results.csv", index=False)
#     print("Results saved to CSV at: plots/reconstruction/masking_analysis_results.csv")

# Call the function to process random images
process_random_images("dataset/town7_dataset/test/", "dataset/town7_dataset/test/labeled_test_data_log.csv")

# # Save the results to CSV at the end
# save_results_to_csv()
