import os
import sys
import torch
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import torchvision.transforms as transforms

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# Paths to models
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"

# Load models
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

# Define function for LIME-based masking
def apply_lime_mask(image, mask, mask_value=0):
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np[mask > 0] = mask_value  # Mask positive contributions
    return transforms.ToTensor()(image_np).unsqueeze(0).to(device)

# Define function to calculate image metrics
def calculate_image_metrics(original, modified):
    original_np = np.array(original, dtype=np.float32) / 255.0
    modified_np = np.array(modified, dtype=np.float32) / 255.0

    if original_np.shape != modified_np.shape:
        from skimage.transform import resize
        modified_np = resize(modified_np, original_np.shape, anti_aliasing=True)

    min_dim = min(original_np.shape[0], original_np.shape[1])
    win_size = min(min_dim, 7)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    metrics = {
        "SSIM": ssim(original_np, modified_np, win_size=win_size, channel_axis=-1, data_range=1.0),
        "MSE": mse(original_np, modified_np),
        "PSNR": psnr(original_np, modified_np, data_range=1.0),
        "VIF": vifp(original_np, modified_np),
        "UQI": uqi(original_np, modified_np),
    }
    return metrics

# Define classifier prediction function for LIME
def classifier_prediction(image_tensor):
    """
    Function to generate predictions for LIME from the classifier.
    """
    try:
        with torch.no_grad():
            # Convert input to the correct shape
            image_tensor = torch.tensor(image_tensor.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
            
            # Send the image through encoder and classifier
            latent_vector = encoder(image_tensor)[2]
            prediction = classifier(latent_vector)
            
            # Convert predictions to probabilities
            probabilities = F.softmax(prediction, dim=1).cpu().detach().numpy()
            
            # Check for NaN or invalid values
            if np.isnan(probabilities).any():
                print("Warning: NaN values detected in classifier predictions.")
                probabilities = np.nan_to_num(probabilities, nan=0.5)  # Handle invalid predictions by replacing NaN with 0.5
            return probabilities
    except Exception as e:
        print(f"Error in classifier_prediction: {e}")
        return np.zeros((image_tensor.shape[0], 2))  # Return valid dummy output if an error occurs

# Ensure LIME data doesn't contain NaN
def clean_lime_data(data, labels, weights):
    """
    Cleans the data, labels, and weights used in LIME to ensure no NaN values.
    """
    valid_indices = ~np.isnan(labels)
    return data[valid_indices], labels[valid_indices], weights[valid_indices]

# Wrap the LIME `explain_instance` method with NaN handling
def safe_explain_instance(explainer, image, predict_fn, **kwargs):
    """
    Safely generates a LIME explanation by handling NaN values in the input or prediction.
    """
    try:
        explanation = explainer.explain_instance(image, predict_fn, **kwargs)
        return explanation
    except ValueError as ve:
        print(f"ValueError during LIME explanation: {ve}")
        # Return an empty explanation as fallback
        return None


# Process dataset
def process_dataset(dataset_dir, output_csv):
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "Image File", "Prediction", "Confidence (Before Masking)", "Prediction (After Masking)",
            "Confidence (After Masking)", "Counterfactual Found", "SSIM", "MSE", "PSNR", "UQI", "VIFP"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_filename in os.listdir(dataset_dir):
            if not image_filename.lower().endswith(('.png')):
                continue

            image_path = os.path.join(dataset_dir, image_filename)
            image = Image.open(image_path).convert('RGB')
            input_image = transform(image).unsqueeze(0).to(device)

            # Initial prediction
            latent_vector = encoder(input_image)[2]
            reconstructed_image = decoder(latent_vector)
            prediction = classifier(latent_vector)
            predicted_label = torch.argmax(prediction, dim=1).item()
            predicted_class = "STOP" if predicted_label == 0 else "GO"
            confidence_before = F.softmax(prediction, dim=1).cpu().detach().numpy()

            # LIME explanation
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                np.array(image),
                classifier_prediction,
                hide_color=0,
                num_samples=1000
            )
            
            if explanation is None:
                print(f"Skipping image {image_filename} due to NaN issues in LIME explanation.")
                continue
            
            temp, mask = explanation.get_image_and_mask(
                label=predicted_label,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )

            # Apply mask and get prediction
            masked_image = apply_lime_mask(image, mask)
            latent_vector_masked = encoder(masked_image)[2]
            reconstructed_image_masked = decoder(latent_vector_masked)
            masked_prediction = classifier(latent_vector_masked)
            masked_label = torch.argmax(masked_prediction, dim=1).item()
            masked_class = "STOP" if masked_label == 0 else "GO"
            confidence_after = F.softmax(masked_prediction, dim=1).cpu().detach().numpy()
            counterfactual_found = masked_class != predicted_class

            # Metrics
            metrics = calculate_image_metrics(
                np.array(image),
                masked_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            )

            # Write results to CSV
            writer.writerow({
                "Image File": image_filename,
                "Prediction": predicted_class,
                "Confidence (Before Masking)": confidence_before.tolist(),
                "Prediction (After Masking)": masked_class,
                "Confidence (After Masking)": confidence_after.tolist(),
                "Counterfactual Found": counterfactual_found,
                "SSIM": metrics["SSIM"],
                "MSE": metrics["MSE"],
                "PSNR": metrics["PSNR"],
                "UQI": metrics["UQI"],
                "VIFP": metrics["VIF"],
            })

# Process train and test datasets
process_dataset("dataset/town7_dataset/train/", "plots/lime_masking_train_results.csv")
process_dataset("dataset/town7_dataset/test/", "plots/lime_masking_test_results.csv")
