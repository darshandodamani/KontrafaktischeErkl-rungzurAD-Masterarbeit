#location: Projects/masking/lime_on_latent_features/lime_based_latent_vector_masking.py
"""
lime_based_latent_vector_masking.py

This script performs LIME-based masking in the latent feature space.
It loads an image, encodes it to obtain latent features, uses a tabular LIME explainer
to determine important latent features, and then replaces those features with median values.
When a counterfactual (changed prediction) is found, it computes image similarity metrics,
saves a comparison plot, saves the original image and the reconstructed masked image separately,
updates a results CSV file immediately, and (for True cases) appends replacement info 
(image filename, indices of replaced latent features, original values, and median values)
to a separate CSV file for later evaluation.
"""

import os
import sys
import time
import logging
import csv
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import Tuple

# ------------------------------------------------------------------------------
# Setup Python Paths for local modules
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
from initialize_masking_pipeline import INITIAL_PREDICTIONS_CSV as initial_predictions_csv

# ------------------------------------------------------------------------------
# Configuration and Paths
# ------------------------------------------------------------------------------
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"
median_values_csv = "latent_vectors/combined_median_values.csv"

output_csv = "results/masking/lime_on_latent_masking_results.csv"
replacements_csv = "results/masking/lime_on_latent_replacements.csv"
test_dir = "dataset/town7_dataset/test/"
plot_dir = "plots/lime_on_latent"
os.makedirs(plot_dir, exist_ok=True)

# CSV Headers (for main results)
csv_headers = [
    "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
    "Prediction (After Masking)", "Confidence (After Masking)",
    "Counterfactual Found", "Feature Selection (%)", "Selected Features",
    "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
def load_models() -> Tuple[VariationalEncoder, Decoder, Classifier]:
    """
    Loads the encoder, decoder, and classifier models onto the specified device.
    """
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    classifier.eval()

    logging.info("Models loaded and set to evaluation mode.")
    return encoder, decoder, classifier

# Load the models
encoder, decoder, classifier = load_models()

# ------------------------------------------------------------------------------
# Load Auxiliary Data
# ------------------------------------------------------------------------------
# Load median values as a 1D array
median_values = pd.read_csv(median_values_csv).values.flatten()

# Load initial predictions and existing results CSV
df_initial = pd.read_csv(initial_predictions_csv)
df_results = pd.read_csv(output_csv)

# ------------------------------------------------------------------------------
# Transformations
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor()
])

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def calculate_image_metrics(original: np.ndarray, modified: np.ndarray) -> Dict[str, float]:
    """
    Calculates image similarity metrics (SSIM, MSE, PSNR, UQI, and VIFP) between two images.
    If dimensions differ, the modified image is resized to match the original.
    """
    if original.shape != modified.shape:
        modified = cv2.resize(modified, (original.shape[1], original.shape[0]))
    metrics = {
        "SSIM": round(ssim(original, modified, data_range=1.0, channel_axis=-1), 5),
        "MSE": round(mse(original, modified), 5),
        "PSNR": round(psnr(original, modified, data_range=1.0), 5),
        "UQI": round(uqi(original, modified), 5),
        "VIFP": round(vifp(original, modified), 5),
    }
    return metrics

def plot_and_save_images(original_image: np.ndarray, reconstructed_image: np.ndarray,
                         masked_reconstructed_image: np.ndarray, filename: str) -> None:
    """
    Saves a 3-panel comparison plot showing:
      1. The original image.
      2. The reconstructed image.
      3. The masked reconstructed image.
    """
    reconstructed_resized = cv2.resize(reconstructed_image, (original_image.shape[1], original_image.shape[0]))
    masked_reconstructed_resized = cv2.resize(masked_reconstructed_image, (original_image.shape[1], original_image.shape[0]))
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(reconstructed_resized)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")
    
    axs[2].imshow(masked_reconstructed_resized)
    axs[2].set_title("Masked Reconstructed Image")
    axs[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Plot saved to {filename}")

def predict_with_latent(latent: np.ndarray) -> np.ndarray:
    """
    Helper function for LIME: converts a latent vector (numpy array) to a tensor,
    passes it through the classifier, and returns the softmax probabilities.
    """
    latent_tensor = torch.tensor(latent, dtype=torch.float32).to(device)
    output = classifier(latent_tensor)
    return F.softmax(output, dim=1).cpu().detach().numpy()

def save_individual_images(original_image_np: np.ndarray, reconstructed_masked_np: np.ndarray, base_filename: str) -> None:
    """
    Saves the original image and the reconstructed masked image individually into designated directories.
    """
    orig_dir = "plots/lime_on_latent_original"
    masked_rec_dir = "plots/lime_on_latent_masked_reconstructed"
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(masked_rec_dir, exist_ok=True)
    
    # Convert numpy arrays to PIL images (assumes values in [0,1] or [0,255])
    if original_image_np.max() <= 1.0:
        original_disp = (original_image_np * 255).astype(np.uint8)
    else:
        original_disp = original_image_np.astype(np.uint8)
    if reconstructed_masked_np.max() <= 1.0:
        reconstructed_disp = (reconstructed_masked_np * 255).astype(np.uint8)
    else:
        reconstructed_disp = reconstructed_masked_np.astype(np.uint8)
    
    original_pil = Image.fromarray(original_disp)
    reconstructed_pil = Image.fromarray(reconstructed_disp)
    
    orig_save_path = os.path.join(orig_dir, f"{base_filename}_original.png")
    masked_save_path = os.path.join(masked_rec_dir, f"{base_filename}_masked_reconstructed.png")
    
    original_pil.save(orig_save_path)
    reconstructed_pil.save(masked_save_path)
    
    logging.info(f"Saved original image to: {orig_save_path}")
    logging.info(f"Saved reconstructed masked image to: {masked_save_path}")

def append_replacement_info(info: Dict[str, str]) -> None:
    """
    Appends a row of replacement information to the replacements CSV file.
    The info dictionary should contain:
      - "Image File"
      - "Replaced Feature Indices"
      - "Original Latent Values"
      - "Replaced (Median) Values"
    """
    file_exists = os.path.exists(replacements_csv)
    with open(replacements_csv, "a", newline="") as csvfile:
        fieldnames = ["Image File", "Replaced Feature Indices", "Original Latent Values", "Replaced (Median) Values"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(info)
    logging.info(f"Appended replacement info for {info['Image File']} to {os.path.abspath(replacements_csv)}")

# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_lime_on_latent_masking() -> None:
    """
    Runs LIME-based masking on each image from the initial predictions.
    For each image:
      - Loads and transforms the image.
      - Encodes it to obtain latent features.
      - Uses LIME on the latent vector to assess feature importance.
      - Gradually replaces important features with median values until a counterfactual is found.
      - Computes image metrics using the decoder's reconstruction from the masked latent vector.
      - Saves a combined comparison plot.
      - Saves the original image and the reconstructed masked image separately.
      - Updates the results CSV file immediately.
      - For True cases, immediately appends replacement info (image filename, indices, original values, replaced values)
        to a separate CSV file.
    """
    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(test_dir, image_filename)
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)
        
        # Default values
        counterfactual_found = False
        final_prediction = row["Prediction (Before Masking)"]
        predicted_class = 0 if final_prediction == "STOP" else 1
        confidence_final_str = "N/A"
        metrics: Dict[str, float] = {}
        selected_features: List[int] = []
        selected_features_str = "N/A"
        
        # Encode image to obtain latent vector
        latent_vector = encoder(input_image)[2]
        latent_vector_np = latent_vector.cpu().detach().numpy().reshape(1, -1)
        
        # Set up LIME explainer on latent features
        explainer = LimeTabularExplainer(
            latent_vector_np,
            mode="classification",
            feature_names=[f"latent_{i}" for i in range(latent_vector_np.shape[1])],
            discretize_continuous=False
        )
        explanation = explainer.explain_instance(
            latent_vector_np.flatten(), predict_with_latent, num_features=latent_vector_np.shape[1]
        )
        
        # Gradually mask important latent features based on LIME explanation
        percentage_value = 0.0
        step_size = 0.01
        positive_importance_list = sorted(
            [(feature, weight) for feature, weight in explanation.as_list() if weight > 0],
            key=lambda x: abs(x[1]), reverse=True
        )
        
        # Loop until a counterfactual is found or percentage reaches 100%
        while percentage_value < 1.0:
            percentage_value += step_size
            num_features_to_select = int(len(positive_importance_list) * percentage_value)
            selected_features = [int(feature.split("_")[-1]) for feature, _ in positive_importance_list[:num_features_to_select]]
            
            # Create a masked latent vector by replacing selected features with median values
            masked_latent_vector = latent_vector_np.flatten().copy()
            for feature_index in selected_features:
                masked_latent_vector[feature_index] = median_values[feature_index]
            
            masked_latent_tensor = torch.tensor(masked_latent_vector, dtype=torch.float32).to(device).reshape(1, -1)
            masked_prediction = classifier(masked_latent_tensor)
            confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
            confidence_final_str = ", ".join(map(str, confidence_final))
            predicted_label_after_masking = torch.argmax(masked_prediction, dim=1).item()
            final_prediction = "STOP" if predicted_label_after_masking == 0 else "GO"
            
            if predicted_label_after_masking != predicted_class:
                counterfactual_found = True
                selected_features_str = ", ".join(map(str, selected_features))
                
                # Compute image quality metrics using the decoder's reconstruction from the masked latent vector
                input_image_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                reconstructed_masked_image = decoder(masked_latent_tensor).squeeze(0)
                reconstructed_masked_image_np = reconstructed_masked_image.permute(1, 2, 0).cpu().detach().numpy()
                reconstructed_masked_image_np_resized = cv2.resize(
                    reconstructed_masked_image_np,
                    (input_image_np.shape[1], input_image_np.shape[0])
                )
                metrics = calculate_image_metrics(input_image_np, reconstructed_masked_image_np_resized)
                
                # Save a combined comparison plot (showing original, reconstructed, and masked reconstructed)
                plot_filename = os.path.join(plot_dir, f"{os.path.splitext(image_filename)[0]}_comparison.png")
                plot_and_save_images(input_image_np, reconstructed_masked_image_np_resized, reconstructed_masked_image_np_resized, plot_filename)
                
                # Save individual images: original and reconstructed masked image
                base_filename = os.path.splitext(image_filename)[0]
                save_individual_images(input_image_np, reconstructed_masked_image_np_resized, base_filename)
                
                # For True cases, record and immediately append the replaced latent feature info
                orig_latent = latent_vector_np.flatten()
                original_values = [orig_latent[i] for i in selected_features]
                replaced_values = [median_values[i] for i in selected_features]
                replacement_info = {
                    "Image File": image_filename,
                    "Replaced Feature Indices": selected_features_str,
                    "Original Latent Values": ", ".join(map(str, original_values)),
                    "Replaced (Median) Values": ", ".join(map(str, replaced_values))
                }
                append_replacement_info(replacement_info)
                break
            else:
                # If no counterfactual found, clear the fields
                confidence_final_str = "N/A"
                selected_features_str = "N/A"
                metrics = {"SSIM": np.nan, "MSE": np.nan, "PSNR": np.nan, "UQI": np.nan, "VIFP": np.nan}
        
        total_time_taken = round(time.time() - start_time, 5)
        
        # Update the results DataFrame for the current image
        df_results.loc[df_results['Image File'] == image_filename, [
            "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
            "Feature Selection (%)", "Selected Features",
            "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ]] = [
            final_prediction,
            confidence_final_str,
            bool(counterfactual_found),
            f"{percentage_value * 100:.2f}%" if counterfactual_found else np.nan,
            selected_features_str if counterfactual_found else np.nan,
            metrics.get("SSIM", np.nan),
            metrics.get("MSE", np.nan),
            metrics.get("PSNR", np.nan),
            metrics.get("UQI", np.nan),
            metrics.get("VIFP", np.nan),
            total_time_taken
        ]
        
        df_results.to_csv(output_csv, index=False)
        # Print terminal results for this image
        print(f"Image {image_filename}: Counterfactual Found = {counterfactual_found}. CSV updated.")
        logging.info(f"Updated CSV for image {image_filename}: Counterfactual Found = {counterfactual_found}, Time Taken = {total_time_taken}s")
    
    print(f"LIME on Latent Masking results saved to {os.path.abspath(output_csv)}")
    logging.info(f"LIME on Latent Masking results saved to {os.path.abspath(output_csv)}")

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    process_lime_on_latent_masking()
