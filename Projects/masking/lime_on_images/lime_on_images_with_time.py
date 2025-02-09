#location: Projects/masking/lime_on_images/lime_on_images_with_time.py
import os
import sys
import time
import logging
from typing import Dict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
from typing import Tuple

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# Add Python Paths for Local Modules
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "autoencoder")))
from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))
from initialize_masking_pipeline import INITIAL_PREDICTIONS_CSV as initial_predictions_csv

# ------------------------------------------------------------------------------
# Paths and Constants
# ------------------------------------------------------------------------------
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_final.pth"
OUTPUT_CSV = "results/masking/lime_on_image_masking_results.csv"
TEST_DIR = "dataset/town7_dataset/test/"

# Directories for saving combined plots and individual images
PLOT_DIR = "plots/lime_on_image_masking"
LIME_ORIG_DIR = "plots/lime_on_image_original"
LIME_MASKED_REC_DIR = "plots/lime_on_image_masked_reconstructed"

for d in [PLOT_DIR, LIME_ORIG_DIR, LIME_MASKED_REC_DIR]:
    os.makedirs(d, exist_ok=True)

# (Optionally) define a target image size if desired – here we use the original size.
# If you want to resize images, modify the transformation below.
# For LIME, we work with the original image.
lime_transform = transforms.ToTensor()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
def load_models() -> Tuple[VariationalEncoder, Decoder, Classifier]:
    """
    Loads the encoder, decoder, and classifier models.
    """
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    classifier.eval()

    return encoder, decoder, classifier


# ------------------------------------------------------------------------------
# Image Quality Metrics Calculation
# ------------------------------------------------------------------------------
def calculate_image_metrics(original: torch.Tensor, modified: torch.Tensor) -> Dict[str, float]:
    """
    Computes SSIM, MSE, PSNR, UQI, and VIFP between two images.
    Expects image tensor values in the range [0, 1]. Converts images to uint8 [0,255] before metric computation.
    """
    original_np = original.cpu().squeeze().numpy().transpose(1, 2, 0)
    modified_np = modified.cpu().squeeze().numpy().transpose(1, 2, 0)

    # Scale to [0,255]
    original_np = (original_np * 255).astype(np.uint8)
    modified_np = (modified_np * 255).astype(np.uint8)

    metrics = {
        "SSIM": round(float(ssim(original_np, modified_np, channel_axis=-1, data_range=255)), 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np, data_range=255), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }
    return metrics


# ------------------------------------------------------------------------------
# LIME Classifier Prediction Function
# ------------------------------------------------------------------------------
def classifier_prediction(image_np: np.ndarray) -> np.ndarray:
    """
    Prediction function for LIME. Converts a numpy image (H, W, C) to tensor, passes it through
    the encoder and classifier, and returns prediction probabilities.
    """
    # Rearrange to (N, C, H, W)
    image_tensor = torch.tensor(image_np.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
    with torch.no_grad():
        latent_vector = encoder(image_tensor)[2]
        prediction = classifier(latent_vector)
        probabilities = F.softmax(prediction, dim=1).cpu().detach().numpy()
    return np.nan_to_num(probabilities)


# ------------------------------------------------------------------------------
# Apply LIME Mask
# ------------------------------------------------------------------------------
def apply_lime_mask(image: Image.Image, mask: np.ndarray, mask_value: float = 0) -> torch.Tensor:
    """
    Applies a LIME mask to an image.
    The mask is applied on the normalized image (range [0,1]) and then the result is converted to tensor.
    """
    image_np = np.array(image, dtype=np.float32) / 255.0
    # Set masked pixels to mask_value
    image_np[mask > 0] = mask_value
    image_np = np.clip(image_np, 0, 1)
    return transforms.ToTensor()(Image.fromarray((image_np * 255).astype(np.uint8))).unsqueeze(0).to(device)


# ------------------------------------------------------------------------------
# Plot and Save Combined Image Comparison
# ------------------------------------------------------------------------------
def plot_and_save_images(input_image: torch.Tensor,
                         reconstructed_image: torch.Tensor,
                         masked_image: torch.Tensor,
                         reconstructed_masked_image: torch.Tensor,
                         filename: str) -> None:
    """
    Saves a 4-panel plot comparing:
      1. The original image.
      2. The reconstructed image from the original.
      3. The masked image (after LIME masking).
      4. The reconstructed masked image.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    input_h, input_w = input_image.shape[2], input_image.shape[3]
    reconstructed_resized = F.interpolate(reconstructed_image, size=(input_h, input_w), mode='bilinear', align_corners=False)
    reconstructed_masked_resized = F.interpolate(reconstructed_masked_image, size=(input_h, input_w), mode='bilinear', align_corners=False)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(input_image.cpu().squeeze().permute(1, 2, 0).numpy())
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(reconstructed_resized.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")

    axs[2].imshow(masked_image.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[2].set_title("Masked Image")
    axs[2].axis("off")

    axs[3].imshow(reconstructed_masked_resized.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[3].set_title("Reconstructed Masked Image")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    logging.info(f"Saved combined plot to: {filename}")


# ------------------------------------------------------------------------------
# Save Individual Images
# ------------------------------------------------------------------------------
def save_individual_images(original_pil: Image.Image, reconstructed_masked_tensor: torch.Tensor, base_filename: str) -> None:
    """
    Saves the original image and the reconstructed masked image individually
    to designated folders.
    """
    # Save original image in LIME_ORIG_DIR
    original_save_path = os.path.join(LIME_ORIG_DIR, f"{base_filename}_original.png")
    original_pil.save(original_save_path)
    logging.info(f"Saved original image to: {original_save_path}")

    # Resize the reconstructed masked image to the original image's size.
    original_size = (original_pil.height, original_pil.width)
    reconstructed_resized = F.interpolate(reconstructed_masked_tensor, size=original_size, mode='bilinear', align_corners=False)
    reconstructed_pil = to_pil_image(reconstructed_resized.cpu().squeeze(0))
    masked_rec_save_path = os.path.join(LIME_MASKED_REC_DIR, f"{base_filename}_masked_reconstructed.png")
    reconstructed_pil.save(masked_rec_save_path)
    logging.info(f"Saved reconstructed masked image to: {masked_rec_save_path}")


# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_lime_on_image_masking() -> None:
    """
    Runs LIME-based masking on each test image. For each image, it:
      - Generates a LIME explanation and obtains a mask.
      - Applies the mask to obtain a masked image.
      - Runs the classifier on the masked image.
      - (If the prediction changes) computes image quality metrics.
      - Updates the results CSV immediately.
      - Saves a combined comparison plot.
      - Saves the original image and the reconstructed masked image individually.
    """
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(OUTPUT_CSV)
    global encoder, decoder, classifier
    encoder, decoder, classifier = load_models()

    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(TEST_DIR, image_filename)
        base_filename, _ = os.path.splitext(image_filename)

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue

        # Convert the image to tensor (without resizing) for later metric computation.
        input_image = transforms.ToTensor()(pil_image).unsqueeze(0).to(device)

        # Default values.
        counterfactual_found = False
        final_prediction = row["Prediction (Before Masking)"]
        confidence_final_str = "N/A"
        metrics = {"SSIM": "", "MSE": "", "PSNR": "", "UQI": "", "VIFP": ""}

        # Generate LIME explanation.
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(pil_image), classifier_prediction, hide_color=0, num_samples=1000
        )
        # Choose label based on the original prediction.
        label = 0 if final_prediction == "STOP" else 1
        temp, mask = explanation.get_image_and_mask(
            label=label, positive_only=True, num_features=10, hide_rest=False
        )

        # Apply LIME mask.
        masked_image = apply_lime_mask(pil_image, mask)

        # Run classifier on masked image.
        with torch.no_grad():
            latent_vector_masked = encoder(masked_image)[2]
            masked_prediction = classifier(latent_vector_masked)
            confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
            confidence_final_str = ", ".join(map(str, confidence_final))
            final_prediction = "STOP" if torch.argmax(masked_prediction, dim=1).item() == 0 else "GO"

        # Determine if a counterfactual is found.
        counterfactual_found = final_prediction != row["Prediction (Before Masking)"]

        # If counterfactual is found, compute quality metrics.
        if counterfactual_found:
            metrics = calculate_image_metrics(input_image, masked_image)

            # Reconstruct the masked image.
            with torch.no_grad():
                reconstructed_masked_image = decoder(latent_vector_masked)
            # (Optionally, you could also reconstruct the original image.)
            # Save combined comparison plot.
            plot_filename = os.path.join(PLOT_DIR, f"{base_filename}_LIME.png")
            # For visualization, here we assume "reconstructed_image" from the original is not used.
            # You can add original reconstruction if desired.
            plot_and_save_images(input_image, input_image, masked_image, reconstructed_masked_image, plot_filename)

            # Save individual images.
            save_individual_images(pil_image, reconstructed_masked_image, base_filename)

        total_time_taken = round(time.time() - start_time + float(row["Time Taken (s)"]), 5)

        # Update CSV for this image.
        df_results.loc[df_results['Image File'] == image_filename, [
            "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
            "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ]] = [
            final_prediction,
            confidence_final_str,
            counterfactual_found,
            "Superpixel",  # using a placeholder since LIME works with superpixels
            "N/A",
            metrics["SSIM"],
            metrics["MSE"],
            metrics["PSNR"],
            metrics["UQI"],
            metrics["VIFP"],
            total_time_taken
        ]

        df_results.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Updated CSV for image {image_filename}: CE Found = {counterfactual_found}, Time Taken = {round(time.time()-start_time,5)}s")

    logging.info(f"LIME-based masking results saved to {OUTPUT_CSV}")


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    process_lime_on_image_masking()
