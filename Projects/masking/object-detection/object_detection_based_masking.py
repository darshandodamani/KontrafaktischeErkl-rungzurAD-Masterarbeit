#location: Projects/masking/object-detection/object_detection_based_masking.py
import os
import sys
import time
import logging
import warnings
from typing import Tuple, Dict, List

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi

# ------------------------------------------------------------------------------
# Suppress specific FutureWarning from torch.utils.checkpoint
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.utils.checkpoint")

# Suppress specific FutureWarning from torch.cuda.amp.autocast
warnings.filterwarnings("ignore", category=FutureWarning, message="`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.")
# Use a regex pattern to catch any message mentioning torch.cuda.amp.autocast
warnings.filterwarnings("ignore", message=".*torch\\.cuda\\.amp\\.autocast.*", category=FutureWarning)

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# Add Python Paths for local modules
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
OUTPUT_CSV = "results/masking/object_detection_masking_results.csv"
TEST_DIR = "dataset/town7_dataset/test/"

# Directories for saving plots and individual images
PLOT_DIR = "plots/object_detection_masking_images"
OBJECT_DET_ORIG_DIR = "plots/object_detection_original"
OBJECT_DET_MASKED_REC_DIR = "plots/object_detection_masked_reconstructed"

for d in [PLOT_DIR, OBJECT_DET_ORIG_DIR, OBJECT_DET_MASKED_REC_DIR]:
    os.makedirs(d, exist_ok=True)

# Image dimensions (height, width)
IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
def load_models() -> Tuple[VariationalEncoder, Decoder, Classifier]:
    """
    Load encoder, decoder, and classifier models and set them to evaluation mode.
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
# Transformation Pipeline
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
])


# ------------------------------------------------------------------------------
# Metrics Calculation
# ------------------------------------------------------------------------------
def calculate_image_metrics(original_image: torch.Tensor, modified_image: torch.Tensor) -> Dict[str, float]:
    """
    Calculate similarity metrics between the original and modified images.
    """
    # Convert tensors to numpy arrays; expected values in [0, 1]
    original_np = original_image.cpu().squeeze().permute(1, 2, 0).numpy()
    modified_np = modified_image.cpu().squeeze().permute(1, 2, 0).detach().numpy()

    # When using full=True, the function returns (score, full_ssim_image).
    ssim_value = ssim(original_np, modified_np, channel_axis=-1, full=True, data_range=1.0)[0]
    metrics = {
        "SSIM": round(float(ssim_value), 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np, data_range=1.0), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }
    return metrics


# ------------------------------------------------------------------------------
# Plotting & Saving Combined Images
# ------------------------------------------------------------------------------
def plot_and_save_images(input_image: torch.Tensor,
                         reconstructed_image: torch.Tensor,
                         masked_image: torch.Tensor,
                         reconstructed_masked_image: torch.Tensor,
                         filename: str) -> None:
    """
    Save a combined plot of:
      - The original image,
      - Its reconstruction,
      - The masked image,
      - And the reconstruction of the masked image.
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
    Save the original image and the reconstructed masked image individually
    to designated folders.
    """
    # Save the original image.
    original_save_path = os.path.join(OBJECT_DET_ORIG_DIR, f"{base_filename}_original.png")
    original_pil.save(original_save_path)
    logging.info(f"Saved original image to: {original_save_path}")

    # Resize the reconstructed masked image to (IMAGE_HEIGHT, IMAGE_WIDTH) and convert to PIL.
    reconstructed_masked_resized = F.interpolate(reconstructed_masked_tensor, size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                                  mode="bilinear", align_corners=False)
    reconstructed_masked_pil = to_pil_image(reconstructed_masked_resized.cpu().squeeze(0))
    masked_rec_save_path = os.path.join(OBJECT_DET_MASKED_REC_DIR, f"{base_filename}_masked_reconstructed.png")
    reconstructed_masked_pil.save(masked_rec_save_path)
    logging.info(f"Saved reconstructed masked image to: {masked_rec_save_path}")


# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_object_detection_masking() -> None:
    """
    Run object detection-based masking on each test image, update the results CSV after each image,
    and save individual original and reconstructed masked images.
    """
    # Load CSV files with initial predictions and previous results.
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(OUTPUT_CSV)

    # Load our models: encoder, decoder, and classifier.
    encoder, decoder, classifier = load_models()

    # Load the YOLO model for object detection (loaded once).
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, autoshape=True)

    # Process each image.
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

        input_image = transform(pil_image).unsqueeze(0).to(device)

        # Default results
        counterfactual_found = False
        final_prediction = row["Prediction (Before Masking)"]
        bbox_dimensions = "N/A"
        bbox_position = "N/A"
        confidence_final_str = "N/A"
        metrics = {"SSIM": "", "MSE": "", "PSNR": "", "UQI": "", "VIFP": ""}
        objects_detected: List[str] = []

        with torch.no_grad():
            # Run YOLO object detection on the PIL image.
            results = yolo_model(pil_image)
            detections = results.xyxy[0]  # bounding boxes: [x_min, y_min, x_max, y_max, confidence, class]
            if detections.numel() > 0:
                # Get names for detected objects.
                objects_detected = [results.names[int(det[5])] for det in detections]
                # Process each detection.
                for det in detections:
                    x_min, y_min, x_max, y_max = map(int, det[:4])
                    bbox_dimensions = f"({x_max - x_min}, {y_max - y_min})"
                    bbox_position = f"({x_min}, {y_min})"

                    # Create masked image by zeroing out the detected bounding box region.
                    masked_image = input_image.clone()
                    masked_image[:, :, y_min:y_max, x_min:x_max] = 0

                    # Get prediction on the masked image.
                    latent_vector_masked = encoder(masked_image)[2]
                    masked_prediction = classifier(latent_vector_masked)
                    confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
                    confidence_final_str = ", ".join(map(str, confidence_final))
                    predicted_label_after_masking = torch.argmax(masked_prediction, dim=1).item()
                    final_prediction = "STOP" if predicted_label_after_masking == 0 else "GO"

                    # If a counterfactual is found, compute metrics and save images.
                    if final_prediction != row["Prediction (Before Masking)"]:
                        counterfactual_found = True
                        metrics = calculate_image_metrics(input_image, masked_image)

                        # Reconstruct the masked image using the decoder.
                        reconstructed_masked_image = decoder(latent_vector_masked)
                        # Also get the reconstruction for the original input (for combined plotting).
                        latent_vector_original = encoder(input_image)[2]
                        reconstructed_image = decoder(latent_vector_original)

                        # Save the combined plot image.
                        plot_filename = os.path.join(PLOT_DIR, f"{base_filename}_det_{x_min}_{y_min}_{x_max}_{y_max}.png")
                        plot_and_save_images(input_image, reconstructed_image, masked_image, reconstructed_masked_image, plot_filename)

                        # Save individual images.
                        save_individual_images(pil_image, reconstructed_masked_image, base_filename)
                        break  # Stop after the first counterfactual for this image.

        total_time_taken = round(time.time() - start_time + float(row["Time Taken (s)"]), 5)

        # Update the results DataFrame for the current image.
        df_results.loc[df_results['Image File'] == image_filename, [
            "Prediction (After Masking)",
            "Confidence (After Masking)",
            "Counterfactual Found",
            "Grid Size",            # Here storing bounding box dimensions.
            "Grid Position",        # Here storing bounding box position.
            "SSIM",
            "MSE",
            "PSNR",
            "UQI",
            "VIFP",
            "Objects Detected",
            "Time Taken (s)"
        ]] = [
            final_prediction,
            confidence_final_str,
            counterfactual_found,
            bbox_dimensions,
            bbox_position,
            metrics["SSIM"],
            metrics["MSE"],
            metrics["PSNR"],
            metrics["UQI"],
            metrics["VIFP"],
            ", ".join(objects_detected) if objects_detected else "None",
            total_time_taken
        ]

        # Write out the CSV file immediately after processing each image.
        df_results.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Updated CSV for image {image_filename}")

    logging.info(f"Object detection-based masking results saved to {OUTPUT_CSV}")


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    process_object_detection_masking()
