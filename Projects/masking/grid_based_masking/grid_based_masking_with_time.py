#location: Projects/masking/grid_based_masking/grid_based_masking_with_time.py
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import matplotlib.pyplot as plt
import logging
from torchvision.transforms.functional import to_pil_image

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
# Constants and Paths
# ------------------------------------------------------------------------------
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_final.pth"
OUTPUT_CSV = "results/masking/grid_based_masking_results.csv"
TEST_DIR = "dataset/town7_dataset/test/"

# Directories for saving plots and individual images
PLOT_DIR = "plots/grid_based_masking_images"
GRID_BASED_ORIGINAL_DIR = "plots/grid_based_original"
GRID_BASED_MASKED_RECONSTRUCTED_DIR = "plots/grid_based_masked_reconstructed"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(GRID_BASED_ORIGINAL_DIR, exist_ok=True)
os.makedirs(GRID_BASED_MASKED_RECONSTRUCTED_DIR, exist_ok=True)

IMAGE_HEIGHT, IMAGE_WIDTH = 80, 160

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
def load_models():
    """
    Load the encoder, decoder, and classifier models.
    
    Returns:
        tuple: (encoder, decoder, classifier) all set to evaluation mode.
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
# Utility Functions
# ------------------------------------------------------------------------------
def calculate_image_metrics(original_image: torch.Tensor, modified_image: torch.Tensor) -> dict:
    """
    Calculate image quality metrics between the original and modified images.

    Args:
        original_image (torch.Tensor): Original image tensor.
        modified_image (torch.Tensor): Modified image tensor.

    Returns:
        dict: Dictionary containing SSIM, MSE, PSNR, UQI, and VIFP metrics.
    """
    original_np = original_image.cpu().squeeze().numpy().transpose(1, 2, 0)
    modified_np = modified_image.cpu().squeeze().numpy().transpose(1, 2, 0)
    original_np = (original_np * 255).astype(np.uint8)
    modified_np = (modified_np * 255).astype(np.uint8)
    
    metrics = {
        "SSIM": round(float(ssim(original_np, modified_np, channel_axis=-1, full=True)[0]), 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }
    return metrics

def plot_and_save_images(input_image: torch.Tensor, 
                         reconstructed_image: torch.Tensor, 
                         masked_image: torch.Tensor, 
                         reconstructed_masked_image: torch.Tensor, 
                         filename: str) -> None:
    """
    Save a plot of the input image, reconstructed image, grid-masked image, 
    and reconstructed grid-masked image.

    Args:
        input_image (torch.Tensor): Original input image.
        reconstructed_image (torch.Tensor): Reconstructed image from the encoder-decoder.
        masked_image (torch.Tensor): Image after applying grid-based masking.
        reconstructed_masked_image (torch.Tensor): Reconstructed image from the masked image.
        filename (str): Path to save the plot.
    """
    # Convert tensors to numpy arrays for visualization
    input_np = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    masked_np = masked_image.cpu().squeeze().permute(1, 2, 0).numpy()

    # Resize reconstructed images to match the input image size
    input_size = input_image.size()[2:]  # (Height, Width)
    reconstructed_np = F.interpolate(reconstructed_image, size=input_size, mode="bilinear", align_corners=False).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    reconstructed_masked_np = F.interpolate(reconstructed_masked_image, size=input_size, mode="bilinear", align_corners=False).cpu().squeeze().permute(1, 2, 0).detach().numpy()

    # Create a subplot with 4 images
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(input_np)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(reconstructed_np)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")

    axs[2].imshow(masked_np)
    axs[2].set_title("Grid-Masked Image")
    axs[2].axis("off")

    axs[3].imshow(reconstructed_masked_np)
    axs[3].set_title("Reconstructed Masked Image")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    logging.info(f"Saved plot to: {filename}")

def apply_grid_mask(image: torch.Tensor, grid_size: tuple, pos: int) -> torch.Tensor:
    """
    Apply a grid-based mask to an image tensor.

    Args:
        image (torch.Tensor): Input image tensor of shape [1, C, H, W].
        grid_size (tuple): Tuple (num_rows, num_cols) representing grid divisions.
        pos (int): Position index of the grid cell to mask.

    Returns:
        torch.Tensor: Masked image tensor.
    """
    num_rows, num_cols = grid_size
    masked_image = image.clone()
    
    row_idx = pos // num_cols
    col_idx = pos % num_cols
    row_start = row_idx * (IMAGE_HEIGHT // num_rows)
    row_end = (row_idx + 1) * (IMAGE_HEIGHT // num_rows)
    col_start = col_idx * (IMAGE_WIDTH // num_cols)
    col_end = (col_idx + 1) * (IMAGE_WIDTH // num_cols)
    
    masked_image[:, :, row_start:row_end, col_start:col_end] = 0
    return masked_image

# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def process_grid_based_masking() -> None:
    """
    Process images with grid-based masking to find counterfactuals and update the results CSV.
    Also saves individual original and reconstructed masked images.
    """
    # Load initial predictions and results CSV
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(OUTPUT_CSV)
    
    encoder, decoder, classifier = load_models()

    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(TEST_DIR, image_filename)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            continue
        
        input_image = transform(image).unsqueeze(0).to(device)
        
        counterfactual_found = False
        original_prediction = row["Prediction (Before Masking)"]
        final_prediction = original_prediction
        grid_size_found = None
        grid_position_found = None
        confidence_final = None
        metrics = {}
        
        # Use no_grad() for inference
        with torch.no_grad():
            for grid_size in [(10, 5), (4, 2)]:  # Try multiple grid sizes
                num_positions = grid_size[0] * grid_size[1]
                for pos in range(num_positions):
                    masked_image = apply_grid_mask(input_image, grid_size, pos)
                    
                    latent_vector_masked = encoder(masked_image)[2]
                    masked_prediction = classifier(latent_vector_masked)
                    confidence_final_tensor = F.softmax(masked_prediction, dim=1)
                    confidence_final = confidence_final_tensor.cpu().detach().numpy().flatten()
                    confidence_final_str = ", ".join(map(str, confidence_final))
                    predicted_label_after_masking = torch.argmax(masked_prediction, dim=1).item()
                    final_prediction = "STOP" if predicted_label_after_masking == 0 else "GO"
                    
                    if final_prediction != original_prediction:
                        counterfactual_found = True
                        grid_size_found = f"{grid_size}"
                        grid_position_found = pos
                        metrics = calculate_image_metrics(input_image, masked_image)
                        
                        # Reconstruct images for visualization
                        latent_vector_original = encoder(input_image)[2]
                        reconstructed_image = decoder(latent_vector_original)
                        reconstructed_masked_image = decoder(latent_vector_masked)
                        
                        # Save the combined plot image
                        base_filename, _ = os.path.splitext(image_filename)
                        plot_filename = os.path.join(PLOT_DIR, f"{base_filename}_grid_{grid_size[0]}x{grid_size[1]}_pos_{pos}.png")
                        plot_and_save_images(input_image, reconstructed_image, masked_image, reconstructed_masked_image, plot_filename)
                        
                        # Save individual images
                        original_save_path = os.path.join(GRID_BASED_ORIGINAL_DIR, f"{base_filename}_grid_{grid_size[0]}x{grid_size[1]}_pos_{pos}.png")
                        masked_reconstructed_save_path = os.path.join(GRID_BASED_MASKED_RECONSTRUCTED_DIR, f"{base_filename}_grid_{grid_size[0]}x{grid_size[1]}_pos_{pos}.png")
                        
                        # Convert tensors to PIL images for saving
                        original_pil = to_pil_image(input_image.squeeze(0).cpu())
                        masked_reconstructed_pil = to_pil_image(F.interpolate(reconstructed_masked_image, size=(IMAGE_HEIGHT, IMAGE_WIDTH), mode="bilinear", align_corners=False).cpu().squeeze(0))
                        
                        original_pil.save(original_save_path)
                        masked_reconstructed_pil.save(masked_reconstructed_save_path)
                        
                        logging.info(f"Saved original image to: {original_save_path}")
                        logging.info(f"Saved reconstructed masked image to: {masked_reconstructed_save_path}")
                        
                        break  # Found a counterfactual: break inner loop
                if counterfactual_found:
                    break

        end_time = time.time()
        # Update the time: add the new time to the previous value if it exists.
        previous_time = row.get("Time Taken (s)", 0)
        total_time_taken = round(end_time - start_time + float(previous_time), 5)
        
        # Update the DataFrame with the results for this image
        df_results.loc[df_results['Image File'] == image_filename, [
            "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found", 
            "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ]] = [
            final_prediction,
            confidence_final_str if confidence_final is not None else None,
            counterfactual_found,
            grid_size_found,
            grid_position_found,
            metrics.get("SSIM"),
            metrics.get("MSE"),
            metrics.get("PSNR"),
            metrics.get("UQI"),
            metrics.get("VIFP"),
            total_time_taken
        ]
        
        # Write out the CSV file after each image is processed
        df_results.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Updated CSV for image {image_filename}")

    logging.info(f"Grid-based masking results saved to {OUTPUT_CSV}")

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    process_grid_based_masking()
