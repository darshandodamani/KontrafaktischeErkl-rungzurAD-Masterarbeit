#loaction: Projects/masking/lime_on_latent_features/latent_statistics_calculator.py
"""
latent_statistics_calculator.py

This script computes dataset-wide latent feature statistics using a pretrained encoder.
It processes images from one or more dataset directories, extracts latent vectors,
and computes statistics (median, mean, min, max, and standard deviation) across all images.
Individual latent vectors are saved to disk, and the computed statistics are saved as CSV files.
"""

import os
import sys
import time
import logging
from typing import List, Tuple, Dict

import torch
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image

# ------------------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# Add Python Path for Local Modules
# ------------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels  # Imported if needed

# ------------------------------------------------------------------------------
# Model Paths and Device Configuration
# ------------------------------------------------------------------------------
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_final.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------------------------
def load_models() -> Tuple[VariationalEncoder, Decoder, Classifier]:
    """
    Load the encoder, decoder, and classifier models onto the specified device.
    """
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    try:
        encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
        logging.info(f"Encoder model loaded successfully from {ENCODER_PATH}")
    except Exception as e:
        logging.error(f"Error loading encoder model from {ENCODER_PATH}: {e}")
        raise

    try:
        decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device, weights_only=True))
        logging.info(f"Decoder model loaded successfully from {DECODER_PATH}")
    except Exception as e:
        logging.error(f"Error loading decoder model from {DECODER_PATH}: {e}")
        raise

    try:
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))
        logging.info(f"Classifier model loaded successfully from {CLASSIFIER_PATH}")
    except Exception as e:
        logging.error(f"Error loading classifier model from {CLASSIFIER_PATH}: {e}")
        raise

    encoder.eval()
    decoder.eval()
    classifier.eval()

    return encoder, decoder, classifier


# Load the models
encoder, decoder, classifier = load_models()


# ------------------------------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------------------------------
def preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Load an image from the given path, apply resizing and normalization,
    and return a tensor with a batch dimension.
    """
    logging.info(f"Preprocessing image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        logging.debug(f"Original image size: {image.size}")
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        raise

    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    try:
        transformed_image = transform(image)
        logging.debug(f"Transformed image shape: {transformed_image.shape}")
    except Exception as e:
        logging.error(f"Error transforming image {image_path}: {e}")
        raise

    transformed_image = transformed_image.unsqueeze(0).to(device)
    logging.info(f"Image preprocessing completed. Final shape: {transformed_image.shape}")
    return transformed_image


# ------------------------------------------------------------------------------
# Dataset Latent Statistics Computation
# ------------------------------------------------------------------------------
def compute_dataset_medians(
    dataset_paths: List[str],
    encoder: VariationalEncoder,
    device: torch.device
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute dataset-wide latent feature statistics (median, mean, min, max, and standard deviation)
    for images from the given dataset paths. Latent vectors are saved individually, and statistics
    are saved as CSV files.

    Returns:
        A tuple containing:
         - median_values: np.ndarray of median values for each latent feature.
         - latent_vectors_dict: Dictionary mapping image filenames to their latent vector (np.ndarray).
    """
    logging.info(f"Starting computation of dataset-wide medians for datasets: {dataset_paths}")
    all_latent_vectors = []
    all_image_names = []
    total_images_count = 0

    results_dir = "latent_vectors"
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Using latent vectors directory: {results_dir}")

    total_processed_count = 0
    total_skipped_count = 0

    for dataset_path in dataset_paths:
        logging.info(f"Processing dataset: {dataset_path}")
        processed_count = 0
        skipped_count = 0

        # List only .png files
        image_files = [f for f in os.listdir(dataset_path) if f.endswith('.png')]
        dataset_images_count = len(image_files)
        total_images_count += dataset_images_count

        for image_filename in image_files:
            image_path = os.path.join(dataset_path, image_filename)
            logging.info(f"Processing image: {image_filename}")

            try:
                image_tensor = preprocess_image(image_path, device)
                processed_count += 1
                total_processed_count += 1
            except Exception as e:
                logging.error(f"Skipping image {image_filename} due to preprocessing error: {e}")
                skipped_count += 1
                total_skipped_count += 1
                continue

            try:
                with torch.no_grad():
                    latent_vector = encoder(image_tensor)[2].cpu().numpy()  # latent vector assumed at index 2
                logging.debug(f"Latent vector for {image_filename}: {latent_vector}")
                all_latent_vectors.append(latent_vector)
                all_image_names.append(image_filename)

                latent_vector_filename = os.path.join(results_dir, f"latent_{image_filename}.npy")
                np.save(latent_vector_filename, latent_vector)
                logging.info(f"Saved latent vector for {image_filename} to {latent_vector_filename}")
            except Exception as e:
                logging.error(f"Error processing latent vector for image {image_filename}: {e}")
                skipped_count += 1
                total_skipped_count += 1
                continue

        logging.info(f"Total images processed in {dataset_path}: {processed_count}")
        logging.info(f"Total images skipped in {dataset_path}: {skipped_count}")

    logging.info(f"Total images processed: {total_processed_count}")
    logging.info(f"Total images skipped: {total_skipped_count}")
    logging.info(f"Total images in all datasets: {total_images_count}")

    if total_processed_count + total_skipped_count != total_images_count:
        logging.warning("Mismatch in total image count (processed + skipped != total images)")
    else:
        logging.info("Processed and skipped image counts match the total number of images.")

    if not all_latent_vectors:
        logging.warning("No latent vectors computed. Check dataset paths or preprocessing.")
        return np.array([]), {}

    # Create a dictionary mapping image names to latent vectors
    latent_vectors_dict = {name: vec for name, vec in zip(all_image_names, all_latent_vectors)}

    try:
        all_latent_vectors_stacked = np.vstack(all_latent_vectors)
        logging.info(f"Stacked latent vectors shape: {all_latent_vectors_stacked.shape}")
    except Exception as e:
        logging.error(f"Error stacking latent vectors: {e}")
        raise

    try:
        median_values = np.median(all_latent_vectors_stacked, axis=0)
        mean_values = np.mean(all_latent_vectors_stacked, axis=0)
        min_values = np.min(all_latent_vectors_stacked, axis=0)
        max_values = np.max(all_latent_vectors_stacked, axis=0)
        std_dev_values = np.std(all_latent_vectors_stacked, axis=0)
        logging.info(f"Dataset-wide median values: {median_values}")
        logging.info(f"Dataset-wide mean values: {mean_values}")
        logging.info(f"Dataset-wide min values: {min_values}")
        logging.info(f"Dataset-wide max values: {max_values}")
        logging.info(f"Dataset-wide standard deviation: {std_dev_values}")
    except Exception as e:
        logging.error(f"Error computing dataset statistics: {e}")
        raise

    try:
        # Define feature names and save statistics as CSV files
        feature_names = [f"feature_{i}" for i in range(len(median_values))]
        pd.DataFrame([median_values], columns=feature_names).to_csv(os.path.join(results_dir, "median_values.csv"), index=False)
        pd.DataFrame([mean_values], columns=feature_names).to_csv(os.path.join(results_dir, "mean_values.csv"), index=False)
        pd.DataFrame([min_values], columns=feature_names).to_csv(os.path.join(results_dir, "min_values.csv"), index=False)
        pd.DataFrame([max_values], columns=feature_names).to_csv(os.path.join(results_dir, "max_values.csv"), index=False)
        pd.DataFrame([std_dev_values], columns=feature_names).to_csv(os.path.join(results_dir, "std_dev_values.csv"), index=False)
        logging.info("Saved dataset statistics CSV files successfully.")
    except Exception as e:
        logging.error(f"Error saving statistics to CSV files: {e}")
        raise

    return median_values, latent_vectors_dict


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Define dataset directories (adjust paths as necessary)
    dataset_paths = ["dataset/town7_dataset/test/", "dataset/town7_dataset/train/"]
    try:
        median_values, latent_vectors_dict = compute_dataset_medians(dataset_paths, encoder, device)
        if median_values.size > 0:
            logging.info("Median computation completed successfully.")
            # Log an example latent vector for each image
            for image_name, latent_vector in latent_vectors_dict.items():
                logging.info(f"Latent vector for {image_name}: {latent_vector}")
    except Exception as e:
        logging.error(f"Error during median computation: {e}")

    # Final message indicating where the values are saved.
    save_dir = os.path.abspath("latent_vectors")
    final_message = f"All latent vectors and dataset statistics have been saved in the directory: {save_dir}"
    logging.info(final_message)
    print(final_message)
