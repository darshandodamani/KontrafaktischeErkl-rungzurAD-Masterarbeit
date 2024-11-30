from sympy import im
import torch
import os
import numpy as np
import pandas as pd
from torchvision.transforms import transforms
from PIL import Image
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels

# Paths to models
encoder_path = "model/epochs_500_latent_128/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128/classifier_final.pth"

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(
    device
)  # Example latent dims
decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
classifier = Classifier().to(device)

try:
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    logging.info(f"Encoder model loaded successfully from {encoder_path}")
except Exception as e:
    logging.error(f"Error loading encoder model from {encoder_path}: {e}")
    raise

try:
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    logging.info(f"Decoder model loaded successfully from {decoder_path}")
except Exception as e:
    logging.error(f"Error loading decoder model from {decoder_path}: {e}")
    raise

try:
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))
    logging.info(f"Classifier model loaded successfully from {classifier_path}")
except Exception as e:
    logging.error(f"Error loading classifier model from {classifier_path}: {e}")
    raise

encoder.eval()
decoder.eval()
classifier.eval()

# Function to preprocess the image (same as used in the main code)
def preprocess_image(image_path, device):
    logging.info(f"Preprocessing image: {image_path}")
    
    # Load and convert the image to RGB
    try:
        image = Image.open(image_path).convert("RGB")
        logging.debug(f"Original image size: {image.size}")
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        raise
    
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((80, 160)),
            transforms.ToTensor(),
        ]
    )
    
    # Apply transformations
    try:
        transformed_image = transform(image)
        logging.debug(f"Transformed image shape: {transformed_image.shape}")
    except Exception as e:
        logging.error(f"Error transforming image {image_path}: {e}")
        raise
    
    # Add batch dimension and move to the specified device
    transformed_image = transformed_image.unsqueeze(0).to(device)
    logging.info(f"Image preprocessing completed. Final shape: {transformed_image.shape}")
    
    return transformed_image

# Function to compute dataset-wide median for each latent feature
def compute_dataset_medians(dataset_paths, encoder, device):
    logging.info(f"Starting computation of dataset-wide medians for datasets: {dataset_paths}")
    all_latent_vectors = []
    all_image_names = []
    total_images_count = 0

    # Ensure the results directory exists
    results_dir = "latent_vectors"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logging.info(f"Created directory for latent vectors: {results_dir}")

    # Track the number of processed and skipped images
    total_processed_count = 0
    total_skipped_count = 0

    for dataset_path in dataset_paths:
        logging.info(f"Processing dataset: {dataset_path}")
        processed_count = 0
        skipped_count = 0

        # Count the total number of images in the dataset
        dataset_images_count = len([f for f in os.listdir(dataset_path) if f.endswith(('.png'))])
        total_images_count += dataset_images_count

        # Loop through the dataset and compute latent vectors for each image
        for image_filename in os.listdir(dataset_path):
            image_path = os.path.join(dataset_path, image_filename)

            if image_path.endswith(('.png')): 
                logging.info(f"Processing image: {image_filename}")
                
                # Preprocess the image
                try:
                    image = preprocess_image(image_path, device)
                    processed_count += 1
                    total_processed_count += 1
                except Exception as e:
                    logging.error(f"Skipping image {image_filename} due to preprocessing error: {e}")
                    skipped_count += 1
                    total_skipped_count += 1
                    continue

                # Get latent vector from the encoder
                try:
                    with torch.no_grad():
                        latent_vector = encoder(image)[2].cpu().numpy()
                        logging.debug(f"Latent vector for {image_filename}: {latent_vector}")
                        all_latent_vectors.append(latent_vector)
                        all_image_names.append(image_filename)

                        # Save latent vector to a file
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
    logging.info(f"Total images in both datasets: {total_images_count}")

    # Verify if the total number of processed and skipped images matches the total image count
    if total_processed_count + total_skipped_count != total_images_count:
        logging.warning("The total number of processed and skipped images does not match the total number of images in the datasets.")
    else:
        logging.info("The total number of processed and skipped images matches the total number of images in the datasets.")

    # Check if latent_vectors is empty
    if not all_latent_vectors:
        logging.warning("No latent vectors were computed. Please check the dataset paths or preprocessing steps.")
        # return None, None
        return np.array([]), {}

    # Stack all latent vectors into a numpy array
    # info: all latent vectors into a NumPy array (np.vstack()) and compute the median across all the vectors along the feature dimension (axis=0). This approach helps to get a median value for each feature.
    try:
        all_latent_vectors = np.vstack(all_latent_vectors)
        logging.info(f"Stacked latent vectors shape: {all_latent_vectors.shape}")
    except Exception as e:
        logging.error(f"Error stacking latent vectors: {e}")
        raise

    # Compute the median for each feature across the entire dataset
    try:
        median_values = np.median(all_latent_vectors, axis=0)
        mean_values = np.mean(all_latent_vectors, axis=0)
        min_values = np.min(all_latent_vectors, axis=0)
        max_values = np.max(all_latent_vectors, axis=0)
        std_dev_values = np.std(all_latent_vectors, axis=0)
        # Log statistics
        logging.info(f"Dataset-wide median values for each latent feature: {median_values}")
        logging.info(f"Dataset-wide mean values for each latent feature: {mean_values}")
        logging.info(f"Dataset-wide minimum values for each latent feature: {min_values}")
        logging.info(f"Dataset-wide maximum values for each latent feature: {max_values}")
        logging.info(f"Dataset-wide standard deviation for each latent feature: {std_dev_values}")
    except Exception as e:
        logging.error(f"Error computing dataset statistics: {e}")
        raise

    # Save the combined latent vectors and median values to a single CSV file
    try:
        # Define paths for each statistic CSV file
        median_values_csv = os.path.join(results_dir, "median_values.csv")
        mean_values_csv = os.path.join(results_dir, "mean_values.csv")
        min_values_csv = os.path.join(results_dir, "min_values.csv")
        max_values_csv = os.path.join(results_dir, "max_values.csv")
        std_dev_values_csv = os.path.join(results_dir, "std_dev_values.csv")

        # Save each statistic as a separate CSV file
        pd.DataFrame([median_values], columns=[f"feature_{i}" for i in range(len(median_values))]).to_csv(median_values_csv, index=False)
        pd.DataFrame([mean_values], columns=[f"feature_{i}" for i in range(len(mean_values))]).to_csv(mean_values_csv, index=False)
        pd.DataFrame([min_values], columns=[f"feature_{i}" for i in range(len(min_values))]).to_csv(min_values_csv, index=False)
        pd.DataFrame([max_values], columns=[f"feature_{i}" for i in range(len(max_values))]).to_csv(max_values_csv, index=False)
        pd.DataFrame([std_dev_values], columns=[f"feature_{i}" for i in range(len(std_dev_values))]).to_csv(std_dev_values_csv, index=False)

        # Log success
        logging.info(f"Saved median values to CSV file: {median_values_csv}")
        logging.info(f"Saved mean values to CSV file: {mean_values_csv}")
        logging.info(f"Saved minimum values to CSV file: {min_values_csv}")
        logging.info(f"Saved maximum values to CSV file: {max_values_csv}")
        logging.info(f"Saved standard deviation values to CSV file: {std_dev_values_csv}")
    except Exception as e:
        logging.error(f"Error saving statistics to CSV files: {e}")
        raise


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Encoder model is loaded above
    dataset_paths = ["dataset/town7_dataset/test/", "dataset/town7_dataset/train/"]
    try:
        median_values, latent_vectors_dict = compute_dataset_medians(dataset_paths, encoder, device)
        if median_values is not None:
            logging.info("Median computation completed successfully.")
            # Example of accessing individual latent vectors
            for image_name, latent_vector in latent_vectors_dict.items():
                logging.info(f"Latent vector for {image_name}: {latent_vector}")
    except Exception as e:
        logging.error(f"Error during median computation: {e}")