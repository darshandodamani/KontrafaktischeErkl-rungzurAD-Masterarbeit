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
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    logging.info(f"Encoder model loaded successfully from {encoder_path}")
except Exception as e:
    logging.error(f"Error loading encoder model from {encoder_path}: {e}")
    raise

try:
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    logging.info(f"Decoder model loaded successfully from {decoder_path}")
except Exception as e:
    logging.error(f"Error loading decoder model from {decoder_path}: {e}")
    raise

try:
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
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
            transforms.Resize((80, 160)),  # Assuming CARLA image size
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
def compute_dataset_medians(dataset_path, encoder, device):
    logging.info(f"Starting computation of dataset-wide medians for dataset at: {dataset_path}")
    latent_vectors = []
    image_names = []

    # Ensure the results directory exists
    results_dir = "latent_vectors"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        logging.info(f"Created directory for latent vectors: {results_dir}")

    # Loop through the dataset and compute latent vectors for each image
    for image_filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_filename)

        if image_path.endswith(".png"):  # Ensure we're processing only image files
            logging.info(f"Processing image: {image_filename}")
            
            # Preprocess the image
            try:
                image = preprocess_image(image_path, device)
            except Exception as e:
                logging.error(f"Skipping image {image_filename} due to preprocessing error: {e}")
                continue

            # Get latent vector from the encoder
            try:
                with torch.no_grad():
                    latent_vector = encoder(image)[2].cpu().numpy()
                    logging.debug(f"Latent vector for {image_filename}: {latent_vector}")
                    latent_vectors.append(latent_vector)
                    image_names.append(image_filename)

                    # Save latent vector to a file
                    latent_vector_filename = os.path.join(results_dir, f"latent_{image_filename}.npy")
                    np.save(latent_vector_filename, latent_vector)
                    logging.info(f"Saved latent vector for {image_filename} to {latent_vector_filename}")
            except Exception as e:
                logging.error(f"Error processing latent vector for image {image_filename}: {e}")
                continue

    # Check if latent_vectors is empty
    if not latent_vectors:
        logging.warning("No latent vectors were computed. Please check the dataset path or preprocessing steps.")
        return None

    # Stack all latent vectors into a numpy array
    try:
        latent_vectors = np.vstack(latent_vectors)
        logging.info(f"Stacked latent vectors shape: {latent_vectors.shape}")
    except Exception as e:
        logging.error(f"Error stacking latent vectors: {e}")
        raise

    # Compute the median for each feature across the entire dataset
    try:
        median_values = np.median(latent_vectors, axis=0)
        logging.info(f"Dataset-wide median values for each latent feature:\n {median_values}")
    except Exception as e:
        logging.error(f"Error computing median values: {e}")
        raise

    # Save the latent vectors and median values to a CSV file
    try:
        df = pd.DataFrame(latent_vectors, index=image_names)
        df.to_csv(os.path.join(results_dir, "latent_vectors.csv"))
        logging.info(f"Saved latent vectors to CSV file: {os.path.join(results_dir, 'latent_vectors.csv')}")

        median_df = pd.DataFrame([median_values], columns=[f"feature_{i}" for i in range(len(median_values))])
        median_df.to_csv(os.path.join(results_dir, "median_values.csv"), index=False)
        logging.info(f"Saved median values to CSV file: {os.path.join(results_dir, 'median_values.csv')}")
    except Exception as e:
        logging.error(f"Error saving latent vectors or median values to CSV: {e}")
        raise
    
    return median_values

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Encoder model is loaded above for both test and train dataset
    dataset_path = ["dataset/town7_dataset/train/", "dataset/town7_dataset/test/"]
    print("Computing median values for the dataset...")
    print("Total number of dataset: ", len(dataset_path))
    try:
        for dataset in dataset_path:
            median_values = compute_dataset_medians(dataset, encoder, device)
            if median_values is not None:
                logging.info("Median computation completed successfully.")
    except Exception as e:
        logging.error(f"Error during median computation: {e}")
