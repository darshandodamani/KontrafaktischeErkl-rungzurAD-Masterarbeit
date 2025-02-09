#location: Projects/masking/initialize_masking_pipeline.py
"""
common_predictions_initializer.py

This script computes initial classification predictions for each image in the test dataset,
saves these predictions to a common CSV file, and then initializes masking method-specific
results CSV files with appropriate headers and the initial prediction data.
"""

import os
import sys
import csv
import time
import logging
from typing import Dict, Any

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------------------------
# Add Python Path for Local Modules
# ------------------------------------------------------------------------------
# This appends the "autoencoder" directory (one level up) to sys.path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# ------------------------------------------------------------------------------
# Configuration and Paths
# ------------------------------------------------------------------------------
# Model paths
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
DECODER_PATH = "model/epochs_500_latent_128_town_7/decoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_final.pth"

# Data directories and CSV files
TEST_DIR = "dataset/town7_dataset/test/"
INITIAL_PREDICTIONS_CSV = "results/masking/common_initial_predictions.csv"

# Dictionary mapping method names to their results CSV file paths.
METHODS_RESULTS: Dict[str, str] = {
    "grid_based": "results/masking/grid_based_masking_results.csv",
    "lime_on_image": "results/masking/lime_on_image_masking_results.csv",
    "object_detection": "results/masking/object_detection_masking_results.csv",
    "lime_on_latent": "results/masking/lime_on_latent_masking_results.csv"
}

# Column headers for each method
METHODS_HEADERS: Dict[str, list] = {
    "grid_based": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
    ],
    "lime_on_image": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
    ],
    "object_detection": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Objects Detected",
        "Time Taken (s)"
    ],
    "lime_on_latent": [
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
        "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
        "Feature Selection (%)", "Selected Features", "SSIM", "MSE", "PSNR", "UQI", "VIFP",
        "Time Taken (s)"
    ]
}

# ------------------------------------------------------------------------------
# Device and Model Initialization
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load the models and set them to evaluation mode.
encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
classifier = Classifier().to(device)

encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device, weights_only=True))
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))

encoder.eval()
decoder.eval()
classifier.eval()
logging.info("Models loaded and set to evaluation mode.")

# ------------------------------------------------------------------------------
# Transformation
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def compute_initial_predictions(dataset_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Runs initial classification for each image in the dataset and saves the results to a CSV file.

    Args:
        dataset_dir (str): Directory containing the test images.
        output_csv (str): Path to the CSV file where initial predictions will be saved.

    Returns:
        pd.DataFrame: DataFrame with the initial predictions.
    """
    if os.path.exists(output_csv):
        print("Initial predictions file already exists. Skipping re-computation.")
        logging.info(f"Initial predictions CSV {output_csv} already exists.")
        return pd.read_csv(output_csv)
    
    print("Computing initial predictions for test dataset...")
    logging.info("Computing initial predictions for test dataset...")
    start_time = time.time()
    records = []
    
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)", "Time Taken (s)"
        ])
        
        for image_file in sorted(os.listdir(dataset_dir)):
            if not image_file.endswith(".png"):
                continue
            
            image_path = os.path.join(dataset_dir, image_file)
            image = Image.open(image_path).convert("RGB")
            input_image = transform(image).unsqueeze(0).to(device)
            
            image_start_time = time.time()
            with torch.no_grad():
                latent_vector = encoder(input_image)[2]
                prediction = classifier(latent_vector)
                predicted_label = torch.argmax(prediction, dim=1).item()
                predicted_class = "STOP" if predicted_label == 0 else "GO"
                confidence = F.softmax(prediction, dim=1).cpu().numpy().flatten()
            
            image_time_taken = round(time.time() - image_start_time, 5)
            records.append([
                image_file,
                predicted_class,
                list(map(lambda x: round(float(x), 5), confidence)),
                image_time_taken
            ])
    
    df = pd.DataFrame(records, columns=[
        "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)", "Time Taken (s)"
    ])
    df.to_csv(output_csv, index=False)
    logging.info(f"Initial predictions computed in {round(time.time()-start_time, 5)}s and saved to {output_csv}")
    print(f"Initial predictions saved to {output_csv}")
    return df

def initialize_method_results() -> None:
    """
    Ensures that each masking method has a results CSV file populated with the initial prediction data.
    """
    df_initial = compute_initial_predictions(TEST_DIR, INITIAL_PREDICTIONS_CSV)
    
    for method, method_csv in METHODS_RESULTS.items():
        if os.path.exists(method_csv):
            logging.info(f"Results CSV for {method} already exists. Skipping initialization.")
            continue  # Skip if file already exists
        
        df_method = df_initial.copy()
        # Add missing columns as empty strings
        for col in METHODS_HEADERS[method]:
            if col not in df_method.columns:
                df_method[col] = ""
        # Reorder columns
        df_method = df_method[METHODS_HEADERS[method]]
        df_method.to_csv(method_csv, index=False)
        print(f"Initialized {method_csv} with headers and initial prediction data.")
        logging.info(f"Initialized {method_csv} with initial predictions.")

def update_method_results(method: str, results: Dict[str, Any]) -> None:
    """
    Updates the CSV file for a specific masking method with computed results.
    
    Args:
        method (str): The key for the masking method (e.g., "grid_based").
        results (Dict[str, Any]): A dictionary mapping image filenames to their computed result.
    """
    method_csv = METHODS_RESULTS[method]
    df = pd.read_csv(method_csv)
    for image_file, result in results.items():
        df.loc[df['Image File'] == image_file, f"{method} Results"] = result
    df.to_csv(method_csv, index=False)
    print(f"Updated results saved to {method_csv}")
    logging.info(f"Updated results saved to {method_csv}")

# ------------------------------------------------------------------------------
# Main Script Execution
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Main execution function.
    This script initializes the common initial predictions and method-specific CSV files.
    Then, it prompts the user (or downstream scripts) to run the respective masking methods,
    which will read from the common initial predictions CSV and update their results accordingly.
    """
    initialize_method_results()
    print("Initialization complete.")
    print("Now run the respective masking scripts ensuring they read the predictions from the common CSV file and update their results accordingly.")
    logging.info("Common initial predictions and method-specific CSVs have been initialized.")

if __name__ == "__main__":
    main()
