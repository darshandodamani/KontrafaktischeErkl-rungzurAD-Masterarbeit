import os
import sys
import csv
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# Paths to models
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"

test_dir = "dataset/town7_dataset/test/"
initial_predictions_csv = "plots/common_initial_predictions.csv"
methods_results = {
    "grid_based": "plots/grid_based_masking_results.csv",
    "lime_on_image": "plots/lime_on_image_masking_results.csv",
    "object_detection": "plots/object_detection_masking_results.csv",
    "lime_on_latent": "plots/lime_on_latent_masking_results.csv"
}

# Column headers for each method
methods_headers = {
    "grid_based": ["Image File", "Prediction (Before Masking)", "Confidence (Before Masking)", "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found", "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"],
    "lime_on_image": ["Image File", "Prediction (Before Masking)", "Confidence (Before Masking)", "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found", "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"],
    "object_detection": ["Image File", "Prediction (Before Masking)", "Confidence (Before Masking)", "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found", "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Objects Detected", "Time Taken (s)"],
    "lime_on_latent": ["Image File", "Prediction (Before Masking)", "Confidence (Before Masking)", "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found", "Feature Selection (%)", "Selected Features", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"]
}

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

# Transformations
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

def compute_initial_predictions(dataset_dir, output_csv):
    """
    Runs initial classification once for each image in the dataset and saves results to a CSV file.
    """
    if os.path.exists(output_csv):
        print("Initial predictions file already exists. Skipping re-computation.")
        return pd.read_csv(output_csv)
    
    print("Computing initial predictions for test dataset...")
    start_time = time.time()
    records = []
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image File", "Prediction (Before Masking)", "Confidence (Before Masking)", "Time Taken (s)"])
        
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
                records.append([image_file, predicted_class, list(map(lambda x: round(float(x), 5), confidence)), image_time_taken])
    
    df = pd.DataFrame(records, columns=["Image File", "Prediction (Before Masking)", "Confidence (Before Masking)", "Time Taken (s)"])
    df.to_csv(output_csv, index=False)
    print(f"Initial predictions saved to {output_csv}")
    return df

def initialize_method_results():
    """Ensures that each masking method has a results CSV with initial prediction data populated."""
    df_initial = compute_initial_predictions(test_dir, initial_predictions_csv)
    
    for method, method_csv in methods_results.items():
        if os.path.exists(method_csv):
            continue  # Skip if file already exists
        
        df_method = df_initial.copy()
        for col in methods_headers[method]:
            if col not in df_method.columns:
                df_method[col] = ""
        
        df_method = df_method[methods_headers[method]]  # Ensure correct order of columns
        df_method.to_csv(method_csv, index=False)
        print(f"Initialized {method_csv} with headers and initial prediction data.")

def update_method_results(method, results):
    """Updates the CSV file for a specific masking method with computed results."""
    method_csv = methods_results[method]
    
    df = pd.read_csv(method_csv)
    for image_file, result in results.items():
        df.loc[df['Image File'] == image_file, f"{method} Results"] = result
    
    df.to_csv(method_csv, index=False)
    print(f"Updated results saved to {method_csv}")

# Step 1: Initialize common initial predictions and create method-specific CSVs
initialize_method_results()

# Step 2: Now, run individual masking methods, ensuring they read the predictions from the common CSV
print("Now run the respective masking scripts ensuring they read the predictions from the common CSV file and update their results accordingly.")
