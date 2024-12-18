import os
import sys
import csv
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
from torchvision import transforms

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# Paths to models and datasets
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"
median_values_csv = "latent_vectors/combined_median_values.csv"

# CSV output files
output_csv_train = "plots/lime_plots/lime_based_counterfactual_results_train.csv"
output_csv_test = "plots/lime_plots/lime_based_counterfactual_results_test.csv"

# Dataset directories
train_dir = "dataset/town7_dataset/train/"
test_dir = "dataset/town7_dataset/test/"

# Define headers for the CSV
csv_headers = [
    "Image File", "Prediction", "Feature Selection (%)", "Selected Features",
    "Counterfactual Found", "Confidence", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
classifier = Classifier().to(device)

encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

encoder.eval()
decoder.eval()
classifier.eval()

# Load median values
median_values = pd.read_csv(median_values_csv).values.flatten()

# Transformations
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor()
])

# Logging function
def log_to_csv(file_path, headers, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)  # Write headers only if file is new
        writer.writerow(data)

# Process dataset
def process_dataset(image_dir, output_csv):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)

        # Start timing
        start_time = time.time()

        # Encode and classify the image
        latent_vector = encoder(input_image)[2]
        predicted_label = torch.argmax(classifier(latent_vector), dim=1).item()
        predicted_class = "STOP" if predicted_label == 0 else "GO"

        # LIME explanation
        latent_vector_np = latent_vector.cpu().detach().numpy().reshape(1, -1)

        def predict_with_latent(latent):
            latent_tensor = torch.tensor(latent, dtype=torch.float32).to(device)
            output = classifier(latent_tensor)
            return F.softmax(output, dim=1).cpu().detach().numpy()

        explainer = LimeTabularExplainer(
            latent_vector_np,
            mode='classification',
            feature_names=[f'latent_{i}' for i in range(latent_vector_np.shape[1])],
            discretize_continuous=False
        )

        explanation = explainer.explain_instance(
            latent_vector_np.flatten(),
            predict_with_latent,
            num_features=len(latent_vector_np.flatten())
        )

        # Masking loop
        percentage_value = 0
        step_size = 0.01
        counterfactual_found = False
        confidence = None
        metrics = {}
        selected_features = []

        positive_importance_list = sorted(
            [(feature, weight) for feature, weight in explanation.as_list() if weight > 0],
            key=lambda x: abs(x[1]), reverse=True
        )

        while percentage_value < 1:
            percentage_value += step_size
            num_features_to_select = int(len(positive_importance_list) * percentage_value)
            selected_features = [int(feature.split("_")[-1]) for feature, _ in positive_importance_list[:num_features_to_select]]
            
            masked_latent_vector = latent_vector_np.flatten()
            for feature_index in selected_features:
                masked_latent_vector[feature_index] = median_values[feature_index]
            
            masked_latent_tensor = torch.tensor(masked_latent_vector, dtype=torch.float32).to(device).reshape(1, -1)
            reconstructed_image_after_masking = decoder(masked_latent_tensor).squeeze(0)
            
            masked_image_predicted_label = classifier(masked_latent_tensor)
            predicted_label_after_masking = torch.argmax(masked_image_predicted_label, dim=1).item()
            predicted_class_after_masking = "STOP" if predicted_label_after_masking == 0 else "GO"

            if predicted_class_after_masking != predicted_class:
                counterfactual_found = True
                confidence = F.softmax(masked_image_predicted_label, dim=1).cpu().detach().numpy().tolist()[0]
                
                # Resize the reconstructed image to match the input image dimensions
                reconstructed_image_resized = Image.fromarray(
                    (reconstructed_image_after_masking.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                ).resize((input_image.shape[3], input_image.shape[2]))  # Match input image dimensions (width, height)
                
                # Convert resized images to numpy arrays for comparison
                input_image_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert input tensor to numpy
                reconstructed_image_np = np.array(reconstructed_image_resized) / 255.0  # Normalize to [0, 1]
                
                # Calculate metrics
                metrics = {
                    "SSIM": ssim(input_image_np, reconstructed_image_np, data_range=1.0, channel_axis=-1),
                    "MSE": mse(input_image_np, reconstructed_image_np),
                    "PSNR": psnr(input_image_np, reconstructed_image_np, data_range=1.0),
                    "UQI": uqi(input_image_np, reconstructed_image_np),
                    "VIFP": vifp(input_image_np, reconstructed_image_np),
                }
                break


        # Ensure metrics have default values if no counterfactual explanation is found
        if not counterfactual_found:
            metrics = {"SSIM": "", "MSE": "", "PSNR": "", "UQI": "", "VIFP": ""}

        # Log results
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)

        log_to_csv(output_csv, csv_headers, [
            image_file, predicted_class, f"{percentage_value * 100:.2f}%", selected_features,
            counterfactual_found, confidence, 
            metrics.get("SSIM", ""), metrics.get("MSE", ""), metrics.get("PSNR", ""),
            metrics.get("UQI", ""), metrics.get("VIFP", ""), time_taken
        ])

    print(f"Processing completed for dataset: {image_dir}. Results saved to {output_csv}")

# Run for train and test datasets
process_dataset(train_dir, output_csv_train)
process_dataset(test_dir, output_csv_test)


