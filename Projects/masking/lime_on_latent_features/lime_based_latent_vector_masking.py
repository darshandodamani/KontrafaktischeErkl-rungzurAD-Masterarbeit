import os
import sys
import csv
import time
import torch
import numpy as np
import pandas as pd
import cv2
from skimage.transform import resize
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
from torchvision import transforms
from PIL import Image

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from initialize_masking_pipeline import initialize_method_results, update_method_results, initial_predictions_csv

# Paths to models and datasets
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"
median_values_csv = "latent_vectors/combined_median_values.csv"

# Output CSV files
output_csv = "plots/lime_on_latent_masking_results.csv"
test_dir = "dataset/town7_dataset/test/"
plot_dir = "plots/lime_on_latent"
os.makedirs(plot_dir, exist_ok=True)

# Define headers for CSV
csv_headers = [
    "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
    "Prediction (After Masking)", "Confidence (After Masking)",
    "Counterfactual Found", "Feature Selection (%)", "Selected Features",
    "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
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

# Load initial predictions
df_initial = pd.read_csv(initial_predictions_csv)
df_results = pd.read_csv(output_csv)

# Transformations
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor()
])

def calculate_image_metrics(original, modified):
    """Calculates image similarity metrics after resizing modified to match original."""
    if original.shape != modified.shape:
        modified = cv2.resize(modified, (original.shape[1], original.shape[0]))  # Resize to match original

    metrics = {
        "SSIM": round(ssim(original, modified, data_range=1.0, channel_axis=-1), 5),
        "MSE": round(mse(original, modified), 5),
        "PSNR": round(psnr(original, modified, data_range=1.0), 5),
        "UQI": round(uqi(original, modified), 5),
        "VIFP": round(vifp(original, modified), 5),
    }
    return metrics

def plot_and_save_images(original_image, reconstructed_image, masked_reconstructed_image, filename):
    """
    Saves a comparison plot of original, reconstructed, and masked reconstructed images.
    """
    # Ensure all images have the same size
    reconstructed_resized = cv2.resize(reconstructed_image, (original_image.shape[1], original_image.shape[0]))
    masked_reconstructed_resized = cv2.resize(masked_reconstructed_image, (original_image.shape[1], original_image.shape[0]))

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Set up the plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(reconstructed_resized)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis('off')

    axs[2].imshow(masked_reconstructed_resized)
    axs[2].set_title("Masked Reconstructed Image")
    axs[2].axis('off')

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Plot saved to {filename}")


def predict_with_latent(latent):
    """Helper function for LIME to classify latent features."""
    latent_tensor = torch.tensor(latent, dtype=torch.float32).to(device)
    output = classifier(latent_tensor)
    return F.softmax(output, dim=1).cpu().detach().numpy()

def process_lime_on_latent_masking():
    """Runs LIME-based masking and updates the results CSV row-by-row."""
    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(test_dir, image_filename)

        # Load image
        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)

        # Defaults
        counterfactual_found = False
        final_prediction = row["Prediction (Before Masking)"]
        predicted_class = 0 if final_prediction == "STOP" else 1
        confidence_final_str = "N/A"
        metrics = {}
        selected_features = []

        # Encode image
        latent_vector = encoder(input_image)[2]
        latent_vector_np = latent_vector.cpu().detach().numpy().reshape(1, -1)

        # LIME Explanation
        explainer = LimeTabularExplainer(
            latent_vector_np,
            mode="classification",
            feature_names=[f"latent_{i}" for i in range(latent_vector_np.shape[1])],
            discretize_continuous=False
        )

        explanation = explainer.explain_instance(
            latent_vector_np.flatten(), predict_with_latent, num_features=latent_vector_np.shape[1]
        )

        # Masking loop
        percentage_value = 0.0
        step_size = 0.01
        positive_importance_list = sorted(
            [(feature, weight) for feature, weight in explanation.as_list() if weight > 0],
            key=lambda x: abs(x[1]), reverse=True
        )

        while percentage_value < 1.0:
            percentage_value += step_size
            num_features_to_select = int(len(positive_importance_list) * percentage_value)
            selected_features = [int(feature.split("_")[-1]) for feature, _ in positive_importance_list[:num_features_to_select]]

            masked_latent_vector = latent_vector_np.flatten()
            for feature_index in selected_features:
                masked_latent_vector[feature_index] = median_values[feature_index]

            masked_latent_tensor = torch.tensor(masked_latent_vector, dtype=torch.float32).to(device).reshape(1, -1)
            masked_prediction = classifier(masked_latent_tensor)
            confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
            confidence_final_str = ", ".join(map(str, confidence_final))
            predicted_label_after_masking = torch.argmax(masked_prediction, dim=1).item()
            final_prediction = "STOP" if predicted_label_after_masking == 0 else "GO"
            predicted_class_after_masking = predicted_label_after_masking

            # Counterfactual found, calculate confidence and metrics
            if predicted_class_after_masking != predicted_class:
                counterfactual_found = final_prediction != row["Prediction (Before Masking)"]
                confidence_after = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()

                # Convert list values into strings
                confidence_final_str = ", ".join(map(str, confidence_after))
                selected_features_str = ", ".join(map(str, selected_features))

                # Calculate metrics only if counterfactual found
                input_image_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                reconstructed_image_after_masking = decoder(masked_latent_tensor).squeeze(0)
                reconstructed_image_np = reconstructed_image_after_masking.permute(1, 2, 0).cpu().detach().numpy()
                reconstructed_image_np_resized = cv2.resize(
                    reconstructed_image_np, 
                    (input_image_np.shape[1], input_image_np.shape[0])  # Match original dimensions
                )

                # Calculate metrics only if counterfactual was found
                metrics = {
                    "SSIM": round(ssim(input_image_np, reconstructed_image_np_resized, data_range=1.0, channel_axis=-1), 5) if counterfactual_found else np.nan,
                    "MSE": round(mse(input_image_np, reconstructed_image_np_resized), 5) if counterfactual_found else np.nan,
                    "PSNR": round(psnr(input_image_np, reconstructed_image_np_resized, data_range=1.0), 5) if counterfactual_found else np.nan,
                    "UQI": round(uqi(input_image_np, reconstructed_image_np_resized), 5) if counterfactual_found else np.nan,
                    "VIFP": round(vifp(input_image_np, reconstructed_image_np_resized), 5) if counterfactual_found else np.nan,
                }

                # Save visualization
                plot_filename = f"plots/lime_on_latent/{os.path.splitext(image_filename)[0]}_comparison.png"
                plot_and_save_images(input_image_np, reconstructed_image_np, reconstructed_image_np, plot_filename)

            else:
                # If no counterfactual found, keep fields empty
                confidence_final_str = "N/A"
                selected_features_str = "N/A"
                metrics = {"SSIM": np.nan, "MSE": np.nan, "PSNR": np.nan, "UQI": np.nan, "VIFP": np.nan}

            # Total processing time
            end_time = time.time()
            total_time_taken = round(end_time - start_time, 5)

            # Update CSV
            df_results.loc[df_results['Image File'] == image_filename, [
                "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
                "Feature Selection (%)", "Selected Features",
                "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
            ]] = [
                str(final_prediction),
                confidence_final_str,
                bool(counterfactual_found),
                str(percentage_value * 100) + "%" if counterfactual_found else np.nan,
                selected_features_str if counterfactual_found else np.nan,
                metrics["SSIM"], metrics["MSE"], metrics["PSNR"], metrics["UQI"], metrics["VIFP"],
                float(total_time_taken)
            ]

        df_results.to_csv(output_csv, index=False)

    print(f"LIME on Latent Masking results saved to {output_csv}")

process_lime_on_latent_masking()
