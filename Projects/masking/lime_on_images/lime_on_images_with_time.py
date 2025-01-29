import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from lime import lime_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from initialize_masking_pipeline import initialize_method_results, update_method_results, initial_predictions_csv

# Paths
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"
output_csv = "plots/lime_on_image_masking_results.csv"
test_dir = "dataset/town7_dataset/test/"
plot_dir = "plots/lime_on_image_masking"
os.makedirs(plot_dir, exist_ok=True)

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

# Function to compute image quality metrics
def calculate_image_metrics(original, modified):
    """Computes SSIM, MSE, PSNR, UQI, and VIFP between two images."""
    original_np = original.cpu().squeeze().numpy().transpose(1, 2, 0)
    modified_np = modified.cpu().squeeze().numpy().transpose(1, 2, 0)

    # Convert to uint8 format (0-255)
    original_np = (original_np * 255).astype(np.uint8)
    modified_np = (modified_np * 255).astype(np.uint8)

    # Compute metrics
    metrics = {
        "SSIM": round(ssim(original_np, modified_np, channel_axis=-1, data_range=255), 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np, data_range=255), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }
    return metrics

# LIME classifier prediction function
def classifier_prediction(image_tensor):
    """Runs classifier prediction on images for LIME."""
    with torch.no_grad():
        image_tensor = torch.tensor(image_tensor.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
        latent_vector = encoder(image_tensor)[2]
        prediction = classifier(latent_vector)
        probabilities = F.softmax(prediction, dim=1).cpu().detach().numpy()
        return np.nan_to_num(probabilities)  # Replace NaNs with 0

# Function to apply LIME mask
def apply_lime_mask(image, mask, mask_value=0):
    """Applies LIME mask to an image and converts it into tensor format."""
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np[mask > 0] = mask_value
    image_np = np.clip(image_np, 0, 1)
    return transforms.ToTensor()(Image.fromarray((image_np * 255).astype(np.uint8))).unsqueeze(0).to(device)

# Plot and save images
def plot_and_save_images(input_image, reconstructed_image, masked_image, reconstructed_masked_image, filename):
    """Saves image comparison plots for visualization."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    input_height, input_width = input_image.shape[2], input_image.shape[3]
    reconstructed_resized = F.interpolate(reconstructed_image, size=(input_height, input_width), mode='bilinear')
    reconstructed_masked_resized = F.interpolate(reconstructed_masked_image, size=(input_height, input_width), mode='bilinear')

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(input_image.cpu().squeeze().permute(1, 2, 0).numpy())
    axs[0].set_title("Original Image")
    axs[1].imshow(reconstructed_resized.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[1].set_title("Reconstructed Image")
    axs[2].imshow(masked_image.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[2].set_title("Masked Image")
    axs[3].imshow(reconstructed_masked_resized.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[3].set_title("Reconstructed Masked Image")
    plt.savefig(filename)
    plt.close()

# Process dataset
def process_lime_on_image_masking():
    """Runs LIME-based masking and updates the results CSV row-by-row."""
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(output_csv)

    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(test_dir, image_filename)

        # Load image and convert to tensor
        image = Image.open(image_path).convert("RGB")
        input_image = transforms.ToTensor()(image).unsqueeze(0).to(device)

        counterfactual_found = False
        final_prediction = row["Prediction (Before Masking)"]
        confidence_final_str = "N/A"
        metrics = {"SSIM": "", "MSE": "", "PSNR": "", "UQI": "", "VIFP": ""}  # Empty by default

        # LIME explanation generation
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.array(image), classifier_prediction, hide_color=0, num_samples=1000
        )
        temp, mask = explanation.get_image_and_mask(
            label=0 if final_prediction == "STOP" else 1, positive_only=True, num_features=10, hide_rest=False
        )

        # Apply LIME mask
        masked_image = apply_lime_mask(image, mask)

        # Run classifier on masked image
        latent_vector_masked = encoder(masked_image)[2]
        masked_prediction = classifier(latent_vector_masked)
        confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
        confidence_final_str = ", ".join(map(str, confidence_final))
        final_prediction = "STOP" if torch.argmax(masked_prediction, dim=1).item() == 0 else "GO"
        counterfactual_found = final_prediction != row["Prediction (Before Masking)"]

        # Compute metrics **only if CE is found**
        if counterfactual_found:
            metrics = calculate_image_metrics(input_image, masked_image)

        # Update CSV
        df_results.loc[df_results['Image File'] == image_filename, [
            "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
            "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ]] = [
            final_prediction, confidence_final_str, counterfactual_found,
            "Superpixel", "N/A",
            metrics["SSIM"], metrics["MSE"], metrics["PSNR"], metrics["UQI"], metrics["VIFP"],
            round(time.time() - start_time + row["Time Taken (s)"], 5)
        ]

        df_results.to_csv(output_csv, index=False)
        print(f"✅ Processed {image_filename}: CE Found = {counterfactual_found}, Time Taken = {round(time.time() - start_time, 5)}s")

process_lime_on_image_masking()
