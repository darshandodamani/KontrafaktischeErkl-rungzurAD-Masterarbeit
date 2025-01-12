import os
import sys
import time
import torch
import numpy as np
import csv
import torchvision.transforms as transforms
from PIL import Image
from lime import lime_image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# Paths to models
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"

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

# Define function for LIME-based masking
def apply_lime_mask(image, mask, mask_value=0):
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_np[mask > 0] = mask_value  # Apply mask
    resized_image = cv2.resize(image_np, (160, 80))  # Resize to match model input
    return transforms.ToTensor()(resized_image).unsqueeze(0).to(device)

# Define function to calculate image metrics
def calculate_image_metrics(original, modified):
    original_np = np.array(original, dtype=np.float32) / 255.0
    modified_np = np.array(modified, dtype=np.float32) / 255.0

    if original_np.shape != modified_np.shape:
        from skimage.transform import resize
        modified_np = resize(modified_np, original_np.shape, anti_aliasing=True)

    metrics = {}
    try:
        metrics["SSIM"] = round(ssim(original_np, modified_np, channel_axis=-1, data_range=1.0), 5)
    except ValueError:
        metrics["SSIM"] = 0.0  # Default value for invalid calculations

    try:
        metrics["MSE"] = round(mse(original_np, modified_np), 5)
    except ValueError:
        metrics["MSE"] = float("inf")  # Default for invalid calculations

    try:
        metrics["PSNR"] = round(psnr(original_np, modified_np, data_range=1.0), 5)
    except ValueError:
        metrics["PSNR"] = 0.0

    try:
        metrics["VIFP"] = round(vifp(original_np, modified_np), 5)
    except ValueError:
        metrics["VIFP"] = 0.0

    try:
        metrics["UQI"] = round(uqi(original_np, modified_np), 5)
    except ValueError:
        metrics["UQI"] = 0.0

    # Handle NaN values (replace with 0)
    for key, value in metrics.items():
        if np.isnan(value):
            metrics[key] = 0.0

    return metrics

# Define classifier prediction function for LIME
def classifier_prediction(image_tensor):
    try:
        with torch.no_grad():
            image_tensor = torch.tensor(image_tensor.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)
            latent_vector = encoder(image_tensor)[2]
            prediction = classifier(latent_vector)
            probabilities = F.softmax(prediction, dim=1).cpu().detach().numpy()
            return probabilities
    except Exception as e:
        print(f"Error in classifier_prediction: {e}")
        return np.zeros((image_tensor.shape[0], 2))  # Dummy output

# Plot and save images
def plot_and_save_images(input_image, reconstructed_image, masked_image, reconstructed_masked_image, filename):
    """
    Save a plot with four subplots:
    - Original Image
    - Reconstructed Image (resized to match original)
    - Masked Image
    - Reconstructed Masked Image (resized to match original)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    input_height, input_width = input_image.shape[2], input_image.shape[3]
    reconstructed_resized = F.interpolate(reconstructed_image, size=(input_height, input_width), mode='bilinear')
    reconstructed_masked_resized = F.interpolate(reconstructed_masked_image, size=(input_height, input_width), mode='bilinear')

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

# Process dataset
def process_dataset(dataset_dir, output_csv):
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    total_time = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "Image File", "Prediction (Before Masking)", "Confidence (Before Masking)",
            "Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found",
            "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Grid Size", "Grid Position", "Time Taken (s)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_filename in os.listdir(dataset_dir):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(dataset_dir, image_filename)
            start_time = time.time()

            try:
                image = Image.open(image_path).convert('RGB')
                input_image = transform(image).unsqueeze(0).to(device)

                # Initial prediction
                latent_vector = encoder(input_image)[2]
                prediction = classifier(latent_vector)
                predicted_label = torch.argmax(prediction, dim=1).item()
                predicted_class = "STOP" if predicted_label == 0 else "GO"
                confidence_before = F.softmax(prediction, dim=1).cpu().detach().numpy()

                # LIME explanation
                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(
                    np.array(image),
                    classifier_prediction,
                    hide_color=0,
                    num_samples=1000
                )
                
                temp, mask = explanation.get_image_and_mask(
                    label=predicted_label,
                    positive_only=True,
                    num_features=10,
                    hide_rest=False
                )

                # Apply mask and get prediction for reconstructed masked image
                masked_image = apply_lime_mask(image, mask)

                # Encode masked image
                latent_vector_masked = encoder(masked_image)[2]
                reconstructed_image = decoder(latent_vector)
                reconstructed_masked_image = decoder(latent_vector_masked)

                # Classify the masked image
                masked_prediction = classifier(latent_vector_masked)
                masked_label = torch.argmax(masked_prediction, dim=1).item()
                masked_class = "STOP" if masked_label == 0 else "GO"
                confidence_after = F.softmax(masked_prediction, dim=1).cpu().detach().numpy()
                counterfactual_found = masked_class != predicted_class

                # Calculate metrics
                metrics = calculate_image_metrics(
                    np.array(image),
                    masked_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                )

                end_time = time.time()
                time_taken = round(end_time - start_time, 5)
                total_time += time_taken

                # Write results
                writer.writerow({
                    "Image File": image_filename,
                    "Prediction (Before Masking)": predicted_class,
                    "Confidence (Before Masking)": [round(float(x), 5) for x in confidence_before[0]],
                    "Prediction (After Masking)": masked_class,
                    "Confidence (After Masking)": [round(float(x), 5) for x in confidence_after[0]],
                    "Counterfactual Found": counterfactual_found,
                    "SSIM": round(metrics["SSIM"], 5),
                    "MSE": round(metrics["MSE"], 5),
                    "PSNR": round(metrics["PSNR"], 5),
                    "UQI": round(metrics["UQI"], 5),
                    "VIFP": round(metrics["VIFP"], 5),
                    "Grid Size": "Superpixel",  # LIME does not use a grid
                    "Grid Position": "N/A",  # Placeholder
                    "Time Taken (s)": round(time_taken, 5)
                })

                # Save plot
                plot_and_save_images(
                    input_image, reconstructed_image, masked_image, reconstructed_masked_image,
                    f"plots/lime_on_images/{image_filename.split('.')[0]}_plot.png"
                )

                print(f"Processed {image_filename}: Time {time_taken}s, CE Found: {counterfactual_found}")
            except Exception as e:
                print(f"Error processing {image_filename}: {e}")

        print(f"Results saved to {output_csv}. Total Time Taken: {total_time:.2f} seconds.")

# Process train and test datasets
process_dataset("dataset/town7_dataset/train/", "plots/lime_on_images/lime_on_image_masking_train_results.csv")
process_dataset("dataset/town7_dataset/test/", "plots/lime_on_images/lime_on_image_masking_test_results.csv")
