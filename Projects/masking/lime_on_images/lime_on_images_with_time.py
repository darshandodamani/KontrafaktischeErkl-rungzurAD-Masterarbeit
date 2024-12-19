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
    image_np[mask > 0] = mask_value  # Mask positive contributions

    # Ensure the image is resized to the expected input size
    resized_image = cv2.resize(image_np, (80, 160))  # Replace with model-specific dimensions
    return transforms.ToTensor()(resized_image).unsqueeze(0).to(device)

# Define function to calculate image metrics
def calculate_image_metrics(original, modified):
    original_np = np.array(original, dtype=np.float32) / 255.0
    modified_np = np.array(modified, dtype=np.float32) / 255.0

    if original_np.shape != modified_np.shape:
        from skimage.transform import resize
        modified_np = resize(modified_np, original_np.shape, anti_aliasing=True)

    metrics = {
        "SSIM": ssim(original_np, modified_np, channel_axis=-1, data_range=1.0),
        "MSE": mse(original_np, modified_np),
        "PSNR": psnr(original_np, modified_np, data_range=1.0),
        "VIF": vifp(original_np, modified_np),
        "UQI": uqi(original_np, modified_np),
    }
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
        return np.zeros((image_tensor.shape[0], 2))  # Return dummy output on error

# Process dataset with time tracking
def process_dataset(dataset_dir, output_csv):
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    total_time = 0  # Track total processing time

    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "Image File", "Prediction", "Confidence (Before Masking)", "Prediction (After Masking)",
            "Confidence (After Masking)", "Counterfactual Found", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_filename in os.listdir(dataset_dir):
            if not image_filename.lower().endswith(('.png')):
                continue

            image_path = os.path.join(dataset_dir, image_filename)
            start_time = time.time()  # Start timing for this image

            try:
                image = Image.open(image_path).convert('RGB')
                input_image = transform(image).unsqueeze(0).to(device)

                # Initial prediction
                latent_vector = encoder(input_image)[2]
                latent_vector = latent_vector.view(latent_vector.size(0), -1)

                reconstructed_image = decoder(latent_vector)
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

                # Apply mask and get prediction reconstructed masked image
                masked_image = apply_lime_mask(image, mask)

                # Encode masked image
                latent_vector_masked = encoder(masked_image)[2]
                latent_vector_masked = latent_vector_masked.view(latent_vector_masked.size(0), -1)

                # Decode latent vector to get reconstructed masked image
                reconstructed_image_masked = decoder(latent_vector_masked)

                # Classify the re-encoded masked image
                masked_prediction = classifier(latent_vector_masked)
                masked_label = torch.argmax(masked_prediction, dim=1).item()
                masked_class = "STOP" if masked_label == 0 else "GO"
                confidence_after = F.softmax(masked_prediction, dim=1).cpu().detach().numpy()
                counterfactual_found = masked_class != predicted_class

                # Metrics
                metrics = calculate_image_metrics(
                    np.array(image),
                    masked_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
                )

                end_time = time.time()  # End timing for this image
                time_taken = round(end_time - start_time, 2)
                total_time += time_taken

                # Write results to CSV
                writer.writerow({
                    "Image File": image_filename,
                    "Prediction": predicted_class,
                    "Confidence (Before Masking)": confidence_before.tolist(),
                    "Prediction (After Masking)": masked_class,
                    "Confidence (After Masking)": confidence_after.tolist(),
                    "Counterfactual Found": counterfactual_found,
                    "SSIM": metrics["SSIM"],
                    "MSE": metrics["MSE"],
                    "PSNR": metrics["PSNR"],
                    "UQI": metrics["UQI"],
                    "VIFP": metrics["VIF"],
                    "Time Taken (s)": time_taken
                })

                print(f"Processed {image_filename} - Time Taken: {time_taken}s - Counterfactual Found: {counterfactual_found}")
            except Exception as e:
                print(f"Error processing {image_filename}: {e}")
                
                # Write placeholder row for failed image
                writer.writerow({
                    "Image File": image_filename,
                    "Prediction": "Error",
                    "Confidence (Before Masking)": "N/A",
                    "Prediction (After Masking)": "N/A",
                    "Confidence (After Masking)": "N/A",
                    "Counterfactual Found": "N/A",
                    "SSIM": "N/A",
                    "MSE": "N/A",
                    "PSNR": "N/A",
                    "UQI": "N/A",
                    "VIFP": "N/A",
                    "Time Taken (s)": "N/A"
                })

        print(f"Results saved to {output_csv}. Total Time Taken: {total_time:.2f} seconds.")

# Process train and test datasets
process_dataset("dataset/town7_dataset/train/", "plots/lime_on_images/lime_on_image_masking_train_results.csv")
process_dataset("dataset/town7_dataset/test/", "plots/lime_on_images/lime_on_image_masking_test_results.csv")
