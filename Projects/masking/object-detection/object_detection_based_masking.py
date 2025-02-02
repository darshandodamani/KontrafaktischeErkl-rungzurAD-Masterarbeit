import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
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
output_csv = "plots/object_detection_masking_results.csv"
test_dir = "dataset/town7_dataset/test/"
plot_dir = "plots/object_detection_masking_images"
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

# Load YOLO model once before the loop
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, autoshape=True)

# Transformations
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])


def calculate_image_metrics(original_image, modified_image):
    """Calculate image similarity metrics."""
    original_np = original_image.cpu().squeeze().permute(1, 2, 0).numpy()
    modified_np = modified_image.cpu().squeeze().permute(1, 2, 0).detach().numpy()

    metrics = {
        "SSIM": round(ssim(original_np, modified_np, channel_axis=-1, data_range=1.0), 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np, data_range=1.0), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }
    return metrics


def plot_and_save_images(input_image, reconstructed_image, masked_image, reconstructed_masked_image, filename):
    """Save visualization of original, masked, and reconstructed images."""
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

    print(f"Saved plot to: {filename}")


def process_object_detection_masking():
    """Runs object detection-based masking and updates the results CSV row-by-row."""
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(output_csv)

    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(test_dir, image_filename)

        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)

        counterfactual_found = False
        final_prediction = row["Prediction (Before Masking)"]
        grid_size_found = "N/A"
        grid_position_found = "N/A"
        confidence_final_str = "N/A"
        metrics = {"SSIM": "", "MSE": "", "PSNR": "", "UQI": "", "VIFP": ""}

        results = yolo_model(image)
        detections = results.xyxy[0]
        objects_detected = [results.names[int(det[5])] for det in detections] if detections.numel() > 0 else []

        if detections.numel() > 0:
            for det in detections:
                x_min, y_min, x_max, y_max = map(int, det[:4])
                grid_size_found = f"({x_max - x_min}, {y_max - y_min})"
                grid_position_found = f"({x_min}, {y_min})"

                masked_image = input_image.clone()
                masked_image[:, :, y_min:y_max, x_min:x_max] = 0

                latent_vector_masked = encoder(masked_image)[2]
                masked_prediction = classifier(latent_vector_masked)
                confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
                confidence_final_str = ", ".join(map(str, confidence_final))
                predicted_label_after_masking = torch.argmax(masked_prediction, dim=1).item()
                final_prediction = "STOP" if predicted_label_after_masking == 0 else "GO"

                if final_prediction != row["Prediction (Before Masking)"]:
                    counterfactual_found = True
                    metrics = calculate_image_metrics(input_image, masked_image)
                    break

        end_time = time.time()
        total_time_taken = round(end_time - start_time + row["Time Taken (s)"], 5)

        df_results.loc[df_results['Image File'] == image_filename, ["Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found", "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Objects Detected", "Time Taken (s)"]] = [final_prediction, confidence_final_str, counterfactual_found, grid_size_found, grid_position_found, metrics["SSIM"], metrics["MSE"], metrics["PSNR"], metrics["UQI"], metrics["VIFP"], ", ".join(objects_detected), total_time_taken]

    df_results.to_csv(output_csv, index=False)
    print(f"Object detection-based masking results saved to {output_csv}")

process_object_detection_masking()
