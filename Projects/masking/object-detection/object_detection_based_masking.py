# location: Projects/masking/object-detection/object_detection_based_masking.py
import os
import sys
import time
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
from skimage.transform import resize
import csv
import matplotlib.pyplot as plt
# warning: cache/torch/hub/ultralytics_yolov5_master/models/common.py:892: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead with amp.autocast(autocast):
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

# Load YOLOv5 model for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, autoshape=True)

# Transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

# Helper function to calculate image quality metrics
def calculate_metrics(original, reconstructed):
    """
    Calculate metrics like SSIM, MSE, PSNR, VIFP, and UQI between two images.
    Resizes the reconstructed image to match the original image dimensions.
    """
    original_np = original.cpu().squeeze().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.cpu().squeeze().permute(1, 2, 0).detach().numpy()

    # Ensure reconstructed image has the same dimensions as the original
    reconstructed_np_resized = resize(reconstructed_np, original_np.shape, anti_aliasing=True)

    metrics = {
        "SSIM": ssim(original_np, reconstructed_np_resized, channel_axis=-1, data_range=1.0),
        "MSE": mse(original_np, reconstructed_np_resized),
        "PSNR": psnr(original_np, reconstructed_np_resized, data_range=1.0),
        "VIFP": vifp(original_np, reconstructed_np_resized),
        "UQI": uqi(original_np, reconstructed_np_resized),
    }
    return metrics


# Plot and save images for visualization
def plot_and_save_images(input_image, reconstructed_image, masked_image, reconstructed_masked_image, filename):
    """
    Save a plot with four subplots:
    - Original Image
    - Reconstructed Image (resized to match original)
    - Masked Image
    - Reconstructed Masked Image (resized to match original)
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Resize reconstructed images to match original input dimensions
    input_height, input_width = input_image.shape[2], input_image.shape[3]
    reconstructed_image_resized = F.interpolate(reconstructed_image, size=(input_height, input_width), mode='bilinear', align_corners=False)
    reconstructed_masked_image_resized = F.interpolate(reconstructed_masked_image, size=(input_height, input_width), mode='bilinear', align_corners=False)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(input_image.cpu().squeeze().permute(1, 2, 0).numpy())
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(reconstructed_image_resized.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")

    axs[2].imshow(masked_image.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[2].set_title("Masked Image")
    axs[2].axis("off")

    axs[3].imshow(reconstructed_masked_image_resized.cpu().squeeze().permute(1, 2, 0).detach().numpy())
    axs[3].set_title("Reconstructed Masked Image")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()    
    
    print(f"Saved plot to: {filename}")

# Process the dataset and generate results
def process_dataset(dataset_dir, csv_filename, plot_dir):
    """
    Main function to process the dataset:
    - Apply YOLOv5 object detection.
    - Mask detected objects and evaluate counterfactual explanations.
    - Save results in CSV and generate plots.
    """
    total_start_time = time.time()

    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "Image File", "Initial Prediction", "Final Prediction", 
            "Counterfactual Found", "Grid Size", "Grid Position", 
            "Confidence Initial", "Confidence Final", 
            "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Objects Detected", "Processing Time (s)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_filename in os.listdir(dataset_dir):
            if not image_filename.lower().endswith(('.png')):
                continue

            image_path = os.path.join(dataset_dir, image_filename)
            image = Image.open(image_path).convert('RGB')
            input_image = transform(image).unsqueeze(0).to(device)


            start_time = time.time()

            # Step 1: Initial Prediction and Reconstruction
            latent_vector = encoder(input_image)[2]
            reconstructed_image = decoder(latent_vector)
            initial_prediction = classifier(latent_vector)
            initial_class = "STOP" if torch.argmax(initial_prediction, dim=1).item() == 0 else "GO"
            confidence_initial = F.softmax(initial_prediction, dim=1)[0]

            # Step 2: YOLOv5 Object Detection
            results = yolo_model(image)
            detections = results.xyxy[0]
            objects_detected = [results.names[int(det[5])] for det in detections] if detections.numel() > 0 else []

            counterfactual_found = False
            metrics_masked = {"SSIM": None, "MSE": None, "PSNR": None, "UQI": None, "VIFP": None}
            confidence_final = None
            final_class = None

            # Process each detected object
            if detections.numel() > 0:
                for det in detections:
                    x_min, y_min, x_max, y_max = map(int, det[:4])
                    grid_size = f"({x_max - x_min}, {y_max - y_min})"
                    grid_position = f"({x_min}, {y_min})"

                    # Create masked image
                    masked_image = input_image.clone()
                    masked_image[:, :, y_min:y_max, x_min:x_max] = 0

                    # Reconstruct and classify masked image
                    masked_latent_vector = encoder(masked_image)[2]
                    reconstructed_masked_image = decoder(masked_latent_vector)
                    final_prediction = classifier(masked_latent_vector)
                    confidence_final = F.softmax(final_prediction, dim=1)[0]
                    final_class = "STOP" if torch.argmax(final_prediction, dim=1).item() == 0 else "GO"

                    if final_class != initial_class:
                        counterfactual_found = True
                        metrics_masked = calculate_metrics(input_image, reconstructed_masked_image)
                        plot_and_save_images(
                            input_image, reconstructed_image, masked_image, reconstructed_masked_image,
                            os.path.join(plot_dir, f"{image_filename}_mask_{x_min}_{y_min}.png")
                        )
                        break

            processing_time = time.time() - start_time

            # Save results to CSV
            writer.writerow({
                "Image File": image_filename,
                "Initial Prediction": initial_class,
                "Final Prediction": final_class,
                "Counterfactual Found": counterfactual_found,
                "Grid Size": grid_size if detections.numel() > 0 else None,
                "Grid Position": grid_position if detections.numel() > 0 else None,
                "Confidence Initial": [round(x, 5) for x in confidence_initial.tolist()],
                "Confidence Final": [round(x, 5) for x in confidence_final.tolist()] if confidence_final is not None else None,
                "SSIM": round(metrics_masked["SSIM"], 5) if metrics_masked["SSIM"] else None,
                "MSE": round(metrics_masked["MSE"], 5) if metrics_masked["MSE"] else None,
                "PSNR": round(metrics_masked["PSNR"], 5) if metrics_masked["PSNR"] else None,
                "UQI": round(metrics_masked["UQI"], 5) if metrics_masked["UQI"] else None,
                "VIFP": round(metrics_masked["VIFP"], 5) if metrics_masked["VIFP"] else None,
                "Objects Detected": ", ".join(objects_detected),
                "Processing Time (s)": round(processing_time, 5)
            })


    print(f"Total processing time: {round(time.time() - total_start_time, 2)} seconds")



# Process train and test datasets
process_dataset('dataset/town7_dataset/train/', 
                'plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv',
                'plots/object_detection_using_yolov5/train_plots/')

process_dataset('dataset/town7_dataset/test/', 
                'plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv',
                'plots/object_detection_using_yolov5/test_plots/')
