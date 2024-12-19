# location: Projects/masking/object-detection/object_detection_based_masking.py
import os
import sys
from tracemalloc import start
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import seaborn as sns
import csv
import time
from PIL import Image, ImageDraw
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import torchvision.transforms.functional as F

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

# Load the models
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

# Manually select an image instead of choosing randomly
# test_dir = 'dataset/town7_dataset/train/'
# image_filename = 'town7_000967.png'
# image_path = os.path.join(test_dir, image_filename)

# # Select and preprocess the image
# test_dir = 'dataset/town7_dataset/train/'
# image_filename = random.choice(os.listdir(test_dir))
# print(f"Selected Image: {image_filename}")
# image_path = os.path.join(test_dir, image_filename)

# YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, autoshape=True)

# Helper function to calculate metrics
def calculate_metrics(original, reconstructed):
    original_np = original.cpu().squeeze().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.cpu().squeeze().permute(1, 2, 0).detach().numpy()
    
    reconstructed_np_resized = resize(reconstructed_np, original_np.shape, anti_aliasing=True)

    metrics = {
        "SSIM": ssim(original_np, reconstructed_np_resized, channel_axis=-1, data_range=1.0),
        "MSE": mse(original_np, reconstructed_np_resized),
        "PSNR": psnr(original_np, reconstructed_np_resized, data_range=1.0),
        "VIFP": vifp(original_np, reconstructed_np_resized),
        "UQI": uqi(original_np, reconstructed_np_resized),
    }
    return metrics

# Helper function to plot and save images
def plot_and_save_images(input_image, reconstructed_image, masked_image_tensor, reconstructed_masked_image_resized, filename):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Resize reconstructed image to match the original for consistent plotting
    reconstructed_np = reconstructed_image.cpu().squeeze().permute(1, 2, 0).detach().numpy()
    original_np_shape = input_image.cpu().squeeze().permute(1, 2, 0).numpy().shape
    reconstructed_np_resized = resize(reconstructed_np, original_np_shape, anti_aliasing=True)

    # Convert tensors to numpy for plotting
    original_np = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    masked_np = masked_image_tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    reconstructed_masked_np = reconstructed_masked_image_resized.cpu().squeeze().permute(1, 2, 0).detach().numpy()

    axs[0].imshow(original_np)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(reconstructed_np_resized)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis('off')

    axs[2].imshow(masked_np)
    axs[2].set_title("Masked Image")
    axs[2].axis('off')

    axs[3].imshow(reconstructed_masked_np)
    axs[3].set_title("Reconstructed Masked Image")
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Helper function to plot and save metrics comparison
def plot_and_save_metrics(metrics_original, metrics_masked, filename):
    metrics_names = list(metrics_original.keys())
    original_values = list(metrics_original.values())
    masked_values = list(metrics_masked.values())

    x = np.arange(len(metrics_names))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, original_values, width, label='Original vs Reconstructed')
    ax.bar(x + width/2, masked_values, width, label='Masked Reconstructed vs Original')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Image Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Load and preprocess the image
def process_dataset(dataset_dir, csv_filename):
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    total_start_time = time.time()  # Start time for the entire dataset processing
    # Prepare CSV file to save the summary
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "Image File", "Prediction", "Grid Size & Position", "Grid Position", "Counterfactual Found", 
            "Confidence", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Objects Detected", "Processing Time (s)"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through all images in the dataset directory
        for image_filename in os.listdir(dataset_dir):
            if not image_filename.lower().endswith(('.png')):
                continue

            image_path = os.path.join(dataset_dir, image_filename)
            image = Image.open(image_path).convert('RGB')
            input_image = transform(image).unsqueeze(0).to(device)

            image_start_time = time.time()  # Start time for this image

            # Step 1: Send the image to encoder, decoder, and classifier
            latent_vector = encoder(input_image)[2]
            reconstructed_image = decoder(latent_vector)
            input_predicted_label = classifier(latent_vector)
            predicted_class = "STOP" if torch.argmax(input_predicted_label, dim=1).item() == 0 else "GO"
            
            # Step 2: YOLOv5 Detection
            results = model(image)
            detections = results.xyxy[0]
            classes_detected = [results.names[int(det[5])] for det in detections] if len(detections) > 0 else []
            counterfactual_found = False

            # Default metrics values
            metrics_masked = {"SSIM": None, "MSE": None, "PSNR": None, "UQI": None, "VIFP": None}
            confidence = None
            grid_size_position = None
            grid_position = None

            if len(detections) > 0:
                for obj_index, detection in enumerate(detections):
                    x_min, y_min, x_max, y_max = map(int, detection[:4])
                    confidence = detection[4].item()
                    grid_size_position = f"({x_min}, {y_min}), ({x_max}, {y_max})"
                    grid_position = grid_size_position

                    # Mask object
                    masked_image = input_image.clone().squeeze().permute(1, 2, 0).cpu().numpy()
                    masked_image[y_min:y_max, x_min:x_max] = 0
                    masked_image_tensor = transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

                    # Step 4: Reconstruction and metrics
                    masked_latent_vector = encoder(masked_image_tensor)[2]
                    reconstructed_masked_image = decoder(masked_latent_vector)
                    reconstructed_masked_image_resized = F.resize(reconstructed_masked_image, [80, 160])

                    # Re-encode and classify the reconstructed masked image
                    re_encoded_latent_vector = encoder(reconstructed_masked_image_resized)[2]
                    re_encoded_predicted_label = torch.argmax(classifier(re_encoded_latent_vector), dim=1).item()
                    re_encoded_predicted_class = "STOP" if re_encoded_predicted_label == 0 else "GO"

                    if re_encoded_predicted_class != predicted_class:
                        counterfactual_found = True
                        metrics_masked = calculate_metrics(input_image, reconstructed_masked_image_resized)

                        # Save images and metrics plot only for counterfactual explanations
                        plot_and_save_images(
                            input_image, reconstructed_image, masked_image_tensor, reconstructed_masked_image_resized,
                            f"plots/object_detection_using_yolov5/ce_images_{image_filename.split('.')[0]}_{obj_index + 1}.png"
                        )
                        plot_and_save_metrics(
                            calculate_metrics(input_image, reconstructed_image), metrics_masked,
                            f"plots/object_detection_using_yolov5/ce_metrics_{image_filename.split('.')[0]}_{obj_index + 1}.png"
                        )

            # Calculate processing time for the image
            processing_time = time.time() - image_start_time

            # Write the summary to the CSV file
            writer.writerow({
                "Image File": image_filename,
                "Prediction": predicted_class,
                "Grid Size & Position": grid_size_position,
                "Grid Position": grid_position,
                "Counterfactual Found": counterfactual_found,
                "Confidence": confidence,
                "SSIM": metrics_masked["SSIM"],
                "MSE": metrics_masked["MSE"],
                "PSNR": metrics_masked["PSNR"],
                "UQI": metrics_masked["UQI"],
                "VIFP": metrics_masked["VIFP"],
                "Objects Detected": ', '.join(classes_detected),
                "Processing Time (s)": f"{processing_time:.2f}"
            })

    total_end_time = time.time()  # End time for the entire dataset processing
    print(f"Total time taken to process {dataset_dir}: {total_end_time - total_start_time:.2f} seconds")

    

# Process train and test datasets
process_dataset('dataset/town7_dataset/train/', 'plots/object_detection_using_yolov5/object_detection_counterfactual_summary_train.csv')
process_dataset('dataset/town7_dataset/test/', 'plots/object_detection_using_yolov5/object_detection_counterfactual_summary_test.csv')
