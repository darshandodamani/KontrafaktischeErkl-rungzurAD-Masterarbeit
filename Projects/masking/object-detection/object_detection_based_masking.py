import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import seaborn as sns
import csv
from PIL import Image, ImageDraw
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi

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
test_dir = 'dataset/town7_dataset/train/'
image_filename = random.choice(os.listdir(test_dir))
print(f"Selected Image: {image_filename}")
image_path = os.path.join(test_dir, image_filename)

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

# Load and preprocess the image
def process_dataset(dataset_dir, csv_filename):
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    # Prepare CSV file to save the summary
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "Image File", "Prediction", "Grid Size & Position", "Grid Position", "Counterfactual Found", 
            "Confidence", "SSIM", "MSE", "PSNR", "UQI", "VIFP"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through all images in the dataset directory
        for image_filename in os.listdir(dataset_dir):
            if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(dataset_dir, image_filename)
            image = Image.open(image_path).convert('RGB')
            input_image = transform(image).unsqueeze(0).to(device)

            # Step 1: Send the image to encoder, decoder, and classifier
            latent_vector = encoder(input_image)[2]
            reconstructed_image = decoder(latent_vector)
            input_predicted_label = classifier(latent_vector)
            predicted_class = "STOP" if torch.argmax(input_predicted_label, dim=1).item() == 0 else "GO"
            print(f"Input Image Predicted Label: {predicted_class}")

            # Step 2: YOLOv5 Detection
            results = model(image)
            detections = results.xyxy[0]
            if len(detections) == 0:
                print("No objects detected in the image. Skipping further steps.")
                continue

            print(f"Number of objects detected: {len(detections)}")
            classes_detected = [results.names[int(det[5])] for det in detections]
            print(f"Classes detected: {classes_detected}")

            for det in detections:
                print(f"Detected Object: {results.names[int(det[5])]} (Confidence: {det[4]:.2f})")

            # Step 3-5: Iterate through detected objects for counterfactual generation
            output_summary = []
            for obj_index, detection in enumerate(detections):
                print(f"\nProcessing Object {obj_index + 1}/{len(detections)}...")
                x_min, y_min, x_max, y_max = map(int, detection[:4])
                confidence = detection[4].item()
                cls_label = results.names[int(detection[5])]

                # Mask object
                masked_image = input_image.clone().squeeze().permute(1, 2, 0).cpu().numpy()
                masked_image[y_min:y_max, x_min:x_max] = 0
                masked_image_tensor = transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

                # Step 4: Reconstruction and metrics
                masked_latent_vector = encoder(masked_image_tensor)[2]
                reconstructed_masked_image = decoder(masked_latent_vector)
                metrics_masked = calculate_metrics(input_image, reconstructed_masked_image)

                # Step 5: Counterfactual generation
                masked_predicted_label = torch.argmax(classifier(masked_latent_vector), dim=1).item()
                masked_predicted_class = "STOP" if masked_predicted_label == 0 else "GO"

                print(f"Original Label: {predicted_class}, Masked Label: {masked_predicted_class}")
                if masked_predicted_class != predicted_class:
                    print(f"Counterfactual Explanation Found: Label changed from {predicted_class} to {masked_predicted_class} by masking object {obj_index + 1}.")
                    counterfactual_found = True

                    # Store summary details only if counterfactual explanation is found
                    summary = {
                        "Image File": image_filename,
                        "Prediction": predicted_class,
                        "Grid Size & Position": f"({x_min}, {y_min}), ({x_max}, {y_max})",
                        "Grid Position": f"({x_min}, {y_min}), ({x_max}, {y_max})",
                        "Counterfactual Found": counterfactual_found,
                        "Confidence": confidence,
                        "SSIM": metrics_masked["SSIM"],
                        "MSE": metrics_masked["MSE"],
                        "PSNR": metrics_masked["PSNR"],
                        "UQI": metrics_masked["UQI"],
                        "VIFP": metrics_masked["VIFP"]
                    }
                    output_summary.append(summary)

            # Save the summary to the CSV file
            for summary in output_summary:
                writer.writerow(summary)

# Process train and test datasets
process_dataset('dataset/town7_dataset/train/', 'counterfactual_summary_train.csv')
process_dataset('dataset/town7_dataset/test/', 'counterfactual_summary_test.csv')
