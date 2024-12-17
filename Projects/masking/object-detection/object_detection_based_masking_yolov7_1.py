# location: Projects/masking/object-detection/object_detection_based_masking_yolov7.py
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
from torchvision.ops import nms  # Add NMS function

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

# YOLOv7 Model
# Make sure to download 'yolov7.pt' and place it in the current directory or specify the correct path
try:
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7.pt', trust_repo=True)
except FileNotFoundError as e:
    print("YOLOv7 weights not found. Ensure 'yolov7.pt' is in the correct path.")
    raise e

model.eval()

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

# Mask object function
def mask_object(image_tensor, bbox):
    x_min, y_min, x_max, y_max = map(int, bbox)
    masked_image = image_tensor.clone().squeeze().permute(1, 2, 0).cpu().numpy()
    masked_image[y_min:y_max, x_min:x_max] = 0
    return transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

# Save images function
def save_images(original_image, reconstructed_image, masked_image, reconstructed_masked_image, filename_prefix):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    original_image_np = original_image.cpu().squeeze().permute(1, 2, 0).numpy()
    reconstructed_image_np = reconstructed_image.cpu().squeeze().permute(1, 2, 0).detach().numpy()
    masked_image_np = masked_image.cpu().squeeze().permute(1, 2, 0).numpy()
    reconstructed_masked_image_np = reconstructed_masked_image.cpu().squeeze().permute(1, 2, 0).detach().numpy()

    # Ensure all images are resized to the same shape
    target_shape = original_image_np.shape
    reconstructed_image_np_resized = resize(reconstructed_image_np, target_shape, anti_aliasing=True)
    reconstructed_masked_image_np_resized = resize(reconstructed_masked_image_np, target_shape, anti_aliasing=True)

    axs[0].imshow(original_image_np)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(reconstructed_image_np_resized)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis('off')

    axs[2].imshow(masked_image_np)
    axs[2].set_title("Masked Image")
    axs[2].axis('off')

    axs[3].imshow(reconstructed_masked_image_np_resized)
    axs[3].set_title("Reconstructed Masked Image")
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(f"plots/{filename_prefix}.png")
    plt.close()

# Process dataset function
def process_dataset(dataset_dir, csv_filename):
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])

    images_with_objects = 0
    images_without_objects = 0
    ce_generated_count = 0
    
    start_time = time.time()

    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "Image File", "Prediction", "BBox", "Counterfactual Found",
            "Confidence", "SSIM", "MSE", "PSNR", "UQI", "VIFP"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_filename in os.listdir(dataset_dir):
            if not image_filename.lower().endswith(".png"):
                continue

            image_path = os.path.join(dataset_dir, image_filename)
            image = Image.open(image_path).convert("RGB")
            input_image = transform(image).unsqueeze(0).to(device)

            # Step 1: Encode, decode, classify original
            latent_vector = encoder(input_image)[2]
            reconstructed_image = decoder(latent_vector)
            predicted_label = torch.argmax(classifier(latent_vector), dim=1).item()
            predicted_class = "STOP" if predicted_label == 0 else "GO"

            # Step 2: Object detection using YOLOv7
            results = model(image_path)
            detections = results.pred[0]
            detections_filtered = []

            # Apply Non-Maximum Suppression to filter redundant detections
            if len(detections) > 0:
                boxes = detections[:, :4]  # Bounding box coordinates
                scores = detections[:, 4]  # Confidence scores
                nms_indices = nms(boxes, scores, iou_threshold=0.5)
                detections = detections[nms_indices]

            # Check number of objects detected
            num_objects = len(detections)
            if num_objects == 0:
                images_without_objects += 1
                continue

            images_with_objects += 1
            print(f"Image: {image_filename}, Objects Detected: {num_objects}")

            # Step 3: Process detections (select max 2 objects to mask)
            objects_processed = 0
            for detection in detections:
                x_min, y_min, x_max, y_max, confidence, class_id = detection.tolist()
                if confidence < 0.5:
                    continue

                # Scale bounding box back to original image size
                orig_w, orig_h = image.size
                x_scale = orig_w / 640
                y_scale = orig_h / 640
                x_min = int(x_min * x_scale)
                x_max = int(x_max * x_scale)
                y_min = int(y_min * y_scale)
                y_max = int(y_max * y_scale)

                # Mask the object
                masked_image_tensor = mask_object(input_image, [x_min, y_min, x_max, y_max])

                # Step 4: Encode and classify masked image
                masked_latent_vector = encoder(masked_image_tensor)[2]
                re_encoded_label = torch.argmax(classifier(masked_latent_vector), dim=1).item()
                re_encoded_class = "STOP" if re_encoded_label == 0 else "GO"

                # Step 5: Reconstruction of masked image
                reconstructed_masked_image = decoder(masked_latent_vector)

                # Counterfactual detection
                counterfactual_found = re_encoded_class != predicted_class

                if counterfactual_found:
                    ce_generated_count += 1
                    print(f"Counterfactual generated for object in image: {image_filename}")

                    # Step 6: Metrics
                    metrics = calculate_metrics(input_image, reconstructed_image)
                    save_images(
                        input_image, reconstructed_image, masked_image_tensor, reconstructed_masked_image,
                        f"ce_{image_filename.split('.')[0]}_{objects_processed+1}"
                    )

                    writer.writerow({
                        "Image File": image_filename,
                        "Prediction": predicted_class,
                        "BBox": f"({x_min}, {y_min}, {x_max}, {y_max})",
                        "Counterfactual Found": counterfactual_found,
                        "Confidence": confidence,
                        "SSIM": metrics["SSIM"],
                        "MSE": metrics["MSE"],
                        "PSNR": metrics["PSNR"],
                        "UQI": metrics["UQI"],
                        "VIFP": metrics["VIFP"],
                    })

                    objects_processed += 1
                    if objects_processed >= 2:
                        break

    end_time = time.time()  # End time tracking
    print(f"Time taken to process dataset: {end_time - start_time:.2f} seconds")
    print(f"Images with objects: {images_with_objects}")
    print(f"Images without objects: {images_without_objects}")
    print(f"Counterfactual explanations generated: {ce_generated_count}")

# Process train and test datasets
process_dataset('dataset/town7_dataset/train/', 'results_yolov7_train.csv')
process_dataset('dataset/town7_dataset/test/', 'results_yolov7_test.csv')
