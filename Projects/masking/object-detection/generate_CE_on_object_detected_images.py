import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels

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

# Select a random image from the test directory
image_file = "dataset/town7_dataset/test/"
csv_file = "dataset/town7_dataset/test/labeled_test_data_log.csv"

# Load pre-trained YOLO model (YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Step 1: Preprocess the selected image
def pre_process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)

# Step 2: Get actual label from the CSV
def get_actual_label(image_name):
    label_row = df[df['image_filename'] == image_name]
    if label_row.empty:
        raise ValueError(f"Image {image_name} not found in the CSV file.")
    label = label_row['label'].values[0]
    actual_label = 0 if label == "STOP" else 1
    return actual_label

# Keep selecting images until an object is detected
counterfactual_generated = False
while not counterfactual_generated:
    # Select a random image from the test dataset
    image_files = [f for f in os.listdir(image_file) if f.endswith('.png')]
    image_dir = os.path.join(image_file, random.choice(image_files))
    print(f"Selected image: {image_dir}")

    # Load and process image
    image = pre_process_image(image_dir)

    original_image_name = os.path.basename(image_dir)
    actual_label = get_actual_label(original_image_name)
    print(f"Image {original_image_name} is selected and its label is: {'STOP' if actual_label == 0 else 'GO'}")

    # Step 3: Perform object detection using YOLO
    results = model(image_dir)
    detections = results.xyxy[0]

    if len(detections) == 0:
        print(f"No objects detected in image: {image_dir}. Skipping image.")
        continue

    # Step 4: Start counterfactual generation process
    while not counterfactual_generated and len(detections) > 0:
        # Step 5: Create a copy of the image to apply masks
        masked_image = image.clone().squeeze().permute(1, 2, 0).cpu().numpy()

        # Step 6: Mask the first detected object
        detection_to_mask = detections[0]  # Mask one object at a time
        x_min, y_min, x_max, y_max, conf, cls = map(int, detection_to_mask[:6])

        # Ensure bounding box coordinates are within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(masked_image.shape[1], x_max)
        y_max = min(masked_image.shape[0], y_max)

        # Create a mask by setting pixel values to 0 (black out the detected region)
        masked_image[y_min:y_max, x_min:x_max] = 0

        # Step 7: Convert masked image back to tensor
        masked_image_tensor = transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

        # Step 8: Pass masked image through encoder to get latent vector
        encoder_output = encoder(masked_image_tensor)
        latent_vector = encoder_output[0]
        print(f"Latent vector for the masked image: {latent_vector}")

        # Step 9: Classifier prediction for the original latent vector
        classifier_prediction = torch.softmax(classifier(latent_vector), dim=1)
        predicted_class = torch.argmax(classifier_prediction, dim=1).item()
        predicted_class_label = "STOP" if predicted_class == 0 else "GO"
        print(f"Prediction for the latent vector: {classifier_prediction} (Class: {predicted_class_label})")

        # Step 10: Reconstruct the masked image from the latent vector
        reconstructed_image = F.interpolate(decoder(latent_vector), size=(80, 160), mode='bilinear', align_corners=False).squeeze().permute(1, 2, 0).cpu().detach().numpy()

        # Step 11: Plot original, masked, and reconstructed images side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        original_image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Masked image
        axes[1].imshow(masked_image)
        axes[1].set_title('Masked Image')
        axes[1].axis('off')

        # Reconstructed image
        axes[2].imshow(reconstructed_image)
        axes[2].set_title('Reconstructed Image')
        axes[2].axis('off')

        plt.tight_layout()
        reconstructed_image_path = os.path.join('plots/object_detection_using_yolov5/reconstructed_images', f'reconstructed_{original_image_name}')
        plt.savefig(reconstructed_image_path)
        print(f"Reconstructed image saved at: {reconstructed_image_path}")

        # Step 12: Classifier prediction for the reconstructed image
        reconstructed_image_tensor = transforms.ToTensor()(reconstructed_image).unsqueeze(0).to(device)
        re_encoded_output = encoder(reconstructed_image_tensor)
        re_encoded_latent_vector = re_encoded_output[0]
        print(f"Latent vector for the reconstructed image: {re_encoded_latent_vector}")

        # Step 13: Classifier prediction for the re-encoded latent vector
        re_encoded_prediction = torch.softmax(classifier(re_encoded_latent_vector), dim=1)
        re_encoded_class = torch.argmax(re_encoded_prediction, dim=1).item()
        re_encoded_class_label = "STOP" if re_encoded_class == 0 else "GO"
        print(f"Prediction for the re-encoded latent vector: {re_encoded_prediction} (Class: {re_encoded_class_label})")

        # Step 14: Compare the reconstructed image label with the original label
        if re_encoded_class != actual_label:
            print(f"Counterfactual explanation generated: The masked object influenced the decision. {image_dir}")
            counterfactual_generated = True
        else:
            print("No significant change in prediction after masking. Retrying with another object.")
            # Remove the current detection and retry
            detections = detections[1:]

    if not counterfactual_generated:
        print("No object left to mask. Counterfactual explanation could not be generated.")
    else:
        print("Counterfactual explanation process completed.")
