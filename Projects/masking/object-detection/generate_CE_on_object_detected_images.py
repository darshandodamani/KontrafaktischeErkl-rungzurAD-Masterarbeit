# location: Projects/masking/object-detection/generated_CE_on_object_detected_images.py
import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import seaborn as sns
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
test_dir = 'dataset/town7_dataset/train/'
image_filename = 'town7_000967.png'
image_path = os.path.join(test_dir, image_filename)

# # Select and preprocess the image
# test_dir = 'dataset/town7_dataset/train/'
# image_filename = random.choice(os.listdir(test_dir))
# print(f"Selected Image: {image_filename}")
# image_path = os.path.join(test_dir, image_filename)

# YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, autoshape=True)

# Helper function to calculate metrics
def calculate_metrics(original, reconstructed):
    # Ensure both images have the same shape
    original_np = original.cpu().squeeze().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.cpu().squeeze().permute(1, 2, 0).detach().numpy()
    
    # Resize reconstructed to match original's dimensions
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
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0).to(device)

# Step 1: Send the image to encoder, decoder, and classifier
latent_vector = encoder(input_image)[2]
reconstructed_image = decoder(latent_vector)
input_predicted_label = classifier(latent_vector)
predicted_class = "STOP" if torch.argmax(input_predicted_label, dim=1).item() == 0 else "GO"
print(f"Input Image Predicted Label: {predicted_class}")

# Calculate metrics for reconstructed image
metrics_initial = calculate_metrics(input_image, reconstructed_image)
print("Metrics for initial reconstruction:")
for metric, value in metrics_initial.items():
    print(f"{metric}: {value}")

# Step 2: YOLOv5 Detection
results = model(image)
detections = results.xyxy[0]
if len(detections) == 0:
    print("No objects detected in the image. Skipping further steps.")
    sys.exit()

print(f"Number of objects detected: {len(detections)}")
classes_detected = [results.names[int(det[5])] for det in detections]
print(f"Classes detected: {classes_detected}")

for det in detections:
    print(f"Detected Object: {results.names[int(det[5])]} (Confidence: {det[4]:.2f})")


# Visualize detections
draw = ImageDraw.Draw(image)
for det in detections:
    x_min, y_min, x_max, y_max, conf, cls = map(int, det[:6])
    label = results.names[cls]
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    draw.text((x_min, y_min), label, fill="red")
image.show()

# Step 3-5: Iterate through detected objects for counterfactual generation
for obj_index, detection in enumerate(detections):
    print(f"\nProcessing Object {obj_index + 1}/{len(detections)}...")
    x_min, y_min, x_max, y_max = map(int, detection[:4])

    # Mask object
    masked_image = input_image.clone().squeeze().permute(1, 2, 0).cpu().numpy()
    masked_image[y_min:y_max, x_min:x_max] = 0
    masked_image_tensor = transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

    # Step 4: Reconstruction and metrics
    masked_latent_vector = encoder(masked_image_tensor)[2]
    reconstructed_masked_image = decoder(masked_latent_vector)
    metrics_masked = calculate_metrics(input_image, reconstructed_masked_image)
    # print(f"Metrics after masking object {obj_index + 1}: {metrics_masked}")
    print("Metrics for initial reconstruction:")
    for metric, value in metrics_masked.items():
        print(f"{metric}: {value}")

    # Step 5: Counterfactual generation
    masked_predicted_label = torch.argmax(classifier(masked_latent_vector), dim=1).item()
    masked_predicted_class = "STOP" if masked_predicted_label == 0 else "GO"

    print(f"Original Label: {predicted_class}, Masked Label: {masked_predicted_class}")
    if masked_predicted_class != predicted_class:
        print(f"Counterfactual Explanation Found: Label changed from {predicted_class} to {masked_predicted_class} by masking object {obj_index + 1}.")
        break
else:
    print("No counterfactual explanation generated for any detected object.")
    
# Metrics for plotting
metrics_labels = ["SSIM", "MSE", "PSNR", "VIFP", "UQI"]
initial_metrics_values = [
    metrics_initial["SSIM"],
    metrics_initial["MSE"],
    metrics_initial["PSNR"],
    metrics_initial["VIFP"],
    metrics_initial["UQI"],
]
masked_metrics_values = [
    metrics_masked["SSIM"],
    metrics_masked["MSE"],
    metrics_masked["PSNR"],
    metrics_masked["VIFP"],
    metrics_masked["UQI"],
]

# Plot metrics comparison
plt.figure(figsize=(10, 6))
plt.plot(metrics_labels, initial_metrics_values, marker="o", label="Initial Reconstruction")
plt.plot(metrics_labels, masked_metrics_values, marker="o", label="Masked Reconstruction")
plt.title("Metrics Comparison for Counterfactual Explanation")
plt.xlabel("Metrics")
plt.ylabel("Values")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("plots/object_detection_using_yolov5/metrics_comparison.png")
plt.show()

# Prepare images for visualization
# 1. Input image with bounding boxes
input_with_boxes = image.copy()
draw = ImageDraw.Draw(input_with_boxes)
for det in detections:
    x_min, y_min, x_max, y_max = map(int, det[:4])
    label = results.names[int(det[5])]
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
    draw.text((x_min, y_min), label, fill="red")

# 2. Masked input image
masked_image_np = masked_image_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
masked_image_pil = Image.fromarray((masked_image_np * 255).astype(np.uint8))

# 3. Reconstructed masked image
reconstructed_masked_image_np = reconstructed_masked_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
reconstructed_masked_image_pil = Image.fromarray((reconstructed_masked_image_np * 255).astype(np.uint8))

# Resize all images to the same size for consistent plotting
resize_size = (160, 80)
input_with_boxes = input_with_boxes.resize(resize_size)
masked_image_pil = masked_image_pil.resize(resize_size)
reconstructed_masked_image_pil = reconstructed_masked_image_pil.resize(resize_size)

# Plot images side by side
fig, axes = plt.subplots(1, 4, figsize=(16, 6))

axes[0].imshow(input_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(input_with_boxes)
axes[1].set_title("Object Detection")
axes[1].axis("off")

axes[2].imshow(masked_image_pil)
axes[2].set_title("Masked Image")
axes[2].axis("off")

axes[3].imshow(reconstructed_masked_image_pil)
axes[3].set_title("Reconstructed Image")
axes[3].axis("off")

plt.tight_layout()
plt.savefig("plots/object_detection_using_yolov5/image_visualization.png")

