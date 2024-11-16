import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import seaborn as sns
from PIL import Image
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
image_filename = 'town7_000060.png'
image_path = os.path.join(test_dir, image_filename)

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0).to(device)

# Step 1: Pass the input image through encoder, decoder, and classifier
latent_vector = encoder(input_image)[2]
reconstructed_input_image = decoder(latent_vector)
input_image_predicted_label = classifier(latent_vector)

# Convert the reconstructed image tensor to a PIL image for plotting
reconstructed_image_pil = transforms.ToPILImage()(reconstructed_input_image.squeeze(0).cpu())

# Print the predicted label
predicted_label = torch.argmax(input_image_predicted_label, dim=1).item()
predicted_class = "STOP" if predicted_label == 0 else "GO"
print(f'Input Image Predicted Label: {predicted_class}')

# Step 2: Perform object detection using YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True, autoshape=True)
results = model(image)  # Pass the actual PIL image

# Step 3: Counterfactual generation if objects are detected
detections = results.xyxy[0]
if len(detections) > 0:
    # Mask the first detected object
    masked_image = input_image.clone().squeeze().permute(1, 2, 0).cpu().numpy()
    detection_to_mask = detections[0]  # Mask one object at a time
    x_min, y_min, x_max, y_max, _, _ = map(int, detection_to_mask[:6])

    # Ensure bounding box coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(masked_image.shape[1], x_max)
    y_max = min(masked_image.shape[0], y_max)

    # Create a mask by setting pixel values to 0 (black out the detected region)
    masked_image[y_min:y_max, x_min:x_max] = 0

    # Convert masked image back to tensor
    masked_image_tensor = transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

    # Step 4: Pass masked image through encoder to get latent vector and reconstruct
    masked_latent_vector = encoder(masked_image_tensor)[2]
    reconstructed_masked_image = decoder(masked_latent_vector).squeeze(0)
    reconstructed_masked_image_pil = transforms.ToPILImage()(reconstructed_masked_image.cpu())
    reconstructed_masked_image_pil = reconstructed_masked_image_pil.resize((160, 80))

    # Step 5: Classifier prediction for the masked reconstructed image
    masked_image_predicted_label = classifier(masked_latent_vector)
    predicted_label_after_masking = torch.argmax(masked_image_predicted_label, dim=1).item()
    predicted_class_after_masking = "STOP" if predicted_label_after_masking == 0 else "GO"
    print(f'Reconstructed Image after Masking Predicted Label: {predicted_class_after_masking}')

    # Check if counterfactual explanation is generated
    if predicted_class_after_masking != predicted_class:
        print(f"Counterfactual Explanation Generated: The label has changed from {predicted_class} to {predicted_class_after_masking}")

        # Calculate image quality metrics between input image and reconstructed masked image
        reconstructed_masked_image_np = np.array(reconstructed_masked_image_pil, dtype=np.float32) / 255.0
        original_image_np = np.array(image.resize(reconstructed_masked_image_pil.size), dtype=np.float32) / 255.0

        counterfactual_metrics = {
            "MSE": mse(original_image_np, reconstructed_masked_image_np),
            "SSIM": ssim(original_image_np, reconstructed_masked_image_np, win_size=5, channel_axis=-1, data_range=1.0),
            "PSNR": psnr(original_image_np, reconstructed_masked_image_np, data_range=1.0),
            "VIF": vifp(original_image_np, reconstructed_masked_image_np),
            "UQI": uqi(original_image_np, reconstructed_masked_image_np),
        }

        # Print counterfactual metrics
        for metric, value in counterfactual_metrics.items():
            print(f'{metric} for Counterfactual: {value}')

        # Plot metrics comparison
        metrics = ['MSE', 'SSIM', 'PSNR', 'VIF', 'UQI']
        values = [counterfactual_metrics[metric] for metric in metrics]

        plt.figure(figsize=(10, 5))
        sns.barplot(x=metrics, y=values, palette='viridis', hue=metrics, dodge=False, legend=False)
        plt.title('Comparison of Image Quality Metrics for Counterfactual')
        plt.ylabel('Metric Value')
        plt.savefig('plots/object_detection_using_yolov5/metrics_comparison.png')
    else:
        print("No Counterfactual Explanation Generated: The label remains the same.")

    # Plot original, masked, and reconstructed images side by side
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))

    # Original Image
    original_image = input_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    axes[0].imshow(original_image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    image.save('plots/object_detection_using_yolov5/input_image.png')

    # Masked Image
    axes[1].imshow(masked_image)
    axes[1].set_title('Masked Image')
    axes[1].axis('off')
    plt.imsave('plots/object_detection_using_yolov5/masked_image.png', masked_image)

    # Reconstructed Masked Image
    axes[2].imshow(reconstructed_masked_image_pil)
    axes[2].set_title('Reconstructed Masked Image')
    axes[2].axis('off')
    reconstructed_masked_image_pil.save('plots/object_detection_using_yolov5/reconstructed_masked_image.png')

    plt.tight_layout()
    plt.savefig('plots/object_detection_using_yolov5/all_images.png')
else:
    print("No objects detected in the image. Skipping counterfactual generation.")
