from calendar import c
from idna import encode
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import sys
import os
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from median_calculator import compute_dataset_medians
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
import cv2
import random
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi

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

# Load median values computed earlier
median_values_csv = "latent_vectors/combined_median_values.csv"
median_df = pd.read_csv(median_values_csv)
median_values = median_df.values.flatten()  # Flatten to get a list of median values for each feature

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

# Fetch a random image from the dataset/test/ directory
# test_dir = 'dataset/town7_dataset/test/'
# random_image_filename = random.choice(os.listdir(test_dir))
# print(f'Random Image Filename: {random_image_filename}')
# image_path = os.path.join(test_dir, random_image_filename)

# manually select an image instead of choosing randomly
test_dir = 'dataset/town7_dataset/train/'
image_filename = 'town7_000276.png'
image_path = os.path.join(test_dir, image_filename)

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((80, 160)),  # Resize image to match input size of the models
    transforms.ToTensor()           # Convert image to PyTorch tensor
])
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0)  # Add batch dimension

# Move input image to the same device as the model
input_image = input_image.to(device)

# Pass the image through the encoder, decoder, and classifier
latent_vector = encoder(input_image)[2]               # Encode the image
reconstructed_input_image = decoder(latent_vector)        # Decode the latent vector to reconstruct the image
input_image_predicted_label = classifier(latent_vector)         # Predict the label

# Convert the reconstructed image tensor to a PIL image for plotting
reconstructed_image = reconstructed_input_image.squeeze(0)  # Remove batch dimension
reconstructed_image = transforms.ToPILImage()(reconstructed_image)

# Print the predicted label
predicted_label = torch.argmax(input_image_predicted_label, dim=1).item()
predicted_class = "STOP" if predicted_label == 0 else "GO"
print(f'Input Image Predicted Label: {predicted_class}')

# Resize images to the same dimensions before comparison
input_image_resized = np.array(image.resize(reconstructed_image.size), dtype=np.float32) / 255.0
reconstructed_image_np = np.array(reconstructed_image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

# MSE (Mean Squared Error)
mse_value = mse(input_image_resized, reconstructed_image_np)
print(f'Mean Squared Error (MSE): {mse_value}')

# SSIM (Structural Similarity Index)
ssim_value = ssim(input_image_resized, reconstructed_image_np, win_size=5, channel_axis=-1, data_range=1.0)
print(f'Structural Similarity Index (SSIM): {ssim_value}')

# PSNR (Peak Signal-to-Noise Ratio)
psnr_value = psnr(input_image_resized, reconstructed_image_np, data_range=1.0)
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value}')

# VIF (Visual Information Fidelity)
vif_value = vifp(input_image_resized, reconstructed_image_np)
print(f'Visual Information Fidelity (VIF): {vif_value}')

# UQI (Universal Quality Index)
uqi_value = uqi(input_image_resized, reconstructed_image_np)
print(f'Universal Quality Index (UQI): {uqi_value}')



# Bar chart for metric comparison
metrics = ['MSE', 'SSIM', 'PSNR', 'VIF', 'UQI']
values = [mse_value, ssim_value, psnr_value, vif_value, uqi_value]
plt.figure(figsize=(10, 5))
sns.barplot(x=metrics, y=values, palette='viridis', hue=metrics, dodge=False, legend=False)
plt.title('Comparison of Image Quality Metrics')
plt.ylabel('Metric Value')
plt.savefig('plots/lime_plots/metrics_comparison.png')

# Heatmap of Absolute Differences
difference = np.abs(input_image_resized - reconstructed_image_np).mean(axis=-1)  # Calculate absolute difference for each channel and average
plt.figure(figsize=(10, 5))
sns.heatmap(difference, cmap='hot', cbar=True)
plt.title('Heatmap of Differences')
plt.axis('off')
plt.savefig('plots/lime_plots/heatmap_differences.png')

# LIME-Based Masking on Latent Vector
from lime import lime_tabular

# Convert latent vector to numpy for LIME
latent_vector_np = latent_vector.cpu().detach().numpy().reshape(1, -1)

# Define a function that LIME will use to simulate changes in the latent space
def predict_with_latent(latent):
    latent_tensor = torch.tensor(latent, dtype=torch.float32).to(device)
    output = classifier(latent_tensor)
    return F.softmax(output, dim=1).cpu().detach().numpy()

# Set up the LIME explainer for latent space
explainer = lime_tabular.LimeTabularExplainer(latent_vector_np, mode='classification', feature_names=[f'latent_{i}' for i in range(latent_vector_np.shape[1])], discretize_continuous=False)

# Explain the latent representation with percentage-based feature selection
explanation = explainer.explain_instance(latent_vector_np.flatten(), predict_with_latent, num_features=len(latent_vector_np.flatten()))

# Sort features by their importance and filter positively and negatively contributing features
positive_importance_list = [
    (feature, weight) for feature, weight in explanation.as_list() if weight > 0
]  # Keep only positive influence features

negative_importance_list = [
    (feature, weight) for feature, weight in explanation.as_list() if weight < 0
]  # Keep only negative influence features

# Sort positively contributing features in descending order by absolute weight
positive_importance_list = sorted(positive_importance_list, key=lambda x: abs(x[1]), reverse=True)

# Sort negatively contributing features in descending order by absolute weight
negative_importance_list = sorted(negative_importance_list, key=lambda x: abs(x[1]), reverse=True)

# Calculate how many features to select based on the top percentage
num_features_to_select = int(len(positive_importance_list) * 0.05)  # Adjust the percentage as needed

# Plot the explanation to visualize both positive and negative feature importance
fig, ax = plt.subplots(figsize=(20, 15))

# Extract feature names and their corresponding importance values for positive and negative features
positive_feature_names, positive_importance_values = zip(*positive_importance_list[:num_features_to_select])
negative_feature_names, negative_importance_values = zip(*negative_importance_list[:num_features_to_select])

# Assign colors based on the sign of importance: green for positive, red for negative
colors = ['green'] * len(positive_importance_values) + ['red'] * len(negative_importance_values)

# Combine positive and negative features for plotting
total_feature_names = list(positive_feature_names) + list(negative_feature_names)
total_importance_values = list(positive_importance_values) + list(negative_importance_values)
print(f"Total Features: {total_feature_names}")

# Set y-axis ticks and labels before plotting
y_pos = range(len(total_feature_names))
ax.set_yticks(y_pos)
ax.set_yticklabels(total_feature_names, fontsize=14, fontweight='bold')

# Plot a bar chart with thicker bars and color coding
ax.barh(y_pos, total_importance_values, color=colors, height=0.8)

# Set labels and title
ax.set_ylabel("Latent Features", fontsize=16, fontweight='bold')
ax.set_xlabel("Feature Importance", fontsize=16, fontweight='bold')
plt.title(f"LIME Explanation: Top {int(num_features_to_select / len(positive_importance_list) * 100)}% Positive and Negative Feature Importance", fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig("/home/selab/darshan/git-repos/plots/lime_plots/lime_explanation_combined_features.png")

# Apply median masking based on LIME's important features
masked_latent_vector = latent_vector_np.flatten()
important_features = [int(feature.split("_")[-1]) for feature, _ in positive_importance_list[:num_features_to_select]]
print(f"Features being masked: {important_features}")
for feature_index in important_features:
    masked_latent_vector[feature_index] = median_values[feature_index]  # Set the important feature to its median value

# Convert masked latent vector back to tensor
masked_latent_tensor = torch.tensor(masked_latent_vector, dtype=torch.float32).to(device).reshape(1, -1)

# Reconstruct the image using the masked latent vector
reconstructed_image_after_masking = decoder(masked_latent_tensor).squeeze(0)
reconstructed_image_after_masking_pil = transforms.ToPILImage()(reconstructed_image_after_masking)

# Now, send the reconstructed image after masking to counterfactual Step.
# To do that, send the image to the encoder and get the latent vector and send that to the classifier to get the label
reconstructed_image_after_masking_tensor = transform(reconstructed_image_after_masking_pil).unsqueeze(0).to(device)
masked_latent_vector_cf = encoder(reconstructed_image_after_masking_tensor)[2]
masked_image_predicted_label = classifier(masked_latent_vector_cf)
predicted_label_after_masking = torch.argmax(masked_image_predicted_label, dim=1).item()
predicted_class_after_masking = "STOP" if predicted_label_after_masking == 0 else "GO"
print(f'Reconstructed Image after Masking Predicted Label: {predicted_class_after_masking}')

# Check if counterfactual explanation is generated
if predicted_class_after_masking != predicted_class:
    print("Counterfactual Explanation Generated: The label has changed from {predicted_class} to {predicted_class_after_masking}")

    # Measure metrics for the counterfactual explanation
    reconstructed_image_after_masking_np = np.array(reconstructed_image_after_masking_pil, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    # MSE (Mean Squared Error) for Counterfactual Explanation
    mse_cf_value = mse(input_image_resized, reconstructed_image_after_masking_np)
    print(f'Mean Squared Error (MSE) for Counterfactual: {mse_cf_value}')

    # SSIM (Structural Similarity Index) for Counterfactual Explanation
    ssim_cf_value = ssim(input_image_resized, reconstructed_image_after_masking_np, win_size=5, channel_axis=-1, data_range=1.0)
    print(f'Structural Similarity Index (SSIM) for Counterfactual: {ssim_cf_value}')

    # PSNR (Peak Signal-to-Noise Ratio) for Counterfactual Explanation
    psnr_cf_value = psnr(input_image_resized, reconstructed_image_after_masking_np, data_range=1.0)
    print(f'Peak Signal-to-Noise Ratio (PSNR) for Counterfactual: {psnr_cf_value}')

    # VIF (Visual Information Fidelity) for Counterfactual Explanation
    vif_cf_value = vifp(input_image_resized, reconstructed_image_after_masking_np)
    print(f'Visual Information Fidelity (VIF) for Counterfactual: {vif_cf_value}')

    # UQI (Universal Quality Index) for Counterfactual Explanation
    uqi_cf_value = uqi(input_image_resized, reconstructed_image_after_masking_np)
    print(f'Universal Quality Index (UQI) for Counterfactual: {uqi_cf_value}')

    # Bar chart for metric comparison between original and counterfactual
    metrics = ['MSE', 'SSIM', 'PSNR', 'VIF', 'UQI']
    original_values = [mse_value, ssim_value, psnr_value, vif_value, uqi_value]
    counterfactual_values = [mse_cf_value, ssim_cf_value, psnr_cf_value, vif_cf_value, uqi_cf_value]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, original_values, width, label='Original', color='blue')
    ax.bar(x + width/2, counterfactual_values, width, label='Counterfactual', color='orange')

    ax.set_xlabel('Metrics', fontsize=14)
    ax.set_ylabel('Values', fontsize=14)
    ax.set_title('Comparison of Image Quality Metrics: Original vs Counterfactual', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig('plots/lime_plots/metrics_comparison_original_vs_counterfactual.png')
else:
    print("No Counterfactual Explanation Generated: The label remains the same.")
# print the input image and the reconstructed image after masking
print(f'Input Image Prediction was :  {predicted_class}, Reconstructed Image after Masking Predicted Label: {predicted_class_after_masking}')


# Plot original and reconstructed images together with the same size
fig, axes = plt.subplots(1, 3, figsize=(25, 10))

# Original Input Image
axes[0].imshow(image.resize(reconstructed_image.size))
axes[0].set_title('Input Image', fontsize=20)
axes[0].axis('off')
# Save the original image
image.save('plots/lime_plots/Input_image.png')

# Reconstructed Image
axes[1].imshow(reconstructed_image)
axes[1].set_title('Reconstructed Input Image', fontsize=20)
axes[1].axis('off')
# Save the reconstructed input image
reconstructed_image.save('plots/lime_plots/Reconstructed_input_image.png')

# Reconstructed Image after Median Masking
reconstructed_image_after_masking_resized = reconstructed_image_after_masking_pil.resize(reconstructed_image.size)
axes[2].imshow(reconstructed_image_after_masking_resized)
axes[2].set_title('LIME Reconstructed Image after Median Masking', fontsize=20)
axes[2].axis('off')
# Save the reconstructed image after masking
reconstructed_image_after_masking_pil.save('plots/lime_plots/Reconstructed_image_after_masking.png')

plt.tight_layout()
plt.savefig('/home/selab/darshan/git-repos/plots/lime_plots/all_images.png')