import os
import sys
from sympy import per
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
from lime.lime_tabular import LimeTabularExplainer
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

# Load median values computed earlier
median_values_csv = "latent_vectors/combined_median_values.csv"
# median_df = pd.read_csv(median_values_csv)
# median_values = median_df.values.flatten()  # Flatten to get a list of median values for each feature

# Load mean, min, max, and median values computed earlier
mean_values_csv = "latent_vectors/mean_values.csv"
min_values_csv = "latent_vectors/min_values.csv"
max_values_csv = "latent_vectors/max_values.csv"


mean_values = pd.read_csv(mean_values_csv).values.flatten()
min_values = pd.read_csv(min_values_csv).values.flatten()
max_values = pd.read_csv(max_values_csv).values.flatten()
median_values = pd.read_csv(median_values_csv).values.flatten()

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
# test_dir = 'dataset/town7_dataset/train/'
# random_image_filename = random.choice(os.listdir(test_dir))
# print(f'Random Image Filename: {random_image_filename}')
# image_path = os.path.join(test_dir, random_image_filename)

# manually select an image instead of choosing randomly
test_dir = 'dataset/town7_dataset/train/'
image_filename = 'town7_000060.png'
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
reconstructed_input_image = decoder(latent_vector)    # Decode the latent vector to reconstruct the image
input_image_predicted_label = classifier(latent_vector)  # Predict the label

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

# Image quality metrics for original vs reconstructed image
metrics = {
    "MSE": mse(input_image_resized, reconstructed_image_np),
    "SSIM": ssim(input_image_resized, reconstructed_image_np, win_size=5, channel_axis=-1, data_range=1.0),
    "PSNR": psnr(input_image_resized, reconstructed_image_np, data_range=1.0),
    "VIF": vifp(input_image_resized, reconstructed_image_np),
    "UQI": uqi(input_image_resized, reconstructed_image_np),
}

# Print image quality metrics
for metric, value in metrics.items():
    print(f'{metric}: {value}')

# Bar chart for metric comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), hue=list(metrics.keys()), palette='viridis', dodge=False, legend=False)
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
latent_vector_np = latent_vector.cpu().detach().numpy().reshape(1, -1)

# Define a function that LIME will use to simulate changes in the latent space
def predict_with_latent(latent):
    latent_tensor = torch.tensor(latent, dtype=torch.float32).to(device)
    output = classifier(latent_tensor)
    return F.softmax(output, dim=1).cpu().detach().numpy()

# Set up the LIME explainer for latent space
explainer = LimeTabularExplainer(latent_vector_np, mode='classification', feature_names=[f'latent_{i}' for i in range(latent_vector_np.shape[1])], discretize_continuous=False)

# Explain the latent representation with percentage-based feature selection
explanation = explainer.explain_instance(latent_vector_np.flatten(), predict_with_latent, num_features=len(latent_vector_np.flatten()))

# Sort features by their importance and filter positively contributing features
positive_importance_list = sorted(
    [(feature, weight) for feature, weight in explanation.as_list() if weight > 0],
    key=lambda x: abs(x[1]), reverse=True
)

# Define masking strategies
masking_strategies = {
    "median": median_values,
    "mean": mean_values,
    "min": min_values,
    "max": max_values
}

# Iterate through masking strategies
for strategy_name, masking_values in masking_strategies.items():
    print(f"\nApplying {strategy_name.capitalize()}-Based Masking...")
    #loop to 
    percentage_value = 0
    step_size = 0.01

    #state the loops with the percentage value and do the iteration util it find the couterfactul explanation
    while(percentage_value < 1):
        percentage_value += step_size

        # Calculate how many features to select based on the top percentage
        num_features_to_select = int(len(positive_importance_list) * percentage_value)  # Adjust the percentage as needed

        # Print important features being masked
        important_features = [int(feature.split("_")[-1]) for feature, _ in positive_importance_list[:num_features_to_select]]
        print("{:<15} {:<20} {:<20}".format('Feature Index', 'Original Value', 'Median Value'))
        print("{:<15} {:<20} {:<20}".format('-' * 15, '-' * 20, '-' * 20))

        # Apply median masking based on LIME's important features
        masked_latent_vector = latent_vector_np.flatten()
        for feature_index in important_features:
            original_value = masked_latent_vector[feature_index]
            median_value = median_values[feature_index]
            print("{:<15} {:<20} {:<20}".format(feature_index, original_value, median_value))
            masked_latent_vector[feature_index] = median_value

        # Convert masked latent vector back to tensor
        masked_latent_tensor = torch.tensor(masked_latent_vector, dtype=torch.float32).to(device).reshape(1, -1)

        # Reconstruct the image using the masked latent vector
        reconstructed_image_after_masking = decoder(masked_latent_tensor).squeeze(0)
        reconstructed_image_after_masking_pil = transforms.ToPILImage()(reconstructed_image_after_masking)

        # Counterfactual analysis
        reconstructed_image_after_masking_tensor = transform(reconstructed_image_after_masking_pil).unsqueeze(0).to(device)
        masked_latent_vector_cf = encoder(reconstructed_image_after_masking_tensor)[2]
        masked_image_predicted_label = classifier(masked_latent_vector_cf)
        predicted_label_after_masking = torch.argmax(masked_image_predicted_label, dim=1).item()
        predicted_class_after_masking = "STOP" if predicted_label_after_masking == 0 else "GO"
        print(f'Reconstructed Image after Masking Predicted Label: {predicted_class_after_masking}')

        if predicted_class_after_masking != predicted_class:
            break
            
    # Print the percentage value
    print(f"Percentage Value: {percentage_value}")
    print("*******")

    # Check if counterfactual explanation is generated
    if predicted_class_after_masking != predicted_class:
        print(f"Counterfactual Explanation Generated: The label has changed from {predicted_class} to {predicted_class_after_masking}")

        # Metrics for counterfactual explanation
        reconstructed_image_after_masking_np = np.array(reconstructed_image_after_masking_pil, dtype=np.float32) / 255.0
        counterfactual_metrics = {
            "MSE": mse(input_image_resized, reconstructed_image_after_masking_np),
            "SSIM": ssim(input_image_resized, reconstructed_image_after_masking_np, win_size=5, channel_axis=-1, data_range=1.0),
            "PSNR": psnr(input_image_resized, reconstructed_image_after_masking_np, data_range=1.0),
            "VIF": vifp(input_image_resized, reconstructed_image_after_masking_np),
            "UQI": uqi(input_image_resized, reconstructed_image_after_masking_np),
        }

        # Print metrics for counterfactual explanation
        for metric, value in counterfactual_metrics.items():
            print(f'{metric} for Counterfactual: {value}')

        # Bar chart for metric comparison between original and counterfactual
        original_values = list(metrics.values())
        counterfactual_values = list(counterfactual_metrics.values())

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, original_values, width, label='Original', color='blue')
        ax.bar(x + width / 2, counterfactual_values, width, label='Counterfactual', color='orange')

        ax.set_xlabel('Metrics', fontsize=14)
        ax.set_ylabel('Values', fontsize=14)
        ax.set_title('Comparison of Image Quality Metrics: Original vs Counterfactual', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys())
        ax.legend()

        plt.tight_layout()
        plt.savefig('plots/lime_plots/metrics_comparison_original_vs_counterfactual.png')
        print(f"Conclusion: The masked features significantly impacted the classification outcome, changing the label from {predicted_class} to {predicted_class_after_masking}. This indicates that these features {important_features} play a crucial role in the model's decision-making process, providing valuable insight into the model's behavior.")
    else:
        print("No Counterfactual Explanation Generated: The label remains the same.")


# Plot original, reconstructed, and counterfactual images together with the same size
fig, axes = plt.subplots(1, 3, figsize=(25, 10))

# Original Input Image
axes[0].imshow(image.resize(reconstructed_image.size))
axes[0].set_title('Input Image', fontsize=20)
axes[0].axis('off')
image.save('plots/lime_plots/Input_image.png')

# Reconstructed Image
axes[1].imshow(reconstructed_image)
axes[1].set_title('Reconstructed Input Image', fontsize=20)
axes[1].axis('off')
reconstructed_image.save('plots/lime_plots/Reconstructed_input_image.png')

# Reconstructed Image after Median Masking
reconstructed_image_after_masking_resized = reconstructed_image_after_masking_pil.resize(reconstructed_image.size)
axes[2].imshow(reconstructed_image_after_masking_resized)
axes[2].set_title('LIME Reconstructed Image after Median Masking', fontsize=20)
axes[2].axis('off')
reconstructed_image_after_masking_pil.save('plots/lime_plots/Reconstructed_image_after_masking.png')

plt.tight_layout()
plt.savefig('/home/selab/darshan/git-repos/plots/lime_plots/all_images.png')