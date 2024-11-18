import csv
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
# test_dir = 'dataset/town7_dataset/train/'
# image_filename = 'town7_000670.png'
# image_path = os.path.join(test_dir, image_filename)

# Define dataset directory and list all image files
datatest_dir = 'dataset/town7_dataset/test/'
image_files = [f for f in os.listdir(datatest_dir) if f.endswith('.png')]
image_paths = [os.path.join(datatest_dir, f) for f in image_files]

# Output file for results
output_file = "plots/grid_based_masking_images/grid_based_counterfactual_results_test.csv"

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])
# Process each image in the dataset
for image_path in image_paths:  # Iterate through each image path
    image = Image.open(image_path).convert('RGB')  # Open the image
    input_image = transform(image).unsqueeze(0).to(device)  # Preprocess the image
    
    # Your existing processing code for the image goes here
    # Example:
    print(f"Processing image: {image_path}")

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

# apply grid based masking on the Inut Image
def grid_masking(input_image, grid_size=(10, ), mask_value=0, pos=0):
      # Mask the first detected object
    masked_image = input_image.clone().squeeze().permute(1, 2, 0).cpu().numpy()
    
    # compute x and y coordinates for the grid field identified by pos
    x, y = grid_size
    x_pos = pos % x
    y_pos = pos // x
    x_size = masked_image.shape[1] // x
    y_size = masked_image.shape[0] // y
    x_start = x_pos * x_size
    y_start = y_pos * y_size
    x_end = x_start + x_size
    y_end = y_start + y_size
    masked_image[y_start:y_end, x_start:x_end] = mask_value
    


    # Convert masked image back to tensor
    return transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

min_confidence = 0.0
counterfactual_found = False

# Define grid sizes to try
grid_sizes = [(10, 5), (4, 2)]

# Loop over the grid sizes
for grid in grid_sizes:
    grid_rows, grid_cols = grid
    print(f"Trying grid size: {grid_rows}x{grid_cols}")
    
    # Iterate over the grid positions
    for pos in range(grid_rows * grid_cols):
        # Apply grid-based masking
        grid_based_masked_image = grid_masking(input_image, grid_size=grid, pos=pos)

        # Pass the masked image through encoder, decoder, and classifier
        latent_vector_after_masking = encoder(grid_based_masked_image)[2]
        reconstructed_image_after_masking = decoder(latent_vector_after_masking)
        predicted_class_after_masking = classifier(latent_vector_after_masking)

        # Convert the reconstructed image after masking to a PIL image
        reconstructed_image_after_masking_pil = transforms.ToPILImage()(reconstructed_image_after_masking.squeeze(0).cpu())

        # Compute confidence of the predicted class after masking
        confidence = F.softmax(predicted_class_after_masking, dim=1)[0]
        predicted_label_after_masking = torch.argmax(predicted_class_after_masking, dim=1).item()
        predicted_class_after_masking = "STOP" if predicted_label_after_masking == 0 else "GO"

        # Check if the predicted class has changed (counterfactual generated)
        if predicted_class_after_masking != predicted_class:
            print(f"Counterfactual explanation generated at grid position {pos}")
            print(f"Grid Position: {pos}, Confidence: {confidence}")

            # Check if the confidence is above the minimum threshold
            if confidence[0] > min_confidence or confidence[1] > min_confidence:
                counterfactual_found = True
                break  # Stop searching for this grid size

    # If a counterfactual was found, stop checking other grid sizes
    if counterfactual_found:
        break

# Final output
if counterfactual_found:
    print(f"Counterfactual explanation generated for grid size {grid} at position {pos}")
    print(f"Reconstructed Image after Masking Predicted Label: {predicted_class_after_masking}")
else:
    print("No counterfactual explanation generated for any grid size.")

# Plot the original image, reconstructed image, and reconstructed image after grid-based masking
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(reconstructed_image_pil)
# plt.title("Reconstructed Image")
# plt.axis('off')
# masked image
plt.subplot(1, 3, 2)
plt.imshow(grid_based_masked_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
plt.title(f"Masked Image\n(Grid Size: {grid[0]}x{grid[1]}, Pos: {pos})")
plt.axis('off')

if counterfactual_found:
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_image_after_masking_pil)
    plt.title(f"Reconstructed Image after Masking\n(Grid Size: {grid[0]}x{grid[1]}, Pos: {pos})")
    plt.axis('off')

# Save the plots
plt.savefig("plots/grid_based_masking_images/grid_based_masking_result.png")

def calculate_image_metrics(original_image, modified_image):
    original_np = original_image.cpu().squeeze().numpy().transpose(1, 2, 0)
    modified_np = modified_image.cpu().squeeze().numpy().transpose(1, 2, 0)
    original_np = (original_np * 255).astype(np.uint8)
    modified_np = (modified_np * 255).astype(np.uint8)
    
    # Print the dimensions to debug
    print(f"Original image shape (for SSIM): {original_np.shape}")
    print(f"Modified image shape (for SSIM): {modified_np.shape}")
    
    # Dynamically calculate win_size
    min_dim = min(original_np.shape[0], original_np.shape[1])
    win_size = min(min_dim, 7)  # Ensure win_size <= min_dim
    if win_size % 2 == 0:       # Ensure win_size is odd
        win_size -= 1
    if win_size < 3:            # Minimum valid win_size
        win_size = 3

    print(f"Calculated win_size: {win_size}")
    
    # Calculate metrics with corrected SSIM API
    metrics = {
        "SSIM": ssim(original_np, modified_np, win_size=win_size, channel_axis=-1),  # Use channel_axis instead of multichannel
        "MSE": mse(original_np, modified_np),
        "PSNR": psnr(original_np, modified_np),
        "UQI": uqi(original_np, modified_np),
        "VIFP": vifp(original_np, modified_np),
    }
    return metrics

# Open CSV file for writing results
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow([
        "Image File", "Prediction", "Grid Size", "Grid Position", 
        "Counterfactual Found", "Confidence", "SSIM", "MSE", "PSNR", "UQI", "VIFP"
    ])
    
    # Process each image in the dataset
    for image_filename in image_files:
        image_path = os.path.join(datatest_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        input_image = transform(image).unsqueeze(0).to(device)
        
        # Initial prediction
        latent_vector = encoder(input_image)[2]
        reconstructed_input_image = decoder(latent_vector)
        input_image_predicted_label = classifier(latent_vector)
        predicted_label = torch.argmax(input_image_predicted_label, dim=1).item()
        predicted_class = "STOP" if predicted_label == 0 else "GO"
        
        # Try grid sizes
        counterfactual_found = False
        confidence = None
        grid_size_found = None
        grid_position_found = None
        metrics = {}
        
        for grid in grid_sizes:
            grid_rows, grid_cols = grid
            for pos in range(grid_rows * grid_cols):
                # Apply grid-based masking
                grid_based_masked_image = grid_masking(input_image, grid_size=grid, pos=pos)
                latent_vector_after_masking = encoder(grid_based_masked_image)[2]
                reconstructed_image_after_masking = decoder(latent_vector_after_masking)
                predicted_class_after_masking = classifier(latent_vector_after_masking)
                confidence = F.softmax(predicted_class_after_masking, dim=1)[0]
                predicted_label_after_masking = torch.argmax(predicted_class_after_masking, dim=1).item()
                predicted_class_after_masking = "STOP" if predicted_label_after_masking == 0 else "GO"

                # Check for counterfactual explanation
                if predicted_class_after_masking != predicted_class:
                    counterfactual_found = True
                    grid_size_found = grid
                    grid_position_found = pos
                    metrics = calculate_image_metrics(
                        input_image.squeeze(),
                        grid_based_masked_image.squeeze(),
                    )
                    break
            if counterfactual_found:
                break
        
        # Save results for the current image
        writer.writerow([
            image_filename,
            predicted_class,
            grid_size_found,
            grid_position_found,
            counterfactual_found,
            confidence.tolist() if confidence is not None else None,  # Extract confidence as a list
            metrics.get("SSIM", None),
            metrics.get("MSE", None),
            metrics.get("PSNR", None),
            metrics.get("UQI", None),
            metrics.get("VIFP", None)
        ])

        print(f"Processed {image_filename} - Counterfactual Found: {counterfactual_found}")

print(f"Results saved to {output_file}.")