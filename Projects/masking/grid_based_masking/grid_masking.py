import os
import cv2
import numpy as np 
import sys
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save plots without a display
import matplotlib.pyplot as plt
import torch

# Add Python path to include the directory where 'encoder.py' is located (if needed)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the images from the dataset
dataset_dir = 'dataset/town7_dataset/test'
image_files = os.listdir(dataset_dir)

# Ensure there are enough images in the dataset
dataset_size = 10
if len(image_files) < dataset_size:
    raise FileNotFoundError(f"Not enough images found in directory: {dataset_dir}")

selected_images = []

# Randomly select 10 images from the dataset
while len(selected_images) < dataset_size:
    image_filename = random.choice(image_files)
    image_path = os.path.join(dataset_dir, image_filename)

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    selected_images.append(image)

# Perform Grid-based Masking with Random Shifts
def grid_masking_with_shift(image, grid_size_range=(16, 64), keep_ratio=0.5, mask_color=(0, 0, 0)):
    """
    Apply GridMask to the input image with random shifts.

    Args:
        image (numpy.ndarray): Input image.
        grid_size_range (tuple): Range for selecting grid size (default: (16, 64)).
        keep_ratio (float): Ratio of grid units to keep (default: 0.5).
        mask_color (tuple): Color to use for masking (default: (0, 0, 0)).

    Returns:
        numpy.ndarray: Image with grid-based masking applied.
    """
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    # Randomly select grid size within the specified range
    grid_size = random.randint(grid_size_range[0], grid_size_range[1])

    # Random shift for δx and δy
    delta_x = random.randint(0, grid_size)
    delta_y = random.randint(0, grid_size)

    # Loop through the image creating a grid pattern with random shifts
    for i in range(delta_y, h, grid_size):
        for j in range(delta_x, w, grid_size):
            if random.random() > keep_ratio:  # Mask or keep based on keep ratio
                mask[i:i + grid_size, j:j + grid_size] = 0

    # Apply mask to the image
    masked_image = image * mask[..., np.newaxis]  # Preserve color channels

    return masked_image

print("Starting grid-based masking...")

# Select an image for demonstration
image = selected_images[0]
print(f"Selected image dimensions: {image.shape}")

# Create copies of the original image for further processing
original_image = image.copy()  # Original image for reference

# Apply grid-based masking with random shifts to the image
masked_image = grid_masking_with_shift(
    original_image, 
    grid_size_range=(16, 64), 
    keep_ratio=0.5, 
    mask_color=(0, 0, 0)
)
print("Grid-based masking applied.")

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Save original and masked images
original_output_path = os.path.join(output_dir, "original_image.png")
masked_output_path = os.path.join(output_dir, "grid_masked_image.png")
cv2.imwrite(original_output_path, original_image)
cv2.imwrite(masked_output_path, masked_image)

# Save plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
plt.title("Grid-based Masked Image")
plt.axis("off")

plot_output_path = os.path.join(output_dir, "grid_masking_plot.png")
plt.savefig(plot_output_path)
print(f"Images and plot saved to {output_dir}")
