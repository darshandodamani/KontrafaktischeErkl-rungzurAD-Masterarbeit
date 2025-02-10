import os
import shutil
import re

# ------------------------------------------------------------------------------
# 📌 Configuration: Define Paths
# ------------------------------------------------------------------------------
# Path to Intersections file
INTERSECTIONS_FILE = "results/method_comparision/Intersections_4.txt"

# Define source directories for images
GRID_ORIGINAL_DIR = "images/grid_based/original"
GRID_RECONSTRUCTED_DIR = "images/grid_based/reconstructed"
LIME_IMAGE_RECONSTRUCTED_DIR = "images/lime_on_images/reconstructed"
LIME_LATENT_RECONSTRUCTED_DIR = "images/lime_on_latent/reconstructed"

# Destination directory for extracted images
DEST_DIR = "counterfactual_comparison/static/images/"
os.makedirs(DEST_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 📌 Step 1: Read Intersections File and Extract Image Filenames
# ------------------------------------------------------------------------------
def extract_filenames_from_intersections(file_path):
    """
    Extracts image filenames that belong to 'Grid-Based Masking, LIME on Latent Features, and LIME on Images'.
    
    Args:
        file_path (str): Path to the intersections text file.
    
    Returns:
        list: List of selected image filenames.
    """
    selected_images = []
    
    with open(file_path, "r") as file:
        lines = file.readlines()
    
    # Regex pattern to match the line containing "Grid-Based Masking and LIME on Latent Features and LIME on Images"
    pattern = r"Grid-Based Masking and LIME on Latent Features and LIME on Images:\s*\d+,\s*\[(.*?)\]"
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            images_str = match.group(1)
            selected_images = [img.strip().strip("'") for img in images_str.split(",")]
            break  # Found the line, no need to continue
    
    return selected_images

# ------------------------------------------------------------------------------
# 📌 Step 2: Copy and Rename Selected Images
# ------------------------------------------------------------------------------
def copy_and_rename_images(image_list, dest_dir):
    """
    Copies and renames images from source directories into the destination directory.
    
    Args:
        image_list (list): List of image filenames to be copied.
        dest_dir (str): Destination directory where images will be stored.
    """
    for idx, image_name in enumerate(image_list, start=1):
        # Source Paths
        original_path = os.path.join(GRID_ORIGINAL_DIR, image_name)
        grid_reconstructed_path = os.path.join(GRID_RECONSTRUCTED_DIR, image_name)
        lime_image_reconstructed_path = os.path.join(LIME_IMAGE_RECONSTRUCTED_DIR, image_name)
        lime_latent_reconstructed_path = os.path.join(LIME_LATENT_RECONSTRUCTED_DIR, image_name)

        # Destination Filenames
        original_dest = os.path.join(dest_dir, f"original{idx}.png")
        method1_dest = os.path.join(dest_dir, f"method1_cf{idx}.png")
        method2_dest = os.path.join(dest_dir, f"method2_cf{idx}.png")
        method3_dest = os.path.join(dest_dir, f"method3_cf{idx}.png")

        # Copying Files (Only if they exist)
        if os.path.exists(original_path):
            shutil.copy(original_path, original_dest)
            print(f"✅ Copied Original: {original_path} → {original_dest}")
        else:
            print(f"⚠ Warning: {original_path} not found!")

        if os.path.exists(grid_reconstructed_path):
            shutil.copy(grid_reconstructed_path, method1_dest)
            print(f"✅ Copied Grid-Based: {grid_reconstructed_path} → {method1_dest}")
        else:
            print(f"⚠ Warning: {grid_reconstructed_path} not found!")

        if os.path.exists(lime_image_reconstructed_path):
            shutil.copy(lime_image_reconstructed_path, method2_dest)
            print(f"✅ Copied LIME on Images: {lime_image_reconstructed_path} → {method2_dest}")
        else:
            print(f"⚠ Warning: {lime_image_reconstructed_path} not found!")

        if os.path.exists(lime_latent_reconstructed_path):
            shutil.copy(lime_latent_reconstructed_path, method3_dest)
            print(f"✅ Copied LIME on Latent Features: {lime_latent_reconstructed_path} → {method3_dest}")
        else:
            print(f"⚠ Warning: {lime_latent_reconstructed_path} not found!")

# ------------------------------------------------------------------------------
# 📌 Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🔍 Extracting Image Filenames from Intersections File...")
    selected_images = extract_filenames_from_intersections(INTERSECTIONS_FILE)

    if selected_images:
        print(f"✅ Found {len(selected_images)} images to process.")
        copy_and_rename_images(selected_images, DEST_DIR)
        print("\n🎯 All images successfully copied and renamed!")
    else:
        print("⚠ No images found in the intersections file. Check the format or filename pattern.")
