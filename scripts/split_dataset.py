import os
import random
import shutil

data_path = "town7_dataset/"

train_folder = os.path.join(data_path, 'train')
test_folder = os.path.join(data_path, 'test')

image_extensions = ['.png']

# List all files with the specified extensions
image_list = [filename for filename in os.listdir(data_path) if os.path.splitext(filename)[-1] in image_extensions]

# Set a seed for reproducibility
random.seed(42)

# Shuffle the list of images
random.shuffle(image_list)

# Calculate the split sizes
train_size = int(len(image_list) * 0.8)

# Create train and test folders if they don't exist
for folder_path in [train_folder, test_folder]:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Split the images into training and testing sets
for i, f in enumerate(image_list):
    if i < train_size:
        dest_folder = train_folder
    else:
        dest_folder = test_folder
    shutil.copy(os.path.join(data_path, f), os.path.join(dest_folder, f))
