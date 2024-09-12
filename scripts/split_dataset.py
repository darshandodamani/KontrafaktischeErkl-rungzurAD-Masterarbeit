import os
import random
import shutil
import csv

data_path = "dataset/town7_dataset/"
csv_file = os.path.join(data_path, "data_log.csv")

train_folder = os.path.join(data_path, "train")
test_folder = os.path.join(data_path, "test")
train_csv = os.path.join(train_folder, "train_data_log.csv")
test_csv = os.path.join(test_folder, "test_data_log.csv")

image_extensions = [".png"]

# List all files with the specified extensions
image_list = [
    filename
    for filename in os.listdir(data_path)
    if os.path.splitext(filename)[-1] in image_extensions
]

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

# Open CSV files for train and test sets
with (
    open(csv_file, "r") as original_csv,
    open(train_csv, "w", newline="") as train_csvfile,
    open(test_csv, "w", newline="") as test_csvfile,
):
    reader = csv.DictReader(original_csv)
    train_writer = csv.DictWriter(train_csvfile, fieldnames=reader.fieldnames)
    test_writer = csv.DictWriter(test_csvfile, fieldnames=reader.fieldnames)

    train_writer.writeheader()
    test_writer.writeheader()

    # Split the images into training and testing sets
    for i, row in enumerate(reader):
        image_filename = row["image_filename"]
        if image_filename in image_list[:train_size]:
            dest_folder = train_folder
            train_writer.writerow(row)
        else:
            dest_folder = test_folder
            test_writer.writerow(row)
        shutil.copy(
            os.path.join(data_path, image_filename),
            os.path.join(dest_folder, image_filename),
        )
