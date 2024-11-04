import os
import torch
import cv2
import numpy as np
import sys
import random
import matplotlib.pyplot as plt

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

# Load pre-trained YOLO model (YOLOv5) and print version
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
# print(f'YOLO model version: {model._version}')

# Define classes to mask (e.g., vehicles, pedestrians, traffic lights, traffic signs)
classes_to_mask = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'traffic light', 'stop sign', 'sign board']

# Load 10 random images from the dataset
dataset_dir = 'dataset/town7_dataset/test'
image_files = os.listdir(dataset_dir)

# # Ensure there are enough images in the dataset
# dataset_size = 10
# if len(image_files) < dataset_size:
#     raise FileNotFoundError(f"Not enough images found in directory: {dataset_dir}")

selected_images = []

# Randomly select 10 images from the dataset
while len(selected_images) < dataset_size:
    image_filename = random.choice(image_files)
    image_path = os.path.join(dataset_dir, image_filename)

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        continue

    # Perform object detection using YOLOv5
    results = model(image_path)
    detections = results.xyxy[0]  # Get bounding box coordinates

    if len(detections) == 0:
        print(f"No objects detected in image: {image_filename}. Skipping image.")
        continue

    # Filter detections to mask only specific classes
    filtered_detections = [d for d in detections if model.names[int(d[5])] in classes_to_mask]

    if len(filtered_detections) == 0:
        print(f"No relevant objects detected in image: {image_filename}. Skipping image.")
        continue

    selected_images.append(image_filename)

    # Create copies of the original image for further processing
    original_image = image.copy()  # Original image for reference
    detected_image = original_image.copy()  # Image with detected objects highlighted
    masked_image = image.copy()  # Image with masked regions

    # Iterate over filtered bounding boxes and mask the respective regions
    for detection in filtered_detections:
        x_min, y_min, x_max, y_max, conf, cls = map(int, detection[:6])

        # Ensure bounding box coordinates are within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        # Create a mask by setting pixel values to 0 (black out the detected region)
        masked_image[y_min:y_max, x_min:x_max] = 0

        # Draw bounding box on detected image for visualization
        class_name = model.names[int(cls)]
        cv2.rectangle(detected_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)  # Draw rectangle with red color

    # Plot the original image, detected objects, and masked image
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot the original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Plot the image with detected objects highlighted
    axes[1].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detected Objects with Confidence")
    axes[1].axis("off")

    # Print detected object information on the plot
    for idx, detection in enumerate(filtered_detections):
        x_min, y_min, x_max, y_max, conf, cls = detection[:6]
        class_name = model.names[int(cls)]
        axes[1].text(5, 15 + 20 * idx, f"{class_name} {float(conf):.2f}", fontsize=8, color='red')

    # Plot the masked image
    axes[2].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Masked Image")
    axes[2].axis("off")

    # Save the combined plot
    if not os.path.exists('plots/object_detection_using_yolov5'):
        os.makedirs('plots/object_detection_using_yolov5')
    plt.tight_layout()
    plt.savefig(f'plots/object_detection_using_yolov5/plot_{image_filename}_1.png')
    plt.show()

    # Additional plot: Original image with detected objects vs corresponding mask
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot original image with detected objects
    axes[0].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image with Detected Objects")
    axes[0].axis("off")

    # Plot the corresponding masked image
    axes[1].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Masked Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f'plots/object_detection_using_yolov5/plot_{image_filename}_2.png')
    plt.show()

    # Save individual images (original, detected, and masked)
    masked_image_path = f'plots/object_detection_using_yolov5/masked_image_{image_filename}'
    cv2.imwrite(masked_image_path, masked_image)
    # cv2.imwrite(f'plots/object_detection_using_yolov5/original_image_{image_filename}', original_image)
    # cv2.imwrite(f'plots/object_detection_using_yolov5/detected_image_{image_filename}', detected_image)

    # Print information about what YOLO is masking
    for detection in filtered_detections:
        x_min, y_min, x_max, y_max, conf, cls = detection[:6]
        class_name = model.names[int(cls)]
        print(f"Masked object: {class_name} with confidence {conf:.2f} in image: {image_filename}")
