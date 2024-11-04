# location: Projects/masking/object-detection/object_detection_based_masking.py
import os
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Load pre-trained YOLO model (YOLOv5)
# force_model=True | can make it True if you want to reload the model again it means if I don't want to use the cache model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True) 

# Define classes to mask (e.g., vehicles, pedestrians, traffic lights, traffic signs)
# classes_to_mask = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic light', 'stop sign', 'sign board']
# Mask all detected objects
classes_to_mask = model.names

# Load all images from the dataset
dataset_dir = 'dataset/town7_dataset/test' 
csv_dir = 'dataset/town7_dataset/test/labeled_test_data_log.csv'
image_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

# Load CSV file to keep track of labels and detected objects
df = pd.read_csv(csv_dir)
results_list = []

# Initialize counters for statistics
total_images = len(image_files)
images_with_detections = 0
images_masked = 0

# Iterate over all images in the dataset
for image_filename in image_files:
    image_path = os.path.join(dataset_dir, image_filename)

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_filename}. Skipping.")
        continue

    # Perform object detection using YOLOv5
    results = model(image)
    detections = results.xyxy[0]  # Get bounding box coordinates

    if len(detections) == 0:
        print(f"No objects detected in image: {image_filename}. Skipping image.")
        continue

    images_with_detections += 1
    images_masked += 1

    # Create copies of the original image for further processing
    original_image = image.copy()
    detected_image = image.copy()
    masked_image = image.copy()

    # Iterate over detected bounding boxes and mask the respective regions
    for detection in detections:
        x_min, y_min, x_max, y_max, conf, cls = detection[:6]
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

        # Ensure bounding box coordinates are within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)

        # Mask the detected region
        masked_image[y_min:y_max, x_min:x_max] = 0

        # Draw bounding box on detected image for visualization
        class_name = model.names[int(cls)]
        cv2.rectangle(detected_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)

        # Append detection information to results list
        results_list.append({
            'image_filename': image_filename,
            'class_name': class_name,
            'confidence': float(conf),
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        })

    # Plot the original image, detected objects, and masked image
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detected Objects with Confidence")
    axes[1].axis("off")

    for idx, detection in enumerate(detections):
        x_min, y_min, x_max, y_max, conf, cls = detection[:6]
        class_name = model.names[int(cls)]
        axes[1].text(5, 15 + 20 * idx, f"{class_name} {float(conf):.2f}", fontsize=8, color='red')

    axes[2].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Masked Image")
    axes[2].axis("off")

    final_plot_dir = 'plots/object_detection_using_yolov5/final_object_detection_images_with_original'
    os.makedirs(final_plot_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{final_plot_dir}/plot_{image_filename}_1.png')
    plt.close(fig)

    # Save individual images (masked)
    masked_image_dir = 'plots/object_detection_using_yolov5/masked_images'
    os.makedirs(masked_image_dir, exist_ok=True)
    masked_image_path = f'{masked_image_dir}/masked_image_{image_filename}'
    cv2.imwrite(masked_image_path, masked_image)

    # Print information about what YOLO is masking
    for detection in detections:
        x_min, y_min, x_max, y_max, conf, cls = detection[:6]
        class_name = model.names[int(cls)]
        print(f"Masked object: {class_name} with confidence {conf:.2f} in image: {image_filename}")

# Save detection results to CSV
results_df = pd.DataFrame(results_list)
detection_results_csv_path = 'plots/object_detection_using_yolov5/detection_results.csv'
results_df.to_csv(detection_results_csv_path, index=False)
print(f"Detection results saved to: {detection_results_csv_path}")

# Save a copy of the original labeled CSV file in the masked images directory
masked_csv_path = os.path.join(masked_image_dir, 'labeled_test_data_log.csv')
masked_df = df[df['image_filename'].isin(results_df['image_filename'])].copy()
for i, row in masked_df.iterrows():
    image_filename = row['image_filename']
    masked_df.at[i, 'image_filename'] = f'masked_image_{image_filename}'
masked_df.to_csv(masked_csv_path, index=False)
print(f"Labeled test data CSV saved to: {masked_csv_path}")

# Plot summary statistics
fig, ax = plt.subplots(figsize=(10, 6))
labels = ['Total Images', 'Images with Detections', 'Images Masked']
values = [total_images, images_with_detections, images_masked]

ax.bar(labels, values, color=['blue', 'green', 'red'])
ax.set_ylabel('Number of Images')
ax.set_title('Summary of Object Detection and Masking Results')

summary_plot_dir = 'plots/object_detection_using_yolov5'
os.makedirs(summary_plot_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(f'{summary_plot_dir}/summary_plot.png')
plt.show()