import os
import csv
import time
import sys
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import matplotlib.pyplot as plt

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# Paths
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"
train_dir = "dataset/town7_dataset/train/"
test_dir = "dataset/town7_dataset/test/"
output_train_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_train.csv"
output_test_csv = "plots/grid_based_masking_images/grid_based_counterfactual_results_test.csv"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
classifier = Classifier().to(device)

encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))   
classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

encoder.eval()
decoder.eval()
classifier.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

# Function to calculate metrics
def calculate_image_metrics(original_image, modified_image):
    original_np = original_image.cpu().squeeze().numpy().transpose(1, 2, 0)
    modified_np = modified_image.cpu().squeeze().numpy().transpose(1, 2, 0)
    original_np = (original_np * 255).astype(np.uint8)
    modified_np = (modified_np * 255).astype(np.uint8)
    
    metrics = {
        "SSIM": round(ssim(original_np, modified_np, channel_axis=-1), 5),
        "MSE": round(mse(original_np, modified_np), 5),
        "PSNR": round(psnr(original_np, modified_np), 5),
        "UQI": round(uqi(original_np, modified_np), 5),
        "VIFP": round(vifp(original_np, modified_np), 5),
    }
    return metrics

# Grid masking function
def grid_masking(input_image, grid_size=(10, 5), mask_value=0, pos=0):
    masked_image = input_image.clone().squeeze().permute(1, 2, 0).cpu().numpy()
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
    return transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

# Function to save images
def plot_and_save_images(input_image, reconstructed_image, masked_image, reconstructed_masked_image, filename):
    """
    Save a plot of the input image, reconstructed image, grid-masked image, and reconstructed grid-masked image.
    """
    # Convert tensors to numpy arrays for visualization
    input_np = input_image.cpu().squeeze().permute(1, 2, 0).numpy()
    masked_np = masked_image.cpu().squeeze().permute(1, 2, 0).numpy()

    # Resize reconstructed images to match the input image size
    input_size = input_image.size()[2:]  # Height and Width of the input image
    reconstructed_np = F.interpolate(reconstructed_image, size=input_size, mode="bilinear", align_corners=False).cpu().squeeze().permute(1, 2, 0).detach().numpy()
    reconstructed_masked_np = F.interpolate(reconstructed_masked_image, size=input_size, mode="bilinear", align_corners=False).cpu().squeeze().permute(1, 2, 0).detach().numpy()

    # Create the plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(input_np)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(reconstructed_np)
    axs[1].set_title("Reconstructed Image")
    axs[1].axis("off")

    axs[2].imshow(masked_np)
    axs[2].set_title("Grid-Masked Image")
    axs[2].axis("off")

    axs[3].imshow(reconstructed_masked_np)
    axs[3].set_title("Reconstructed Masked Image")
    axs[3].axis("off")

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Saved plot to: {filename}")


# Function to process dataset
def process_dataset(image_dir, output_csv, grid_sizes=[(10, 5), (4, 2)]):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    # Open CSV file for writing results
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow([
            "Image File", "Initial Prediction Class", "Final Prediction Class",
            "Counterfactual Found", "Grid Size", "Grid Position", 
            "Confidence Initial", "Confidence Final", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"
        ])

        total_time = 0  # Track total time for processing all images

        # Ensure plot directory exists
        plot_dir = "plots/grid_based_masking_images"
        os.makedirs(plot_dir, exist_ok=True)

        # Process each image
        for image_filename in image_files:
            image_path = os.path.join(image_dir, image_filename)
            start_time = time.time()
            try:
                image = Image.open(image_path).convert('RGB')
                input_image = transform(image).unsqueeze(0).to(device)
                
                # Initial prediction
                latent_vector = encoder(input_image)[2]
                input_image_predicted_label = classifier(latent_vector)
                predicted_label = torch.argmax(input_image_predicted_label, dim=1).item()
                initial_prediction_class = "STOP" if predicted_label == 0 else "GO"
                confidence_initial = F.softmax(input_image_predicted_label, dim=1)[0]

                # Grid masking
                counterfactual_found = False
                final_prediction_class = None
                grid_size_found = None
                grid_position_found = None
                confidence_final = None
                metrics = {}

                for grid in grid_sizes:
                    grid_rows, grid_cols = grid
                    for pos in range(grid_rows * grid_cols):
                        # Apply grid-based masking
                        grid_based_masked_image = grid_masking(input_image, grid_size=grid, pos=pos)
                        latent_vector_after_masking = encoder(grid_based_masked_image)[2]
                        predicted_class_after_masking = classifier(latent_vector_after_masking)
                        confidence_final = F.softmax(predicted_class_after_masking, dim=1)[0]
                        predicted_label_after_masking = torch.argmax(predicted_class_after_masking, dim=1).item()
                        final_prediction_class = "STOP" if predicted_label_after_masking == 0 else "GO"

                        # Check for counterfactual explanation
                        if final_prediction_class != initial_prediction_class:
                            counterfactual_found = True
                            grid_size_found = grid
                            grid_position_found = pos
                            metrics = calculate_image_metrics(
                                input_image.squeeze(),
                                grid_based_masked_image.squeeze(),
                            )

                            # Save plots
                            plot_filename = os.path.join(
                                plot_dir,
                                f"plot_{image_filename.split('.')[0]}_grid_{grid_rows}x{grid_cols}_pos_{pos}.png"
                            )
                            plot_and_save_images(
                                input_image,
                                decoder(latent_vector),
                                grid_based_masked_image,
                                decoder(latent_vector_after_masking),
                                plot_filename
                            )
                            break
                    if counterfactual_found:
                        break

                end_time = time.time()
                time_taken = round(end_time - start_time, 2)
                total_time += time_taken

                # Save results for the current image
                writer.writerow([
                    image_filename,
                    initial_prediction_class,
                    final_prediction_class,
                    counterfactual_found,
                    grid_size_found,
                    grid_position_found,
                    [round(c, 5) for c in confidence_initial.tolist()] if confidence_initial is not None else None,
                    [round(c, 5) for c in confidence_final.tolist()] if confidence_final is not None else None,
                    metrics.get("SSIM", None),
                    metrics.get("MSE", None),
                    metrics.get("PSNR", None),
                    metrics.get("UQI", None),
                    metrics.get("VIFP", None),
                    time_taken
                ])
                print(f"Processed {image_filename} - Counterfactual Found: {counterfactual_found} - Time: {time_taken}s")
            except Exception as e:
                print(f"Error processing {image_filename}: {e}")

        print(f"Results saved to {output_csv}. Total Time Taken: {total_time:.2f} seconds.")

# Process train and test datasets
process_dataset(train_dir, output_train_csv)
process_dataset(test_dir, output_test_csv)
