import os
import sys
import csv
import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
import matplotlib.pyplot as plt

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder")))

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from initialize_masking_pipeline import initialize_method_results, update_method_results, initial_predictions_csv

# Paths
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"
output_csv = "plots/grid_based_masking_results.csv"
test_dir = "dataset/town7_dataset/test/"
plot_dir = "plots/grid_based_masking_images"
os.makedirs(plot_dir, exist_ok=True)

# Load models
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

# Transformations
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

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


def process_grid_based_masking():
    df_initial = pd.read_csv(initial_predictions_csv)
    df_results = pd.read_csv(output_csv)
    
    for _, row in df_initial.iterrows():
        start_time = time.time()
        image_filename = row["Image File"]
        image_path = os.path.join(test_dir, image_filename)
        
        image = Image.open(image_path).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)
        
        counterfactual_found = False
        final_prediction = row["Prediction (Before Masking)"]
        grid_size_found = None
        grid_position_found = None
        confidence_final = None
        metrics = {}
        
        for grid_size in [(10, 5), (4, 2)]:  # Multiple grid sizes
            for pos in range(grid_size[0] * grid_size[1]):
                masked_image = input_image.clone()
                masked_image[:, :, (pos // grid_size[1]) * (80 // grid_size[0]): ((pos // grid_size[1]) + 1) * (80 // grid_size[0]),
                             (pos % grid_size[1]) * (160 // grid_size[1]): ((pos % grid_size[1]) + 1) * (160 // grid_size[1])] = 0
                
                latent_vector_masked = encoder(masked_image)[2]
                masked_prediction = classifier(latent_vector_masked)
                confidence_final = F.softmax(masked_prediction, dim=1).cpu().detach().numpy().flatten()
                confidence_final_str = ", ".join(map(str, confidence_final))
                predicted_label_after_masking = torch.argmax(masked_prediction, dim=1).item()
                final_prediction = "STOP" if predicted_label_after_masking == 0 else "GO"
                
                if final_prediction != row["Prediction (Before Masking)"]:
                    counterfactual_found = True
                    grid_size_found = str(grid_size)
                    grid_position_found = pos
                    metrics = calculate_image_metrics(input_image, masked_image)
                    
                    plot_filename = os.path.join(plot_dir, f"{image_filename}_grid_{grid_size[0]}x{grid_size[1]}_pos_{pos}.png")
                    plot_and_save_images(input_image, decoder(encoder(input_image)[2]), masked_image, decoder(latent_vector_masked), plot_filename)
                    break
            if counterfactual_found:
                break
        
        end_time = time.time()
        total_time_taken = round(end_time - start_time + row["Time Taken (s)"], 5)
        
        df_results.loc[df_results['Image File'] == image_filename, ["Prediction (After Masking)", "Confidence (After Masking)", "Counterfactual Found", "Grid Size", "Grid Position", "SSIM", "MSE", "PSNR", "UQI", "VIFP", "Time Taken (s)"]] = [final_prediction, confidence_final_str, counterfactual_found, grid_size_found, grid_position_found, metrics.get("SSIM", None), metrics.get("MSE", None), metrics.get("PSNR", None), metrics.get("UQI", None), metrics.get("VIFP", None), total_time_taken]
    
    df_results.to_csv(output_csv, index=False)
    print(f"Grid-based masking results saved to {output_csv}")

process_grid_based_masking()