import torch
import os
import numpy as np
import csv
import sys
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi
from torchvision import transforms
from PIL import Image

#Load models
def load_models(encoder_path, decoder_path, classifier_path, device, latent_dims=128):
    
    # Add Python path to include the directory where 'encoder.py' is located
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
    )
    from encoder import VariationalEncoder
    from decoder import Decoder
    from classifier import ClassifierModel as Classifier
    
    encoder = VariationalEncoder(latent_dims=latent_dims, num_epochs=100).to(device)
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    classifier = Classifier().to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()
    classifier.eval()
    
    return encoder, decoder, classifier

# Image preprocessing
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# Calculate Reconstruction Metrics
def calculate_image_metrics(original, reconstructed):
    original_np = original.cpu().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_np = reconstructed.cpu().squeeze().numpy().transpose(1, 2, 0)
    original_np = (original_np * 255).astype(np.uint8)
    reconstructed_np = (reconstructed_np * 255).astype(np.uint8)

    metrics = {
        "SSIM": ssim(original_np, reconstructed_np, channel_axis=-1),
        "MSE": mse(original_np, reconstructed_np),
        "PSNR": psnr(original_np, reconstructed_np),
        "UQI": uqi(original_np, reconstructed_np),
        "VIFP": vifp(original_np, reconstructed_np),
    }
    return metrics

# Write results to a CSV file
def write_results_to_csv(output_file, fieldnames, data):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

