import sys
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PilImage
import torchvision.transforms as transforms
# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel

# Set device for GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the image
def load_image(image_path):
    return PilImage.open(image_path).convert("RGB")

# Function to transform the image
def transform_image(image, transform):
    return transform(image).unsqueeze(0)

# Load the VAE encoder and decoder models
def load_vae_encoder():
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    encoder.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/epochs_500_latent_128/var_encoder_model.pth",
            map_location=device, weights_only=True,
        )
    )
    encoder.eval()
    return encoder

def load_vae_decoder():
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    decoder.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/epochs_500_latent_128/decoder_model.pth",
            map_location=device, weights_only=True,
        )
    )
    decoder.eval()
    return decoder

# Load classifier
def load_classifier():
    classifier = ClassifierModel(input_size=128, hidden_size=128, output_size=2).to(device)
    classifier.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/epochs_500_latent_128/classifier_final.pth",
            map_location=device,
            weights_only=True,
        )
    )
    classifier.eval()
    return classifier

# Get the latent vector from an image using the encoder
def get_latent_vector(image, encoder, transform):
    image_tensor = transform(image).unsqueeze(0).to(device)  # Convert image to tensor and add batch dimension
    with torch.no_grad():
        mu, logvar, z = encoder(image_tensor)  # Get the latent vector (mu, logvar, z)
    return z

# Randomly mask a subset of the latent vector
def randomly_mask_latent(latent_vector, num_features_to_mask=3):
    latent_vector_masked = latent_vector.clone()  # Clone the original latent vector to keep it intact
    total_features = latent_vector.size(1)  # Total number of features in latent space

    # Randomly select indices to mask
    mask_indices = random.sample(range(total_features), num_features_to_mask)
    print(f"Masked Indices: {mask_indices}")

    # Print latent vector before masking
    print(f"Original Latent Vector:\n{latent_vector}")

    # Mask the selected features
    latent_vector_masked[:, mask_indices] = 0

    # Print latent vector after masking
    print(f"Masked Latent Vector:\n{latent_vector_masked}")

    return latent_vector_masked

# Main function to test with random masking
def main():
    image_path = "/home/selab/darshan/git-repos/dataset/town7_dataset/test/town7_000692.png"  # Change the image path as needed
    img = load_image(image_path)

    # Load the encoder, decoder, and classifier
    encoder = load_vae_encoder()
    decoder = load_vae_decoder()
    classifier = load_classifier()

    transform = transforms.Compose([
        transforms.Resize((160, 80)),  # Resize to match the model's input
        transforms.ToTensor(),
    ])

    # Get the latent vector from the image
    latent_vector = get_latent_vector(img, encoder, transform)

    # Randomly mask some features in the latent vector
    masked_latent_vector = randomly_mask_latent(latent_vector, num_features_to_mask=10)

    # Pass the masked latent vector to the decoder and reconstruct the image
    with torch.no_grad():
        reconstructed_image = decoder(masked_latent_vector)

    # Display the original and reconstructed images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title("Reconstructed Image (Masked Latent)")
    plt.axis("off")
    plt.show()

    # Print the predictions before and after masking
    original_pred = torch.argmax(classifier(latent_vector), dim=1).item()
    masked_pred = torch.argmax(classifier(masked_latent_vector), dim=1).item()

    print(f"Original Prediction: {'STOP' if original_pred == 0 else 'GO'}")
    print(f"Masked Prediction: {'STOP' if masked_pred == 0 else 'GO'}")


if __name__ == "__main__":
    main()