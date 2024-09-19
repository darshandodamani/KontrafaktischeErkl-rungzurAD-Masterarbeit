# Projects/masking/lime/lime_explainer.py
import sys
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

#  the Python path to include the directory where 'encoder.py' and 'decoder.py' are located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

# Import the VariationalEncoder and Decoder from the autoencoder module
from encoder import VariationalEncoder
from decoder import Decoder

# Set the device for models and tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load the VAE encoder and decoder
def load_vae():
    encoder = VariationalEncoder(latent_dims=256).to(
        device
    )  # Move the encoder to the correct device
    decoder = Decoder(latent_dims=256).to(
        device
    )  # Move the decoder to the correct device

    # Load the pre-trained models
    encoder.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/var_encoder_model.pth",
            map_location=device,
        )
    )
    decoder.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/decoder_model.pth", map_location=device
        )
    )

    encoder.eval()
    decoder.eval()

    return encoder, decoder


# Function to get latent vector from the encoder
def get_latent_vector(image, encoder, transform):
    """Transforms the image and encodes it into latent space."""
    image_tensor = (
        transform(image).unsqueeze(0).to(device)
    )  # Convert image to tensor and move to device
    mu, logvar, z = encoder(image_tensor)  # Get the latent vector (mu, logvar, z)

    # Print the original latent vector
    print("Original Latent Vector (z):")
    print(z)
    return z


# Function to apply a mask to the latent space
def apply_latent_space_mask(latent_vector, mask_indices):
    """Apply a mask to the latent vector by zeroing out important features."""
    masked_latent_vector = latent_vector.clone()  # Copy the latent vector

    # Zero-out the specified latent features
    masked_latent_vector[:, mask_indices] = 0

    # Print the masked latent vector
    print(f"Masked Latent Vector (with features at indices {mask_indices} zeroed out):")
    print(masked_latent_vector)

    return masked_latent_vector


# Function to reconstruct image from latent vector
def decode_latent_vector(latent_vector, decoder):
    """Reconstruct the image from the latent vector using the decoder."""
    with torch.no_grad():
        reconstructed_image = decoder(latent_vector).cpu().detach()
    return reconstructed_image


# Main function to perform latent space masking
def main():
    # Load the pre-trained VAE encoder and decoder
    encoder, decoder = load_vae()

    # Transform for image preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((160, 80)),  # Resize to match model input
            transforms.ToTensor(),
        ]
    )

    # Step 1: Get the latent vector from the encoder
    from PIL import Image as PilImage

    image_path = (
        "/home/selab/darshan/git-repos/dataset/town7_dataset/train/town7_000275.png"
    )
    img = PilImage.open(image_path).convert("RGB")
    latent_vector = get_latent_vector(img, encoder, transform)

    # Step 2: Apply the mask to the latent vector
    mask_indices = [
        0,
        1,
        2,
        3,
        4,
    ]  # Zero-out the first 5 dimensions of the latent space
    masked_latent_vector = apply_latent_space_mask(latent_vector, mask_indices)

    # Step 3: Decode the masked latent vector
    reconstructed_image = decode_latent_vector(masked_latent_vector, decoder)

    # Step 4: Visualize the masked latent space reconstruction
    reconstructed_image_np = reconstructed_image.squeeze().numpy().transpose(1, 2, 0)
    plt.imshow(reconstructed_image_np)
    plt.title("Reconstructed Image with Masked Latent Space")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
