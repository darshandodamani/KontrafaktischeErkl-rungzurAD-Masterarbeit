# location: Projects/autoencoder/4_class_setup/vae_decoder.py
import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalDecoder(nn.Module):
    def __init__(self, latent_dims, num_epochs):  # Accept num_epochs for file paths
        super(VariationalDecoder, self).__init__()

        # Save model path dynamically based on epochs and latent dimensions
        self.model_file = os.path.join(
            f"models/epochs_{num_epochs}_latent_{latent_dims}_2/", "vae_decoder.pth"
        )
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

        # Fully connected layers to expand the latent vector
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),  # Latent vector to 1024
            nn.LeakyReLU(),
            nn.Linear(1024, 9 * 4 * 256),  # To match input size for ConvTranspose layers
            nn.LeakyReLU(),
        )

        # Unflatten the latent vector for transposed convolutions
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 9))

        # Transposed convolutions to upsample back to original image size (3, 80, 160)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1, output_padding=(0, 1)
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1, output_padding=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1, output_padding=(0, 0)
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=4, stride=2, padding=1
            ),  # Final layer
            nn.Sigmoid(),  # To normalize output to [0, 1]
        )

    def forward(self, x):
        x = self.decoder_linear(x)  # Expand latent vector
        x = self.unflatten(x)  # Reshape to match ConvTranspose input
        x = self.decoder(x)  # Reconstruct image
        return x

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        print(f"Decoder model saved to {self.model_file}")

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        print(f"Decoder model loaded from {self.model_file}")
