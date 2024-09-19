import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.model_file = os.path.join("model/100_epochs_95_LF/", "decoder_model.pth")
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

        # Fully connected layers to expand the latent vector
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),  # Latent to 1024
            nn.LeakyReLU(),
            nn.Linear(
                1024, 9 * 4 * 256
            ),  # To match the size before transpose convolutions
            nn.LeakyReLU(),
        )

        # Unflatten the latent vector to prepare for ConvTranspose layers
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 9))

        # Transposed convolutions to upsample back to the original image size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),  # (4x9) -> (9x19)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),  # (9x19) -> (19x39)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1
            ),  # (19x39) -> (40x80)
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2),  # (40x80) -> (80x160)
            nn.Sigmoid(),  # Sigmoid to output pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.decoder_linear(x)  # Expand latent vector
        x = self.unflatten(x)  # Reshape into the feature map for ConvTranspose layers
        x = self.decoder(x)  # Upsample to reconstruct image
        return x

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        print(f"Decoder model saved to {self.model_file}")

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        print(f"Decoder model loaded from {self.model_file}")
