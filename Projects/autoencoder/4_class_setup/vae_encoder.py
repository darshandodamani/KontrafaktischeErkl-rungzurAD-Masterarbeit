# location: Projects/autoencoder/4_class_setup/vae_encoder.py
import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, num_epochs):  # Accept num_epochs for file paths
        super(VariationalEncoder, self).__init__()

        # Save model path dynamically based on epochs and latent dimensions
        self.model_file = os.path.join(
            f"models/epochs_{num_epochs}_latent_{latent_dims}_2/", "vae_encoder.pth"
        )
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

        # Encoder network
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),  # Input: 3 channels (RGB), Output: 32 channels
            nn.LeakyReLU(),
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: 64 channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),  # Output: 128 channels
            nn.LeakyReLU(),
        )
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),  # Output: 256 channels
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        # Flatten output for linear layers
        self.flatten = nn.Flatten()

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 80, 160)  # Replace with actual image size
            dummy_output = self.encoder_layer4(
                self.encoder_layer3(self.encoder_layer2(self.encoder_layer1(dummy_input)))
            )
            flattened_size = dummy_output.view(1, -1).size(1)

        # Fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.LeakyReLU(),
        )

        # Latent space representation (mu and logvar)
        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

        # For reparameterization trick
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

    def forward(self, x):
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.flatten(x)
        x = self.linear(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation
        z = mu + std * self.N.sample(mu.shape)  # Reparameterization trick
        return mu, logvar, z  # Return latent variables

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        print(f"Encoder model saved to {self.model_file}")

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        print(f"Encoder model loaded from {self.model_file}")
