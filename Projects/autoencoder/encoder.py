import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, num_epochs):  # Accept num_epochs
        super(VariationalEncoder, self).__init__()

        # Ensure both num_epochs and latent_dims are converted to strings in the file path
        self.model_file = os.path.join(
            f"model/epochs_{str(num_epochs)}_latent_{str(latent_dims)}/",
            "var_encoder_model.pth",
        )
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

        # Encoder network
        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(
                3, 32, 4, stride=2
            ),  # Input: 3 channels (RGB), Output: 32 channels
            nn.LeakyReLU(),
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(
                32, 64, 3, stride=2, padding=1
            ),  # Input: 32 channels, Output: 64 channels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),  # Input: 64 channels, Output: 128 channels
            nn.LeakyReLU(),
        )
        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(
                128, 256, 3, stride=2
            ),  # Input: 128 channels, Output: 256 channels
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

        # Flatten the output for the linear layers
        self.flatten = nn.Flatten()

        # Determine the size of the flattened output
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 80, 160)
            dummy_output = self.encoder_layer4(
                self.encoder_layer3(
                    self.encoder_layer2(self.encoder_layer1(dummy_input))
                )
            )
            flattened_size = dummy_output.view(1, -1).size(1)

        # Fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.LeakyReLU(),
        )

        # Latent space vectors mu and logvar
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
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation
        z = mu + std * self.N.sample(mu.shape)  # Reparameterization trick

        return mu, logvar, z  # Return mu, logvar, and z

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        print(f"Encoder model saved to {self.model_file}")

    def load(self):
        self.load_state_dict(torch.load(self.model_file, map_location=device))
        print(f"Encoder model loaded from {self.model_file}")
