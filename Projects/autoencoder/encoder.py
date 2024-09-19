import os
import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()

        self.model_file = os.path.join(
            "model/100_epochs_95_LF/", "var_encoder_model.pth"
        )
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

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

        # Input size is (3, 80, 160)
        self.flatten = nn.Flatten()

        # Pass a dummy input to determine the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 80, 160)
            dummy_output = self.encoder_layer4(
                self.encoder_layer3(
                    self.encoder_layer2(self.encoder_layer1(dummy_input))
                )
            )
            flattened_size = dummy_output.view(1, -1).size(1)

        self.linear = nn.Sequential(
            nn.Linear(
                flattened_size, 1024
            ),  # Input size depends on flattened dimension
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(1024, latent_dims)
        self.logvar = nn.Linear(1024, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(device)
        self.N.scale = self.N.scale.to(device)

    def forward(self, x):
        x = x.to(device)
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
        return mu, logvar, z  # Return mu, logvar, and z for further calculations

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        print(f"Model saved to {self.model_file}")

    def load(self):
        try:
            self.load_state_dict(torch.load(self.model_file, map_location=device))
            print(f"Model loaded from {self.model_file}")
        except Exception as e:
            print(f"Error loading model from {self.model_file}: {e}")
