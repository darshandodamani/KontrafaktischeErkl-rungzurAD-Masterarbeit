# Projects/autoencoder/train_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from vae import (
    VariationalAutoencoder,
    CustomImageDatasetWithLabels,
)  # Import  VAE and dataset class
from classifier import ClassifierModel  # Import the classifier model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 128  # Latent space size from VAE
hidden_size = 128
output_size = 2  # STOP or GO
num_epochs = 20
learning_rate = 0.001

# Load the trained VAE model
vae_model = VariationalAutoencoder(latent_dims=input_size).to(device)
vae_model.load_state_dict(
    torch.load("model/var_autoencoder.pth", map_location=device, weights_only=True)
)
vae_model.eval()  # Set VAE model to evaluation mode

# Instantiate the classifier
classifier = ClassifierModel(
    input_size=input_size, hidden_size=hidden_size, output_size=output_size
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

# Load the training dataset
data_transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = CustomImageDatasetWithLabels(
    img_dir="dataset/town7_dataset/train/",
    csv_file="dataset/town7_dataset/train/labeled_train_data_log.csv",
    transform=data_transforms,
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    classifier.train()  # Set classifier to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Encode images into latent space using VAE
        latent_vectors = vae_model.encoder(images)[2]

        # Forward pass through the classifier
        outputs = classifier(latent_vectors)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Save the trained classifier
torch.save(classifier.state_dict(), "model/classifier.pth")
print("Classifier saved successfully!")
