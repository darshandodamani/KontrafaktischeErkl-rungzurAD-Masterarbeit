import torch
import os
import numpy as np
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels  # Import the VAE model and custom dataset class from vae.py
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Path to the saved model and dataset
model_path = 'model/var_autoencoder.pth'
data_dir = 'dataset/town7_dataset/train/'
csv_file = 'dataset/town7_dataset/train/labeled_train_data_log.csv'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the trained VAE model
model = VariationalAutoencoder(latent_dims=256).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transformations (if needed)
data_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Use the CustomImageDatasetWithLabels class to load your dataset
dataset = CustomImageDatasetWithLabels(img_dir=data_dir, csv_file=csv_file, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

latent_vectors = []
labels = []

# Extract latent vectors from the encoder
for images, label in dataloader:
    images = images.to(device)
    with torch.no_grad():
        latent_vector = model.encoder(images)  # Get latent vector from encoder
    latent_vectors.append(latent_vector.cpu().numpy())
    labels.append(label.cpu().numpy())  # Save labels to use in visualization

# Convert latent vectors and labels to numpy arrays
latent_vectors = np.concatenate(latent_vectors)
labels = np.concatenate(labels)

# Visualize latent space using t-SNE (for 2D visualization)
tsne = TSNE(n_components=3, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors)

# Plot the latent space
plt.figure(figsize=(8, 8))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.title('Latent Space Visualization (t-SNE)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


