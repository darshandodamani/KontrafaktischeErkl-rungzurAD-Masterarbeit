import torch
import os
import numpy as np  # Don't forget to import numpy
from vae import (
    VariationalAutoencoder,
    CustomImageDatasetWithLabels,
)  # Import the VAE model and custom dataset class from vae.py
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import (
    Axes3D,
)  # Import 3D plotting capabilities if you want 3D

# Path to the saved model and dataset
model_path = "model/var_autoencoder.pth"
data_dir = "dataset/town7_dataset/train/"
csv_file = "dataset/town7_dataset/train/labeled_train_data_log.csv"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the trained VAE model
model = VariationalAutoencoder(latent_dims=256).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define transformations (if needed)
data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Use the CustomImageDatasetWithLabels class to load your dataset
dataset = CustomImageDatasetWithLabels(
    img_dir=data_dir, csv_file=csv_file, transform=data_transforms
)
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

# Visualize latent space using PCA (for 2D visualization)
pca = PCA(n_components=2)
latent_2d_pca = pca.fit_transform(latent_vectors)

# Plot the latent space
plt.figure(figsize=(8, 8))
plt.scatter(latent_2d_pca[:, 0], latent_2d_pca[:, 1], c=labels, cmap="viridis")
plt.colorbar()
plt.title("Latent Space Visualization (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# If you want 3D PCA visualization:
pca_3d = PCA(n_components=3)
latent_3d_pca = pca_3d.fit_transform(latent_vectors)

# Plot the latent space in 3D using PCA
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(
    latent_3d_pca[:, 0],
    latent_3d_pca[:, 1],
    latent_3d_pca[:, 2],
    c=labels,
    cmap="viridis",
    marker="o",
)
plt.colorbar(sc)
ax.set_title("Latent Space Visualization (3D PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.show()
