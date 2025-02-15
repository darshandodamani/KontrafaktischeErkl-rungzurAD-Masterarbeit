import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image
import os
import sys

# ---------------------- Configuration ----------------------
# Define paths
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(SCRIPT_DIR, 'Projects/autoencoder'))

# Import Encoder & Classifier
from encoder import VariationalEncoder
from classifier import ClassifierModel

# Paths to trained models
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128_town_7/classifier_4_classes.pth"

# Define class labels and colors
CLASS_NAMES = ["STOP", "GO", "RIGHT", "LEFT"]
CLASS_COLORS = ["red", "green", "yellow", "purple"]  # Explicit color mapping

# Output directory for saving plots
OUTPUT_DIR = "plots/latent_space_4_classes/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- Load Models ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained encoder
encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
encoder.eval()

# Load trained classifier for 4 classes
classifier = ClassifierModel(input_size=128, hidden_size=128, output_size=4).to(device)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))
classifier.eval()

# ---------------------- Dataset Processing ----------------------
# Define transformations
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

# Path to test dataset and labels
TEST_DIR = "dataset/town7_dataset/test/"
TEST_CSV = "dataset/town7_dataset/test/labeled_test_4_class_data_log.csv"

# Load dataset labels
df = pd.read_csv(TEST_CSV)
image_files = df["image_filename"].tolist()
labels = df["label"].map({"STOP": 0, "GO": 1, "RIGHT": 2, "LEFT": 3}).tolist()

# ---------------------- Extract Latent Features ----------------------
latent_vectors = []
true_labels = []
predicted_labels = []

# Process each test image
for img_file, label in zip(image_files, labels):
    img_path = os.path.join(TEST_DIR, img_file)
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Encode image into latent space
    with torch.no_grad():
        _, _, latent_z = encoder(image)  # Encoder outputs mu, logvar, z
        classifier_output = classifier(latent_z)  # Classifier prediction
        predicted_class = torch.argmax(classifier_output, dim=1).item()

    latent_vectors.append(latent_z.cpu().numpy().flatten())
    true_labels.append(label)
    predicted_labels.append(predicted_class)

# Convert to NumPy arrays
latent_vectors = np.array(latent_vectors)
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# ---------------------- PCA Visualization ----------------------
print("Performing PCA on Latent Features...")
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_vectors)

# Function to plot PCA and t-SNE results with explicit color mapping
def plot_latent_projection(latent_2D, labels, title, filename):
    plt.figure(figsize=(8, 6))
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_idx
        plt.scatter(latent_2D[mask, 0], latent_2D[mask, 1], label=class_name, 
                    color=CLASS_COLORS[class_idx], alpha=0.7, edgecolors='black', linewidth=0.5)

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# Plot PCA (True Labels)
plot_latent_projection(latent_pca, true_labels, 
                       "PCA Projection of Latent Space (True Labels)", 
                       "pca_latent_space_true_labels.png")

# Plot PCA (Predicted Labels)
plot_latent_projection(latent_pca, predicted_labels, 
                       "PCA Projection of Latent Space (Predicted Labels)", 
                       "pca_latent_space_predicted_labels.png")

# ---------------------- t-SNE Visualization ----------------------
print("Performing t-SNE on Latent Features...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latent_tsne = tsne.fit_transform(latent_vectors)

# Plot t-SNE (True Labels)
plot_latent_projection(latent_tsne, true_labels, 
                       "t-SNE Projection of Latent Space (True Labels)", 
                       "tsne_latent_space_true_labels.png")

# Plot t-SNE (Predicted Labels)
plot_latent_projection(latent_tsne, predicted_labels, 
                       "t-SNE Projection of Latent Space (Predicted Labels)", 
                       "tsne_latent_space_predicted_labels.png")

print(f"--------- All visualizations saved in {OUTPUT_DIR} ---------")
