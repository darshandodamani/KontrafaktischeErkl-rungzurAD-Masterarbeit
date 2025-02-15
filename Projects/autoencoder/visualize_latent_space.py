import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image
import seaborn as sns
import os
import sys

# ------------------------------------------------------------------------------
# Configuration: Paths & Device Setup
# ------------------------------------------------------------------------------
# Define script directory
script_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(script_dir, "Projects/autoencoder"))

# Import necessary models
from encoder import VariationalEncoder
from classifier import ClassifierModel

# Define model paths
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
CLASSIFIER_PATH = "model/epochs_500_latent_128/classifier_4_classes.pth"

# Define dataset paths
TEST_DIR = "dataset/town7_dataset/test/"
TEST_CSV = "dataset/town7_dataset/test/labeled_test_4_class_data_log.csv"

# Define output directory for plots
OUTPUT_DIR = "plots/latent_space_4_classes/"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure directory exists

# Set device for computation (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------------------
# Load Pretrained Models (Encoder & Classifier)
# ------------------------------------------------------------------------------
print("Loading models...")

# Load encoder
encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
encoder.eval()  # Set to evaluation mode

# Load classifier (4-class output)
classifier = ClassifierModel(input_size=128, hidden_size=128, output_size=4).to(device)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device, weights_only=True))
classifier.eval()  # Set to evaluation mode

print("Models loaded successfully.")

# ------------------------------------------------------------------------------
# Define Image Transformations
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ------------------------------------------------------------------------------
# Load Test Data & Extract Latent Representations
# ------------------------------------------------------------------------------
print("Loading test dataset...")

df = pd.read_csv(TEST_CSV)  # Load test image labels
image_files = df["image_filename"].tolist()
labels = df["label"].tolist()  # Labels: STOP, GO, RIGHT, LEFT

# Mapping for visualization
label_mapping = {"STOP": 0, "GO": 1, "RIGHT": 2, "LEFT": 3}
color_mapping = ["red", "blue", "green", "purple"]
legend_labels = ["STOP", "GO", "RIGHT", "LEFT"]

# Convert labels to numeric format
true_labels = [label_mapping[label] for label in labels]

latent_vectors = []
predicted_labels = []

print("Processing test images...")

for img_file, true_label in zip(image_files, true_labels):
    img_path = os.path.join(TEST_DIR, img_file)

    try:
        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Encode image to latent space and classify
        with torch.no_grad():
            _, _, latent_z = encoder(image)  # Encoder outputs mu, logvar, z
            classifier_output = classifier(latent_z)  # Classify latent features
            predicted_class = torch.argmax(classifier_output, dim=1).item()

        # Store latent vectors and labels
        latent_vectors.append(latent_z.cpu().numpy().flatten())
        predicted_labels.append(predicted_class)

    except Exception as e:
        print(f"⚠️ Error processing {img_file}: {e}")

# Convert lists to NumPy arrays
latent_vectors = np.array(latent_vectors)
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

print(f" Extracted {len(latent_vectors)} latent vectors.")

# ------------------------------------------------------------------------------
# Apply PCA (Reduce Latent Space from 128D → 2D)
# ------------------------------------------------------------------------------
print("Applying PCA...")
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_vectors)

# Plot PCA with True Labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=latent_pca[:, 0], y=latent_pca[:, 1], hue=true_labels, palette=color_mapping, alpha=0.7)
plt.title("PCA Projection of Latent Space (True Labels)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(legend_labels)
plt.savefig(f"{OUTPUT_DIR}/pca_latent_space_true_labels.png")
plt.close()

# Plot PCA with Predicted Labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=latent_pca[:, 0], y=latent_pca[:, 1], hue=predicted_labels, palette=color_mapping, alpha=0.7)
plt.title("PCA Projection of Latent Space (Predicted Labels)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(legend_labels)
plt.savefig(f"{OUTPUT_DIR}/pca_latent_space_predicted_labels.png")
plt.close()

print(" PCA plots saved.")

# ------------------------------------------------------------------------------
# Apply t-SNE (Reduce Latent Space to 2D)
# ------------------------------------------------------------------------------
print("Applying t-SNE (this may take a while)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
latent_tsne = tsne.fit_transform(latent_vectors)

# Plot t-SNE with True Labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=latent_tsne[:, 0], y=latent_tsne[:, 1], hue=true_labels, palette=color_mapping, alpha=0.7)
plt.title("t-SNE Projection of Latent Space (True Labels)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(legend_labels)
plt.savefig(f"{OUTPUT_DIR}/tsne_latent_space_true_labels.png")
plt.close()

# Plot t-SNE with Predicted Labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=latent_tsne[:, 0], y=latent_tsne[:, 1], hue=predicted_labels, palette=color_mapping, alpha=0.7)
plt.title("t-SNE Projection of Latent Space (Predicted Labels)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(legend_labels)
plt.savefig(f"{OUTPUT_DIR}/tsne_latent_space_predicted_labels.png")
plt.close()

print(" t-SNE plots saved.")
print(f"All plots saved in: {OUTPUT_DIR}")
