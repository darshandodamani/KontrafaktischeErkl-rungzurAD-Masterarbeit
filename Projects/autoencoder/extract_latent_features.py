import torch
import numpy as np
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from encoder import VariationalEncoder

# ---------------------- Configuration ----------------------
# Paths to trained encoder and dataset
ENCODER_PATH = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
TEST_CSV = "dataset/town7_dataset/test/labeled_test_4_class_data_log.csv"
TEST_DIR = "dataset/town7_dataset/test/"
OUTPUT_FILE = "projects/autoencoder/classifiers/latent_features_4_class.npy"

# Define class mapping
CLASS_MAPPING = {"STOP": 0, "GO": 1, "RIGHT": 2, "LEFT": 3}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained encoder
encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location=device))
encoder.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

# Load dataset labels
df = pd.read_csv(TEST_CSV)
image_files = df["image_filename"].tolist()
labels = df["label"].map(CLASS_MAPPING).tolist()

# Extract latent vectors
latent_vectors = []
true_labels = []

for img_file, label in zip(image_files, labels):
    img_path = os.path.join(TEST_DIR, img_file)
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, latent_z = encoder(image)  # Extract latent vector

    latent_vectors.append(latent_z.cpu().numpy().flatten())
    true_labels.append(label)

# Save latent vectors and labels
np.savez(OUTPUT_FILE, features=np.array(latent_vectors), labels=np.array(true_labels))
print(f"✅ Latent features saved to {OUTPUT_FILE}")
