import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn import neighbors
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ------------------------------------------------------------------------------
# Configuration and Hyperparameters
# ------------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128       # Must match your VAE's latent dimension
batch_size = 64        # Match your VAE training/testing batch size

# Directories for saving models and plots
MODEL_DIR = "model/epochs_500_latent_128_town_7/classifiers/"
PLOTS_DIR = "plots/classifier_plots_4_class/"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Class labels for visualization and mapping
CLASS_NAMES = ["STOP", "GO", "RIGHT", "LEFT"]

# Paths for CSV files and images
train_csv = "dataset/town7_dataset/train/labeled_train_4_class_data_log.csv"
test_csv  = "dataset/town7_dataset/test/labeled_test_4_class_data_log.csv"
train_img_dir = "dataset/town7_dataset/train/"
test_img_dir  = "dataset/town7_dataset/test/"

# Path to the saved VAE encoder
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
# Path where the trained KNN classifier will be saved (as a .pth file)
classifier_save_path = os.path.join(MODEL_DIR, "knn_classifier_4_classes.pth")

# ------------------------------------------------------------------------------
# Dataset Definition
# ------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Map textual labels to integers
        self.label_mapping = {"STOP": 0, "GO": 1, "RIGHT": 2, "LEFT": 3}
        self.data["label"] = self.data["label"].map(self.label_mapping)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Assumes CSV has a column named "image_filename"
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]["image_filename"])
        label = self.data.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------------------------------------------------------------------
# Data Transformations and DataLoaders
# ------------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])

train_dataset = CustomImageDataset(csv_file=train_csv, img_dir=train_img_dir, transform=transform)
test_dataset  = CustomImageDataset(csv_file=test_csv,  img_dir=test_img_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------------------------
# Load VAE Encoder for Latent Representation Extraction
# ------------------------------------------------------------------------------
# Get the absolute path of the 'autoencoder' directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOENCODER_PATH = os.path.join(SCRIPT_DIR, "..")  # Move up one level
sys.path.append(AUTOENCODER_PATH)

# Assumes you have a module "encoder.py" defining VariationalEncoder
from encoder import VariationalEncoder

encoder = VariationalEncoder(latent_dims=latent_dim, num_epochs=100).to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
encoder.eval()
print("VAE Encoder loaded successfully.")

# ------------------------------------------------------------------------------
# Extract Latent Representations
# ------------------------------------------------------------------------------
def extract_latent_vectors(data_loader, encoder, device):
    latent_vectors = []
    labels_list = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            # Assuming the encoder returns a tuple: (mu, logvar, latent)
            _, _, latent = encoder(images)
            latent_vectors.append(latent.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.vstack(latent_vectors), np.hstack(labels_list)

print("Extracting latent representations for training data...")
X_train, y_train = extract_latent_vectors(train_loader, encoder, device)
print("Extracting latent representations for test data...")
X_test, y_test = extract_latent_vectors(test_loader, encoder, device)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ------------------------------------------------------------------------------
# Train KNN Classifier
# ------------------------------------------------------------------------------
knn_classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
# Predictions and classification report
y_pred = knn_classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for KNN Classifier (4 Classes)")
cm_path = os.path.join(PLOTS_DIR, "knn_confusion_matrix.png")
plt.savefig(cm_path)

# ROC Curves (One-vs-Rest)
# Binarize labels for ROC computation
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]
y_score = knn_classifier.predict_proba(X_test)

plt.figure()
colors = ['blue', 'red', 'green', 'orange']
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for KNN Classifier (4 Classes)")
plt.legend(loc="lower right")
roc_path = os.path.join(PLOTS_DIR, "knn_roc_curve.png")
plt.savefig(roc_path)

# ------------------------------------------------------------------------------
# Save the Trained KNN Classifier as a .pth File
# ------------------------------------------------------------------------------
# Note: Although KNN is not a PyTorch module, torch.save() uses pickle under the hood.
torch.save(knn_classifier, classifier_save_path)
print(f"KNN classifier saved at: {classifier_save_path}")
