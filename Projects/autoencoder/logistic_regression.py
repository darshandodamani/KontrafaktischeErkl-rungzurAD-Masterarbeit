import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the trained VAE model
latent_dim_size = 128  # Assuming the latent space size of VAE is 128
vae_model = VariationalAutoencoder(latent_dims=latent_dim_size, num_epochs=50).to(
    device
)
vae_model.load_state_dict(
    torch.load(
        "model/epochs_500_latent_128/var_autoencoder.pth",
        map_location=device,
        weights_only=True,
    )
)
print("VAE model loaded successfully!")
vae_model.eval()

# Load the test dataset (for latent feature extraction)
data_dir = "dataset/town7_dataset/test/"
csv_file = "dataset/town7_dataset/test/labeled_test_data_log.csv"
data_transforms = transforms.Compose([transforms.ToTensor()])

test_dataset = CustomImageDatasetWithLabels(
    img_dir=data_dir, csv_file=csv_file, transform=data_transforms
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Collect latent vectors and true labels
latent_vectors = []
labels = []

with torch.no_grad():
    for images, batch_labels, _ in test_loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        # Encode images into latent space
        _, _, latent_vector_batch = vae_model.encoder(images)

        latent_vectors.append(latent_vector_batch.cpu().numpy())
        labels.append(batch_labels.cpu().numpy())

# Convert lists to numpy arrays
latent_vectors = np.concatenate(latent_vectors, axis=0)
labels = np.concatenate(labels, axis=0)

print(f"Extracted Latent Vectors Shape: {latent_vectors.shape}")
print(f"Labels Shape: {labels.shape}")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    latent_vectors, labels, test_size=0.2, random_state=42
)

# Instantiate and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nLogistic Regression Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print(f"Confusion Matrix:\n{conf_matrix}")

# Optionally, plot the confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["STOP", "GO"],
    yticklabels=["STOP", "GO"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix: Logistic Regression")
plt.show()
