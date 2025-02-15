from sympy import erfi
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Configuration and Hyperparameters
# ------------------------------------------------------------------------------
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 128  # Latent space size from VAE
hidden_size = 128
output_size = 4  # 4 classes: STOP, GO, RIGHT, LEFT
num_epochs = 20
learning_rate = 0.001
dropout_rate = 0.5
batch_size = 64  # Match VAE batch size

# Paths
train_csv = "dataset/town7_dataset/train/labeled_train_4_class_data_log.csv"
test_csv = "dataset/town7_dataset/test/labeled_test_4_class_data_log.csv"
train_img_dir = "dataset/town7_dataset/train/"
test_img_dir = "dataset/town7_dataset/test/"

encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
classifier_save_path = "model/epochs_500_latent_128_town_7/classifier_4_classes.pth"
save_plot_dir = "plots/classifier_plots_for_4_classes/"
os.makedirs(save_plot_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Dataset Loader
# ------------------------------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_mapping = {"STOP": 0, "GO": 1, "RIGHT": 2, "LEFT": 3}
        self.data["label"] = self.data["label"].map(self.label_mapping)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]["image_filename"])
        label = self.data.iloc[idx]["label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------------------------------------------------------------------
# Load Encoder (For Extracting Latent Representations) and ClassifierModel
# ------------------------------------------------------------------------------
from encoder import VariationalEncoder  # Import encoder from autoencoder module
from classifiers_4_class import ClassifierModel  # Import classifier from autoencoder module

encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
encoder.eval()

# ------------------------------------------------------------------------------
# Initialize Model, Loss, and Optimizer
# ------------------------------------------------------------------------------
classifier = ClassifierModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

# ------------------------------------------------------------------------------
# Data Transformations and DataLoaders
# ------------------------------------------------------------------------------
transform = transforms.Compose([transforms.Resize((80, 160)), transforms.ToTensor()])

train_dataset = CustomImageDataset(csv_file=train_csv, img_dir=train_img_dir, transform=transform)
test_dataset = CustomImageDataset(csv_file=test_csv, img_dir=test_img_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------------
print("\n---------------  Starting Training... -----------------\n")
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Extract latent vectors from encoder
        with torch.no_grad():
            _, _, latent_vectors = encoder(images)  # Use only the latent vector

        # Forward pass
        outputs = classifier(latent_vectors)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# ------------------------------------------------------------------------------
# Save Trained Model
# ------------------------------------------------------------------------------
torch.save(classifier.state_dict(), classifier_save_path)
print(f"\n Model saved successfully at {classifier_save_path}!")

# ------------------------------------------------------------------------------
# Plot Training Loss & Accuracy
# ------------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_plot_dir, "training_loss_accuracy_4_classes.png"))
plt.close()

print(f"\n ------ Training plots saved in {save_plot_dir} ------")

# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------
classifier.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Extract latent vectors
        _, _, latent_vectors = encoder(images)

        # Get classifier predictions
        outputs = classifier(latent_vectors)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Generate classification report
classification_report_str = classification_report(
    all_labels, all_preds, target_names=["STOP", "GO", "RIGHT", "LEFT"], output_dict=False
)

with open(os.path.join(save_plot_dir, "classification_report.txt"), "w") as f:
    f.write(classification_report_str)

print("\n ---------- Classification report saved! ----------")

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["STOP", "GO", "RIGHT", "LEFT"])
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(save_plot_dir, "confusion_matrix_4_classes.png"))
plt.close()

print(f"\n -----------  All plots and reports saved in {save_plot_dir} -----------")
