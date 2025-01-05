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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 128 * 128 * 3  # Input size for flattened images
hidden_size = 128
output_size = 4  # 4 classes: LEFT, GO, STOP, RIGHT
num_epochs = 20
learning_rate = 0.001
batch_size = 64
dropout_rate = 0.5

# Save directory for plots and reports
save_dir = "plots/classifier_plots_for_4_classes/"
os.makedirs(save_dir, exist_ok=True)
# Save directory for models
save_model_dir = "model/epochs_500_latent_128/"
os.makedirs(save_model_dir, exist_ok=True)

# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Map labels to integers
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

# Define the ClassifierModel
class ClassifierModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, output_size=4, dropout_rate=0.5):
        super(ClassifierModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.fc4 = nn.Linear(hidden_size, output_size)  # Output layer for 4 classes

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        x = self.fc4(x)  # Output logits
        return x

# Dataset and DataLoader
train_transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

train_dataset = CustomImageDataset(
    csv_file="dataset/town7_dataset/train/labeled_train_4_class_data_log.csv",
    img_dir="dataset/town7_dataset/train/",
    transform=train_transforms,
)
test_dataset = CustomImageDataset(
    csv_file="dataset/town7_dataset/test/labeled_test_4_class_data_log.csv",
    img_dir="dataset/town7_dataset/test/",
    transform=test_transforms,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss, and optimizer
model = ClassifierModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, dropout_rate=dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.view(images.size(0), -1).to(device)  # Flatten input
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), os.path.join(save_model_dir, "classifier_4_classes.pth"))
print("Model saved successfully!")

# Plot training loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "training_loss_accuracy_4_classes.png"))
plt.close()

# Evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Generate and save classification report
classification_report_str = classification_report(
    all_labels, all_preds, target_names=["STOP", "GO", "RIGHT", "LEFT"]
)
with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
    f.write(classification_report_str)
print("Classification report saved!")

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["STOP", "GO", "RIGHT", "LEFT"])
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(save_dir, "confusion_matrix_4_classes.png"))
plt.close()

print(f"All plots and reports saved in {save_dir}")
