import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from vae import (
    VariationalAutoencoder,
    CustomImageDatasetWithLabels,
)  # Import the VAE and dataset class
from classifier import ClassifierModel  # Import the improved classifier model
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 128  # Latent space size from VAE (ensure it matches the latent space size of the trained VAE)
hidden_size = 128
output_size = 4  # STOP, GO, RIGHT, LEFT
num_epochs = 20
learning_rate = 0.001
dropout_rate = 0.5  # Dropout rate for the classifier
batch_size = 128  # Match batch size from VAE training

# Paths
model_save_path = "model/epochs_500_latent_128/"
plot_save_path = "plots/classifier_plots/"
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(plot_save_path, exist_ok=True)

# Load the trained VAE model
vae_model = VariationalAutoencoder(latent_dims=input_size, num_epochs=500).to(device)
vae_model.load_state_dict(
    torch.load(
        "model/epochs_500_latent_128/var_autoencoder.pth",
        map_location=device,
        weights_only=True,
    )
)
vae_model.eval()  # Set VAE model to evaluation mode (important for inference)

# Instantiate the classifier
classifier = ClassifierModel(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    dropout_rate=dropout_rate,
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

# Load the training dataset
data_transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = CustomImageDatasetWithLabels(
    img_dir="dataset/town7_dataset/train/",
    csv_file="dataset/town7_dataset/train/labeled_train_4_class_data_log.csv",
    transform=data_transforms,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# To store losses and accuracy for plotting
train_losses = []
train_accuracies = []

# Training loop
for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in train_loader:  # Ensure train_loader returns images, labels, and paths
        images, labels = images.to(device), labels.to(device)

        # Encode images into latent space using VAE (detach to prevent gradients from flowing into VAE)
        with torch.no_grad():
            latent_vectors = vae_model.encoder(images)[2]  # [2] is the latent z vector

        # Forward pass through the classifier
        outputs = classifier(latent_vectors)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    # Store loss and accuracy for plotting
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Print epoch loss and accuracy
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%"
    )

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(model_save_path, f"classifier_epoch_{epoch+1}_4_classes.pth")
        torch.save(classifier.state_dict(), checkpoint_path)
        print(f"Classifier checkpoint saved at {checkpoint_path}")

# Save the final trained classifier
final_model_path = os.path.join(model_save_path, "classifier_final_4_classes.pth")
torch.save(classifier.state_dict(), final_model_path)
print(f"Classifier saved successfully at {final_model_path}!")

# Plot the training loss and accuracy
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.legend()

plt.tight_layout()
plot_path = os.path.join(plot_save_path, f"classifier_training_loss_accuracy_for_{num_epochs}_epochs_{input_size}_4_classes.png")
plt.savefig(plot_path)
print(f"Training plots saved to {plot_path}")