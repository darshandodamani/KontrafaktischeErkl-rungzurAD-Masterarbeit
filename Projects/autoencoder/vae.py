import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from encoder import VariationalEncoder
from decoder import Decoder
import matplotlib.pyplot as plt
import csv
from PIL import Image

# Hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
LATENT_SPACE = 128  # Latent dimension size
KL_WEIGHT_INITIAL = 0.001  # Initial KL weight (for annealing)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define the VAE model
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.model_file = os.path.join("model/100_epochs_128_LF/", "var_autoencoder.pth")
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        mu, logvar, z = self.encoder(x.to(device))  # Get mu, logvar, z from encoder
        return (
            self.decoder(z),
            mu,
            logvar,
            z,
        )  # Return decoded image and latent variables

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()


# Define a custom dataset class
class CustomImageDatasetWithLabels(torch.utils.data.Dataset):
    """Custom dataset loader that reads labels from a CSV file."""

    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        # Read CSV and store image paths and labels
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                img_path = os.path.join(img_dir, row["image_filename"])
                label = 0 if row["label"] == "STOP" else 1
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# KL divergence function
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


train_losses = []
val_losses = []
train_mse_losses = []
train_kl_losses = []
val_mse_losses = []
val_kl_losses = []


def train(model, trainloader, optimizer, epoch, kl_weight):
    model.train()
    train_loss = 0.0
    mse_loss_total = 0.0
    kl_loss_total = 0.0
    for batch_idx, (x, _) in enumerate(trainloader):
        x = x.to(device)
        x_hat, mu, logvar, z = model(x)  # Forward pass through the VAE

        # Reconstruction loss (MSE) and KL Divergence
        mse_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kl_loss = kl_divergence(mu, logvar) * kl_weight
        total_loss = mse_loss + kl_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()
        mse_loss_total += mse_loss.item()
        kl_loss_total += kl_loss.item()

        # Log z and losses
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Latent vector z: {z}")
            print(
                f"\n MSE Loss: {mse_loss.item()}, KL Loss: {kl_loss.item()}, Total Loss: {total_loss.item()}"
            )

        # Save a few reconstructions at the beginning of each epoch
        if batch_idx == 0:
            save_reconstructions(model, trainloader, epoch)

    avg_train_loss = train_loss / len(trainloader.dataset)
    avg_mse_loss = mse_loss_total / len(trainloader.dataset)
    avg_kl_loss = kl_loss_total / len(trainloader.dataset)

    return avg_train_loss, avg_mse_loss, avg_kl_loss


# Save reconstructed images for visualization
def save_reconstructions(model, trainloader, epoch):
    """Save reconstructed images for visual inspection."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(trainloader):
            x = x.to(device)
            x_hat, _, _, _ = model(x)  # Forward pass through the model

            # Plot original and reconstructed images side by side
            plt.figure(figsize=(8, 4))
            for i in range(min(4, x.size(0))):  # Use `x.size(0)` to get the batch size
                original = x[i].cpu().numpy().transpose(1, 2, 0)
                reconstructed = x_hat[i].cpu().detach().numpy().transpose(1, 2, 0)

                plt.subplot(2, 4, i + 1)
                plt.imshow(original)
                plt.title("Original")
                plt.axis("off")

                plt.subplot(2, 4, i + 5)
                plt.imshow(reconstructed)
                plt.title("Reconstructed")
                plt.axis("off")

            if not os.path.exists("dataset/reconstructed_image_100_epochs_128_LF/"):
                os.makedirs("dataset/reconstructed_image_100_epochs_128_LF")

            plt.savefig(
                f"dataset/reconstructed_image_100_epochs_128_LF/reconstructed_epoch_{epoch}.png"
            )
            plt.close()
            break  # Save reconstructions for the first batch only


# Validation (testing) function
def test(model, testloader, kl_weight):
    model.eval()
    val_loss = 0.0
    mse_loss_total = 0.0
    kl_loss_total = 0.0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            x_hat, mu, logvar, z = model(x)
            mse_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
            kl_loss = kl_divergence(mu, logvar) * kl_weight
            total_loss = mse_loss + kl_loss

            val_loss += total_loss.item()
            mse_loss_total += mse_loss.item()
            kl_loss_total += kl_loss.item()

    avg_val_loss = val_loss / len(testloader.dataset)
    avg_mse_loss = mse_loss_total / len(testloader.dataset)
    avg_kl_loss = kl_loss_total / len(testloader.dataset)

    return avg_val_loss, avg_mse_loss, avg_kl_loss


# Main function to run the VAE
def main():
    data_dir = "dataset/town7_dataset/"
    writer = SummaryWriter(f"runs/" + "auto-encoder")

    # Image transformations for training and testing
    train_transforms = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = CustomImageDatasetWithLabels(
        data_dir + "train",
        data_dir + "train/labeled_train_data_log.csv",
        transform=train_transforms,
    )
    test_data = CustomImageDatasetWithLabels(
        data_dir + "test",
        data_dir + "test/labeled_test_data_log.csv",
        transform=test_transforms,
    )

    # Splitting training data into training and validation sets
    train_len = int(len(train_data) * 0.8)
    val_len = len(train_data) - train_len
    train_data, val_data = random_split(train_data, [train_len, val_len])

    # Data loaders
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Model, optimizer
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Selected device: {device}")

    kl_weight = KL_WEIGHT_INITIAL  # Start with a small KL weight

    for epoch in range(NUM_EPOCHS):
        # Increase KL weight slowly over time (annealing)
        kl_weight = min(KL_WEIGHT_INITIAL + epoch * 0.001, 1.0)

        # Train model
        train_loss, train_mse_loss, train_kl_loss = train(
            model, trainloader, optimizer, epoch, kl_weight
        )

        # Validate model
        val_loss, val_mse_loss, val_kl_loss = test(model, validloader, kl_weight)

        # Append the losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mse_losses.append(train_mse_loss)
        val_mse_losses.append(val_mse_loss)
        train_kl_losses.append(train_kl_loss)
        val_kl_losses.append(val_kl_loss)

        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)

        # Print progress
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        print(
            f"  Train MSE Loss: {train_mse_loss:.3f} | Train KL Loss: {train_kl_loss:.3f}"
        )
        print(f"  Val MSE Loss: {val_mse_loss:.3f} | Val KL Loss: {val_kl_loss:.3f}")

        # Save reconstructions for visual inspection
        if epoch % 10 == 0:
            save_reconstructions(model, trainloader, epoch)

    # Plot losses after training
    plot_losses(
        train_losses,
        val_losses,
        train_mse_losses,
        val_mse_losses,
        train_kl_losses,
        val_kl_losses,
    )

    model.save()


def plot_losses(
    train_losses,
    val_losses,
    train_mse_losses,
    val_mse_losses,
    train_kl_losses,
    val_kl_losses,
):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    # Plot total loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Total Loss")
    plt.plot(epochs, val_losses, label="Val Total Loss")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot MSE loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_mse_losses, label="Train MSE Loss")
    plt.plot(epochs, val_mse_losses, label="Val MSE Loss")
    plt.title("MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot KL loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_kl_losses, label="Train KL Loss")
    plt.plot(epochs, val_kl_losses, label="Val KL Loss")
    plt.title("KL Divergence Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/vae_loss_plot_100_epochs_128_LF.png")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")
