import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from encoder import VariationalEncoder
from decoder import Decoder
import matplotlib.pyplot as plt
import csv
from PIL import Image  # Import PIL for image processing

# Hyper-parameters
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LATENT_SPACE = 256  # Latent space size
KL_WEIGHT = 0.05  # Weight for the KL divergence term

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.model_file = os.path.join("model", "var_autoencoder.pth")
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x.to(device))
        return self.decoder(z)

    def save(self):
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()

    def load(self):
        self.load_state_dict(torch.load(self.model_file))
        self.encoder.load()
        self.decoder.load()


# Define a custom dataset that also reads labels from the CSV file
class CustomImageDatasetWithLabels(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        # Read the CSV file and store the image paths and labels
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                img_path = os.path.join(img_dir, row["image_filename"])
                label = (
                    0 if row["label"] == "STOP" else 1
                )  # Convert labels to binary (0 for STOP, 1 for GO)
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def train(model, trainloader, optimizer, epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (x, _) in enumerate(trainloader):
        x = x.to(device)
        z = model.encoder(x)  # Get latent vector
        x_hat = model.decoder(z)  # Reconstruct image from latent vector

        # Custom loss: MSE + KL-Divergence
        mse_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
        kl_loss = KL_WEIGHT * model.encoder.kl
        loss = mse_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Print z and losses every 10 batches or at the start of each epoch
        if batch_idx % 10 == 0 or batch_idx == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Latent vector z: {z}")
            print(
                f"MSE Loss: {mse_loss.item()}, KL Loss: {kl_loss.item()}, Total Loss: {loss.item()}"
            )

        # Save and visualize a few reconstructed images
        if batch_idx == 0:  # Just do this for the first batch of each epoch
            plt.figure(figsize=(8, 4))
            for i in range(
                min(4, x.size(0))
            ):  # Visualize the first 4 images in the batch
                original = x[i].cpu().numpy().transpose(1, 2, 0)
                reconstructed = x_hat[i].cpu().detach().numpy().transpose(1, 2, 0)

                # Plot the original and reconstructed images
                plt.subplot(2, 4, i + 1)
                plt.imshow(original)
                plt.title("Original")
                plt.axis("off")

                plt.subplot(2, 4, i + 5)
                plt.imshow(reconstructed)
                plt.title("Reconstructed")
                plt.axis("off")

            plt.savefig(f"reconstructed_epoch_{epoch}.png")
            # plt.show()

    return train_loss / len(trainloader.dataset)


def test(model, testloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            x_hat = model(x)
            loss = ((x - x_hat) ** 2).sum() + model.encoder.kl
            val_loss += loss.item()

    return val_loss / len(testloader.dataset)


def main():
    data_dir = "dataset/town7_dataset/"

    writer = SummaryWriter(f"runs/" + "auto-encoder")

    # Applying Transformation
    train_transforms = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose([transforms.ToTensor()])

    # Load the labeled dataset
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

    # Split train_data into training and validation datasets
    m = len(train_data)
    train_len = int(m * 0.8)
    val_len = m - train_len  # Ensures the sum equals the length of the dataset
    train_data, val_data = random_split(train_data, [train_len, val_len])

    # Data Loading
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = VariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Selected device: {device}")

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, trainloader, optimizer, epoch)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        val_loss = test(model, validloader)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)
        print(
            f"\nEPOCH {epoch+1}/{NUM_EPOCHS} \t train loss: {train_loss:.3f} \t val loss: {val_loss:.3f}"
        )

    model.save()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")
