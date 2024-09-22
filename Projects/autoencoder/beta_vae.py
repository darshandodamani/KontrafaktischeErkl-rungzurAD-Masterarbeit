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
from PIL import Image
import matplotlib.pyplot as plt
import csv

# Hyper-parameters
NUM_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-5  # Reduced learning rate
LATENT_SPACE = 128  # Latent space size
BETA = 1.0  # Start with a smaller Beta weight for KL divergence
KL_ANNEAL_RATE = 1.05  # Gradual KL weight increase

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BetaVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(BetaVariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z, mu, sigma, kl_loss = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, mu, sigma, kl_loss


class CustomImageDatasetWithLabels(torch.utils.data.Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

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


def train(model, trainloader, optimizer, epoch, beta):
    model.train()
    train_loss = 0.0
    for batch_idx, (x, _) in enumerate(trainloader):
        x = x.to(device)

        # Forward pass through VAE
        x_hat, mu, sigma, kl_loss = model(x)

        # MSE loss (Reconstruction loss)
        mse_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")

        # KL loss with Beta weighting
        kl_loss = beta * kl_loss.sum()

        # Total loss
        loss = mse_loss + kl_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0 or batch_idx == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Latent vector z: {mu}")
            print(
                f"MSE Loss: {mse_loss.item()}, KL Loss: {kl_loss.item()}, Total Loss: {loss.item()}"
            )

        if batch_idx == 0:  # Visualize only the first batch in each epoch
            plt.figure(figsize=(8, 4))
            for i in range(min(4, x.size(0))):
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

            plt.savefig(f"reconstructed_epoch_{epoch}.png")

    return train_loss / len(trainloader.dataset)


def test(model, testloader, beta):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            x_hat, mu, sigma, kl_loss = model(x)
            mse_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
            kl_loss = beta * kl_loss.sum()
            loss = mse_loss + kl_loss
            val_loss += loss.item()

    return val_loss / len(testloader.dataset)


def main():
    data_dir = "dataset/town7_dataset/"

    writer = SummaryWriter(f"runs/" + "beta-vae")

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
    train_len = int(0.8 * len(train_data))
    val_len = len(train_data) - train_len
    train_data, val_data = random_split(train_data, [train_len, val_len])

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = BetaVariationalAutoencoder(latent_dims=LATENT_SPACE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Selected device: {device}")

    beta = BETA
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, trainloader, optimizer, epoch, beta)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)

        val_loss = test(model, validloader, beta)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)

        print(
            f"EPOCH {epoch+1}/{NUM_EPOCHS} \t train loss: {train_loss:.3f} \t val loss: {val_loss:.3f}"
        )

        # Gradually increase Beta for better disentanglement
        beta *= KL_ANNEAL_RATE

    model.encoder.save()
    model.decoder.save()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("\nTerminating...")
