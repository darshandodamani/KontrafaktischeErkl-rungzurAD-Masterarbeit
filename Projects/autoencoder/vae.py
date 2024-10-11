import os
import sched
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
import torch.nn.functional as F
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import base64
from io import BytesIO
import sys

# Hyper-parameters
NUM_EPOCHS = 500
BATCH_SIZE = 256  # Increase to stabilize training
LEARNING_RATE = 1e-4  # Keep the learning rate, but consider reducing it mid-training
LATENT_SPACE = 128  # Reduce latent space for compression
KL_WEIGHT_INITIAL = 0.00005  # Initial KL weight

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define the VAE model
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, num_epochs):  # Add num_epochs as a parameter
        super(VariationalAutoencoder, self).__init__()
        # Pass num_epochs to the encoder and decoder
        self.encoder = VariationalEncoder(latent_dims, num_epochs)
        self.decoder = Decoder(latent_dims, num_epochs)

        # Ensure both num_epochs and latent_dims are converted to strings in the file path
        self.model_file = os.path.join(
            f"model/epochs_{str(num_epochs)}_latent_{str(latent_dims)}/",
            "var_autoencoder.pth",
        )
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

    def forward(self, x):
        print(f"Running forward pass on the input of shape: {x.shape}")
        mu, logvar, z = self.encoder(x.to(device))  # Get mu, logvar, z from encoder
        return (
            self.decoder(z),
            mu,
            logvar,
            z,
        )  # Return decoded image and latent variables

    def save(self):
        print(f"Saving model to {self.model_file}")
        torch.save(self.state_dict(), self.model_file)
        self.encoder.save()
        self.decoder.save()

    def load(self):
        print(f"Loading model from {self.model_file}")
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
        return image, label, img_path  # Return img_path along with image and label


# KL divergence function
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Tracking training statistics
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
    all_z_values = []
    all_image_paths = []  # Store image paths corresponding to each z value
    for batch_idx, (x, _, image_paths) in enumerate(trainloader):
        x = x.to(device)
        x_hat, mu, logvar, z = model(x)  # Forward pass through the VAE

        # Store z values for this batch
        all_z_values.append(z.cpu().detach())  # Store z values
        all_image_paths.extend(image_paths)  # Store corresponding image paths

        # Resize the output (x_hat) to match the input size (x) before computing the loss
        x_hat = F.interpolate(
            x_hat, size=x.shape[2:], mode="bilinear", align_corners=False
        )

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

        # Debugging: Log latent vector and losses every 10 batches
        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Latent vector z: {z}")
            print(
                f"MSE Loss: {mse_loss.item()}, KL Loss: {kl_loss.item()}, Total Loss: {total_loss.item()}"
            )

        # Save reconstructions at the start of the epoch
        if batch_idx == 0:
            save_reconstructions(model, trainloader, epoch)

    # Save all z values to CSV after the epoch
    z_values_tensor = torch.cat(all_z_values, dim=0)  # Concatenate all batch z values
    save_z_to_csv(z_values_tensor, all_image_paths, epoch)

    avg_train_loss = train_loss / len(trainloader.dataset)
    avg_mse_loss = mse_loss_total / len(trainloader.dataset)
    avg_kl_loss = kl_loss_total / len(trainloader.dataset)

    return avg_train_loss, avg_mse_loss, avg_kl_loss, z_values_tensor, all_image_paths


# Function to save z-values and image paths to CSV file for all epochs
def save_z_to_csv(z_values, image_paths, epoch):
    """Saves all latent vectors (z) and corresponding image paths to a single CSV file."""
    filename = f"latent_z_values_epoch_{epoch}.csv"
    try:
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Image Path"] + [f"z{i}" for i in range(z_values.shape[1])]
            )  # Header
            for i, z in enumerate(z_values.cpu().detach().numpy()):
                writer.writerow([image_paths[i]] + z.tolist())
        print(f"Saved latent vectors to {filename}")
    except Exception as e:
        print(f"Error saving latent vectors to {filename}: {e}")


# Save reconstructed images for visualization
def save_reconstructions(model, trainloader, epoch):
    """Save reconstructed images for visual inspection."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch_idx, data in enumerate(trainloader):
            if len(data) == 2:
                x, _ = data
            elif len(data) == 3:
                x, _, _ = data  # Adjust this based on what trainloader returns
            else:
                raise ValueError("Unexpected number of values returned by trainloader")

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

            # Save reconstructions dynamically based on epoch
            save_dir = (
                f"dataset/reconstructed_image_{NUM_EPOCHS}_epochs_{LATENT_SPACE}_LF/"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plt.savefig(f"{save_dir}/reconstructed_epoch_{epoch}.png")
            plt.close()
            break  # Save reconstructions for the first batch only


# def visualize_latent_space(z_values, image_paths):
#     """Performs PCA on z values and creates a scatter plot with image paths."""
#     # Perform PCA to reduce to 2D for visualization
#     pca = PCA(n_components=2)
#     z_pca = pca.fit_transform(z_values.cpu().detach().numpy())

#     # Create a DataFrame for visualization
#     df = pd.DataFrame(
#         {"z1": z_pca[:, 0], "z2": z_pca[:, 1], "image_paths": image_paths}
#     )

#     # Plot with Plotly
#     fig = px.scatter(df, x="z1", y="z2", hover_data=["image_paths"])
#     fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode="markers"))
#     fig.show()


# Function to encode images in base64 for hover text
def encode_image(image_path):
    """
    Encodes an image located at the given path into a base64 string.

    This function reads an image from the specified path, converts it to a
    base64-encoded string, and formats it as an HTML image tag. This is useful
    for embedding images directly into HTML content, such as hover text in
    interactive plots.

    Args:
        image_path (str): The file path to the image to be encoded.

    Returns:
        str: A base64-encoded string representing the image, formatted as an
             HTML <img> tag with specified width and height.
    """
    pil_img = Image.open(image_path)  # Open the image file
    buff = BytesIO()  # Create a buffer to hold the image data
    pil_img.save(buff, format="PNG")  # Save the image data to the buffer in PNG format
    img_str = base64.b64encode(buff.getvalue()).decode(
        "utf-8"
    )  # Encode the buffer data to base64
    return f'<img src="data:image/png;base64,{img_str}" width="100" height="100">'  # Format as HTML <img> tag


# Function to generate latent space visualization with image hovers and colored by label
def visualize_with_hover_images(csv_filename, output_file="latent_space_hover.html"):
    # Load the CSV containing latent z values and image paths
    latent_z_values = pd.read_csv(csv_filename)

    # Perform PCA on the latent z values to reduce dimensions to 2D for visualization
    z_values = latent_z_values[
        [f"z{i}" for i in range(len(latent_z_values.columns) - 2)]
    ]
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_values)

    # Add the PCA result (z1 and z2) to the DataFrame
    latent_z_values["z1"] = z_pca[:, 0]
    latent_z_values["z2"] = z_pca[:, 1]

    # Apply encoding to image paths for hover data
    latent_z_values["image_hover"] = latent_z_values["image_path"].apply(encode_image)

    # Plot with Plotly Express, hover shows image, color by label
    fig = px.scatter(
        latent_z_values,
        x="z1",
        y="z2",
        color="label",
        hover_data=["image_hover"],
        title="Latent Space Visualization with Hover Images and Labels",
        labels={"z1": "Latent Dimension 1", "z2": "Latent Dimension 2"},
        opacity=0.7,
    )

    # Customize the hover template to show the image
    fig.update_traces(
        marker=dict(size=5), hovertemplate="<b>Image:</b><br>%{customdata[0]}"
    )

    # Save the plot as an HTML file
    fig.write_html(output_file)
    print(f"Interactive latent space visualization saved to {output_file}")

    # Optionally, display the plot in the browser
    fig.show()


# Validation function
def test(model, testloader, kl_weight):
    model.eval()
    val_loss = 0.0
    mse_loss_total = 0.0
    kl_loss_total = 0.0
    with torch.no_grad():
        for x, _, _ in testloader:
            x = x.to(device)
            x_hat, mu, logvar, z = model(x)

            # Resize the output (x_hat) to match the input size (x) before computing the loss
            x_hat = F.interpolate(
                x_hat, size=x.shape[2:], mode="bilinear", align_corners=False
            )

            mse_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
            kl_loss = kl_divergence(mu, logvar) * kl_weight
            total_loss = mse_loss + kl_loss

            val_loss += total_loss.item()
            mse_loss_total += mse_loss.item()
            kl_loss_total += kl_loss.item()

            # Debugging: Log the latent vector z during testing for insight
            print(f"Test: Latent vector z: {z}")

    avg_val_loss = val_loss / len(testloader.dataset)
    avg_mse_loss = mse_loss_total / len(testloader.dataset)
    avg_kl_loss = kl_loss_total / len(testloader.dataset)

    return avg_val_loss, avg_mse_loss, avg_kl_loss


# Plot loss function for training and validation
def plot_losses(
    train_losses,
    val_losses,
    train_mse_losses,
    val_mse_losses,
    num_epochs,
    latent_dims,
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
    plt.savefig(
        f"plots/_vae_loss_plot_epochs_{str(num_epochs)}_latent_{str(latent_dims)}.png"
    )
    plt.show()


def main():
    print("Starting VAE Training Process...")

    data_dir = "dataset/town7_dataset/"
    writer = SummaryWriter(f"runs/" + "auto-encoder")

    # Image transformations for training and testing
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(30),
            # transforms.ColorJitter(
            #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            # ),
            transforms.ToTensor(),
        ]
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
    print(f"Initializing VAE with Latent Space Size: {LATENT_SPACE}")
    model = VariationalAutoencoder(latent_dims=LATENT_SPACE, num_epochs=NUM_EPOCHS).to(
        device
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    kl_weight = KL_WEIGHT_INITIAL  # Start with a small KL weight

    patience = 50  # Stop training if no improvement for 50 epochs
    best_val_loss = float("inf")  # Initialize best_val_loss to infinity
    patience_counter = 0  # Counter for patience

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, factor=0.5, verbose=True
    )

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}: Training Phase")
        # Increase KL weight slowly over time (annealing)
        kl_weight = min(KL_WEIGHT_INITIAL + epoch * 0.0001, 1.0)

        # Train model
        (
            train_loss,
            train_mse_loss,
            train_kl_loss,
            z_values_tensor,
            all_image_paths,
        ) = train(model, trainloader, optimizer, epoch, kl_weight)

        # Validate model
        val_loss, val_mse_loss, val_kl_loss = test(model, validloader, kl_weight)

        scheduler.step(val_loss)  # Step the learning rate scheduler at each epoch

        # Append the losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_mse_losses.append(train_mse_loss)
        val_mse_losses.append(val_mse_loss)
        train_kl_losses.append(train_kl_loss)
        val_kl_losses.append(val_kl_loss)

        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)

        # Ensure that all print output is flushed immediately
        sys.stdout.flush()

        # Print progress
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}:")
        print(
            f" for values Number of Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE} | Latent Space: {LATENT_SPACE} | Learning Rate: {LEARNING_RATE} | KL_Weight: {kl_weight} "
        )
        print(f"  Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        print(
            f"  Train MSE Loss: {train_mse_loss:.3f} | Train KL Loss: {train_kl_loss:.3f}"
        )
        print(f"  Val MSE Loss: {val_mse_loss:.3f} | Val KL Loss: {val_kl_loss:.3f}")

        # Check if this is the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # Update best validation loss
            patience_counter = 0  # Reset patience counter
            print(f"New best validation loss: {best_val_loss:.3f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping: No improvement for {patience} epochs")
            break

        # Reduce the leaning after 200 epocs
        if epoch == 200:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-5

        # Save reconstructions for visual inspection
        if epoch % 10 == 0:
            save_reconstructions(model, trainloader, epoch)

        # Save model checkpoint every 50 epochs
        if epoch % 50 == 0:
            model_checkpoint_path = f"model/{NUM_EPOCHS}_epochs_LATENT_SPACE_LF/model_checkpoint_epoch_{epoch}.pth"
            os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), model_checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch}")

    # Plot losses after training
    plot_losses(
        train_losses,
        val_losses,
        train_mse_losses,
        val_mse_losses,
        NUM_EPOCHS,
        LATENT_SPACE,
        train_kl_losses,
        val_kl_losses,
    )

    print("Training Complete. Saving the model...")
    model.save()

    # Save latent space visualization with hoverable images and labels
    csv_filename = f"latent_{LATENT_SPACE}_values_epoch_{NUM_EPOCHS - 1}.csv"  # Assuming final epoch
    visualize_with_hover_images(csv_filename)
    # Visualize latent space
    # visualize_latent_space(z_values_tensor, all_image_paths)  # Call with final z values


if __name__ == "__main__":
    try:
        main()
    except OSError as e:
        print(f"An OSError occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nTerminating...")
        sys.stdout.flush()
