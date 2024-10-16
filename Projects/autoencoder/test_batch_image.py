import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels
from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier
import numpy as np
import signal
import sys

# Paths to models
encoder_path = "model/epochs_500_latent_128/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128/classifier_final.pth"

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(
    device
)  # Example latent dims
decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
classifier = Classifier().to(device)

encoder.load_state_dict(
    torch.load(encoder_path, map_location=device, weights_only=True)
)
decoder.load_state_dict(
    torch.load(decoder_path, map_location=device, weights_only=True)
)
classifier.load_state_dict(
    torch.load(classifier_path, map_location=device, weights_only=True)
)

encoder.eval()
decoder.eval()
classifier.eval()


# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((160, 80)),  # Assuming CARLA image size
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension


# Function to test a single image with loss calculation and display
def test_single_image(image_path, actual_label):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Pass image through the encoder to get latent representation
    with torch.no_grad():
        latent_vector = encoder(image)[2]  # Get the latent vector z

        # Pass latent vector through the decoder to reconstruct the image
        reconstructed_image = decoder(latent_vector)

        # Calculate reconstruction loss (e.g., MSE Loss)
        reconstruction_loss = F.mse_loss(reconstructed_image, image)

        # Get classifier prediction
        prediction = classifier(latent_vector)

        # Print actual label and predicted label
        predicted_class = torch.argmax(prediction, dim=1).item()

        if predicted_class == 0:
            predicted_label = "STOP"
        else:
            predicted_label = "GO"

        # Print actual vs predicted for cross-verification
        actual_label_str = "STOP" if actual_label == 0 else "GO"
        print(f"Image: {image_path}")
        print(f"Actual: {actual_label_str}, Predicted: {predicted_label}")
        print(f"Reconstruction Loss: {reconstruction_loss.item()}")

        return image, reconstructed_image, reconstruction_loss.item(), predicted_label


# Function to display the original and reconstructed images
def show_images(original_image, reconstructed_image, image_path, reconstruction_loss):
    # Convert tensors to numpy arrays for display
    original_image = original_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_image = (
        reconstructed_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    )

    # Plot images side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[1].imshow(reconstructed_image)
    axs[1].set_title(f"Reconstructed Image\nLoss: {reconstruction_loss:.4f}")

    plt.savefig(f"plots/reconstruction_{image_path.split('/')[-1]}.png")
    plt.show()


# Global list to store reconstruction losses
reconstruction_losses = []


def plot_loss():
    """Plot the reconstruction loss after the testing process is interrupted or completed."""
    plt.figure(figsize=(10, 5))
    plt.plot(reconstruction_losses, label="Reconstruction Loss")
    plt.xlabel("Image Index")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss for Each Image")
    plt.legend()
    plt.savefig("plots/reconstruction_loss_over_batch.png")
    plt.show()


# Signal handler to handle interruption (Ctrl+C) and plot the graph
def signal_handler(sig, frame):
    print("Process interrupted. Saving the plot of reconstruction loss...")
    plot_loss()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)  # Catch interruption signal (Ctrl+C)


# Testing on batch of images
def test_on_batch(test_loader):
    try:
        for batch_idx, (images, labels, image_paths) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Pass through encoder and classifier
            latent_vectors = encoder(images)[2]
            predictions = classifier(latent_vectors)
            predicted_classes = torch.argmax(predictions, dim=1)

            for i in range(images.size(0)):
                reconstructed_image = decoder(latent_vectors[i : i + 1])

                # Resize the reconstructed image to match the original input dimensions
                reconstructed_image = F.interpolate(
                    reconstructed_image,
                    size=images[i : i + 1].shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

                actual_label = labels[i].item()
                predicted_label = predicted_classes[i].item()

                # Calculate reconstruction loss
                reconstruction_loss = F.mse_loss(reconstructed_image, images[i : i + 1])

                # Print the actual and predicted class for the image
                actual_label_str = "STOP" if actual_label == 0 else "GO"
                predicted_label_str = "STOP" if predicted_label == 0 else "GO"
                print(f"Image: {image_paths[i]}")
                print(f"Actual: {actual_label_str}, Predicted: {predicted_label_str}")
                print(f"Reconstruction Loss: {reconstruction_loss.item()}")

                # Store the reconstruction loss for plotting
                reconstruction_losses.append(reconstruction_loss.item())

                # Visualize the original and reconstructed image
                show_images(
                    images[i],
                    reconstructed_image,
                    image_paths[i],
                    reconstruction_loss.item(),
                )

    except Exception as e:
        print(f"Error encountered during testing: {e}")
    finally:
        # Always plot the graph when the function exits
        plot_loss()


# Load test data and test the batch
test_dataset = CustomImageDatasetWithLabels(
    img_dir="dataset/town7_dataset/test/",
    csv_file="dataset/town7_dataset/test/labeled_test_data_log.csv",
    transform=transforms.Compose([transforms.ToTensor()]),
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Test the batch
test_on_batch(test_loader)
