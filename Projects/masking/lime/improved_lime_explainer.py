import sys
import os
from PIL import Image as PilImage
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt
import csv

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel

# Set device for GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load the image
def load_image(image_path):
    return PilImage.open(image_path).convert("RGB")


# Load the VAE encoder and decoder models
def load_vae_encoder():
    encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
    encoder.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/epochs_500_latent_128/var_encoder_model.pth",
            map_location=device,
            weights_only=True,
        )
    )
    encoder.eval()
    return encoder


def load_vae_decoder():
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    decoder.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/epochs_500_latent_128/decoder_model.pth",
            map_location=device,
            weights_only=True,
        )
    )
    decoder.eval()
    return decoder


# Load classifier
def load_classifier():
    classifier = ClassifierModel(input_size=128, hidden_size=128, output_size=2).to(
        device
    )
    classifier.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/epochs_500_latent_128/classifier_final.pth",
            map_location=device,
            weights_only=True,
        )
    )
    classifier.eval()
    return classifier


# Get the latent vector from an image using the encoder
def get_latent_vector(image, encoder, transform):
    image_tensor = transform(image).unsqueeze(0).to(device)  # Convert image to tensor
    with torch.no_grad():
        _, _, z = encoder(image_tensor)  # Get the latent vector
    return z


# LIME's batch predict function for latent space
def batch_predict_latent_space(latent_vectors, classifier):
    with torch.no_grad():
        outputs = classifier(latent_vectors)
        probs = F.softmax(outputs, dim=1)
    return probs.cpu().numpy()


# Function to save LIME results and predictions
def save_lime_results(
    image_path, important_features, original_pred, masked_pred, reconstruction_loss
):
    results_file = "lime_results.csv"
    file_exists = os.path.isfile(results_file)

    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Image Path",
                    "Important Features",
                    "Original Prediction",
                    "Masked Prediction",
                    "Reconstruction Loss",
                ]
            )

        writer.writerow(
            [
                image_path,
                important_features,
                original_pred,
                masked_pred,
                reconstruction_loss,
            ]
        )


# Function to run LIME explanations and process multiple images
def process_images(image_paths):
    encoder = load_vae_encoder()
    decoder = load_vae_decoder()
    classifier = load_classifier()

    transform = transforms.Compose(
        [
            transforms.Resize((160, 80)),  # Resize to match the model's input
            transforms.ToTensor(),
        ]
    )

    for image_path in image_paths:
        img = load_image(image_path)

        # Get the latent vector from the image
        latent_vector = get_latent_vector(img, encoder, transform)

        # Convert the latent vector to numpy array for LIME
        latent_vector_np = latent_vector.cpu().numpy().flatten()

        # Use LIME Tabular Explainer to explain the classifier’s decision based on the latent space
        explainer = LimeTabularExplainer(
            latent_vector_np.reshape(1, -1),
            feature_names=[f"latent_dim_{i}" for i in range(latent_vector_np.shape[0])],
            class_names=["STOP", "GO"],
            discretize_continuous=False,
        )

        # Explain the instance
        explanation = explainer.explain_instance(
            latent_vector_np,
            lambda x: batch_predict_latent_space(
                torch.tensor(x).view(-1, 128).float().to(device), classifier
            ),
            num_features=5,  # Explaining with more features
        )

        # Extract numeric indices of important features
        important_features = [
            int(feature.split("_")[-1]) for feature, _ in explanation.as_list()
        ]
        print(f"Important features for {image_path}: {important_features}")

        # Mask those important features and calculate reconstruction loss
        # Mask those important features and calculate reconstruction loss
        for num_features in [1, 2, 5, 10]:  # Vary the number of masked features
            masked_latent_vector = latent_vector.clone()
            masked_latent_vector[:, important_features[:num_features]] = (
                0  # Mask top important features
            )

            with torch.no_grad():
                reconstructed_image = decoder(masked_latent_vector)

                # Resize the reconstructed image and original image to match size
                reconstructed_image_resized = F.interpolate(
                    reconstructed_image,
                    size=(160, 80),
                    mode="bilinear",
                    align_corners=False,
                )
                original_image_resized = transform(img).unsqueeze(0).to(device)

                # Calculate reconstruction loss between original and reconstructed image
                reconstruction_loss = F.mse_loss(
                    reconstructed_image_resized, original_image_resized
                ).item()

                # Classifier prediction on original and masked latent vector
                original_pred = torch.argmax(classifier(latent_vector), dim=1).item()
                masked_pred = torch.argmax(
                    classifier(masked_latent_vector), dim=1
                ).item()

                print(
                    f"Original Prediction: {original_pred}, Masked Prediction: {masked_pred}, Reconstruction Loss: {reconstruction_loss}"
                )

                # Save the LIME results
                save_lime_results(
                    image_path,
                    important_features[:num_features],
                    original_pred,
                    masked_pred,
                    reconstruction_loss,
                )

                # Show the original and reconstructed images
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.title(f"Original Image ({original_pred})")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(
                    reconstructed_image_resized.squeeze().permute(1, 2, 0).cpu().numpy()
                )
                plt.title(f"Masked Reconstruction ({masked_pred})")
                plt.axis("off")
                plt.savefig(f"masked_reconstruction_{image_path.split('/')[-1]}.png")


if __name__ == "__main__":
    # List of image paths to process
    image_paths = [
        "/home/selab/darshan/git-repos/dataset/town7_dataset/train/town7_012099.png",
        "/home/selab/darshan/git-repos/dataset/town7_dataset/train/town7_011345.png",
        # Add more image paths here for batch processing
    ]

    process_images(image_paths)
