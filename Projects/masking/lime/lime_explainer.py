import sys
import os
from PIL import Image as PilImage
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt

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
            map_location=device, weights_only=True,
        )
    )
    encoder.eval()
    return encoder


def load_vae_decoder():
    decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
    decoder.load_state_dict(
        torch.load(
            "/home/selab/darshan/git-repos/model/epochs_500_latent_128/decoder_model.pth",
            map_location=device, weights_only=True,
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
            map_location=device, weights_only=True,
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


# Main function for LIME explanation
def main():
    image_path = "/home/selab/darshan/git-repos/dataset/town7_dataset/test/town7_000692.png"  # Example image path
    img = load_image(image_path)

    # Load the encoder, decoder, and classifier
    encoder = load_vae_encoder()
    decoder = load_vae_decoder()
    classifier = load_classifier()

    transform = transforms.Compose(
        [
            transforms.Resize((160, 80)),  # Resize to match the model's input
            transforms.ToTensor(),
        ]
    )

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

    # Show the explanation
    print("Explanation:", explanation.as_list())

    # Extract numeric indices of important features
    important_features = [
        int(feature.split("_")[-1]) for feature, _ in explanation.as_list()
    ]
    print(f"Important features identified by LIME: {important_features}")

    # Mask those important features
    masked_latent_vector = latent_vector.clone()
    #masked_latent_vector[:, important_features] = 0  # Mask important features

    # Pass the masked latent vector to the decoder and reconstruct the image
    with torch.no_grad():
        reconstructed_image = decoder(masked_latent_vector)

    # Display the original and reconstructed images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.title("Reconstructed Image (Masked Latent)")
    plt.axis("off")
    plt.show()

    # Print latent vectors before and after masking
    print(f"Original Latent Vector: {latent_vector}")
    print(f"Masked Latent Vector: {masked_latent_vector}")

    # Classifier prediction on original and masked latent vector
    with torch.no_grad():
        original_pred = classifier(latent_vector)
        masked_pred = classifier(masked_latent_vector)

    print(f"Original Prediction: {torch.argmax(original_pred, dim=1).item()}")
    print(f"Masked Prediction: {torch.argmax(masked_pred, dim=1).item()}")


if __name__ == "__main__":
    main()
