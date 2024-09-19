import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels
from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# Paths to  models
encoder_path = "model/var_encoder_model.pth"
decoder_path = "model/decoder_model.pth"
classifier_path = "model/classifier.pth"

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


encoder = VariationalEncoder(latent_dims=128).to(device)
decoder = Decoder(latent_dims=128).to(device)
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
            transforms.Resize((160, 80)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension


# Function to test a single image
def test_single_image(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Pass image through the encoder to get latent representation
    with torch.no_grad():
        # mu, logvar, latent_vector = encoder(image)
        latent_vector = encoder(image)[2]  # Get only the latent vector z

        # Pass latent vector through the decoder to reconstruct the image
        reconstructed_image = decoder(latent_vector)

        # Get classifier prediction
        prediction = classifier(latent_vector)

        # Print prediction (0 for STOP, 1 for GO)
        predicted_class = torch.argmax(prediction, dim=1).item()
        if predicted_class == 0:
            print("Prediction: STOP")
        else:
            print("Prediction: GO")

        return reconstructed_image, predicted_class


# Function to visualize the original and reconstructed image
def show_images(original_image_path, reconstructed_image):
    original_image = Image.open(original_image_path)
    reconstructed_image = (
        reconstructed_image.cpu().squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5
    )  # Un-normalize

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")

    axs[1].imshow(reconstructed_image)
    axs[1].set_title("Reconstructed Image")

    plt.show()


# Test the image
image_path = "dataset/town7_dataset/test/town7_000084.png"
reconstructed_image, prediction = test_single_image(image_path)

# Call the visualization function
show_images(image_path, reconstructed_image)
