import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from encoder import VariationalEncoder
from decoder import Decoder
import csv
from classifier import ClassifierModel as Classifier

# Paths to models
encoder_path = "model/200_epochs/var_encoder_model.pth"
decoder_path = "model/200_epochs/decoder_model.pth"
classifier_path = "model/200_epochs/classifier.pth"

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VariationalEncoder(latent_dims=128).to(device)  # Match with latent_dims=128
decoder = Decoder(latent_dims=128).to(device)
classifier = Classifier(input_size=128, hidden_size=128, output_size=2).to(
    device
)  # Adjust input_size to 128

encoder.load_state_dict(
    torch.load(encoder_path, map_location=device, weights_only=True)
)
print("Encoder loaded successfully.")
decoder.load_state_dict(
    torch.load(decoder_path, map_location=device, weights_only=True)
)
print("Decoder loaded successfully.")
classifier.load_state_dict(
    torch.load(classifier_path, map_location=device, weights_only=True)
)
print("Classifier loaded successfully.")

encoder.eval()
decoder.eval()
classifier.eval()


# Function to visualize the original and reconstructed image
def show_images(original_image_path, reconstructed_image, save_path=None):
    original_image = Image.open(original_image_path)
    reconstructed_image = (
        reconstructed_image.detach().cpu().squeeze().permute(1, 2, 0).numpy() * 0.5
        + 0.5
    )

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")

    axs[1].imshow(reconstructed_image)
    axs[1].set_title("Reconstructed Image")

    if save_path:
        plt.savefig(save_path)
        print(f"Image saved at: {save_path}")
    else:
        plt.show()


# Function to process a single image
def process_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((160, 80)),  # Resize to match model input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0).to(device)

    # Get latent vector from encoder
    latent_vector = encoder(image)[2]  # Get only the latent vector z
    print(f"Latent vector for {image_path}: {latent_vector}")

    # Pass latent vector through the decoder to reconstruct the image
    reconstructed_image = decoder(latent_vector)
    print(f"Reconstructed image tensor for {image_path}: {reconstructed_image}")

    # Get classifier prediction
    prediction = classifier(latent_vector)

    # Print prediction (0 for STOP, 1 for GO)
    predicted_class = torch.argmax(prediction, dim=1).item()
    if predicted_class == 0:
        print(f"Prediction for {image_path}: STOP")
    else:
        print(f"Prediction for {image_path}: GO")

    return (
        latent_vector,
        reconstructed_image,
        predicted_class,
    )  # Update this to return all three values


# List of image paths to process
image_paths = [
    "dataset/town7_dataset/train/town7_000078.png",
    "dataset/town7_dataset/train/town7_000090.png",
]

# Process each image and visualize the results
for image_path in image_paths:
    print(f"Processing image: {image_path}")
    latent_vector, reconstructed_image, predicted_class = process_image(image_path)
    show_images(
        image_path,
        reconstructed_image,
        save_path=f"reconstructed_{image_path.split('/')[-1]}",
    )


# CSV file to save the results
output_csv = "predictions.csv"

# Process each image, save the results to CSV, and visualize the results
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    header = ["Image", "Prediction"] + [f"Latent_{i}" for i in range(128)]
    writer.writerow(header)

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        latent_vector, reconstructed_image, prediction_label = process_image(
            image_path
        )  # Ensure three values are returned
        latent_vector = latent_vector.detach().cpu().numpy().flatten()
        row = [image_path, prediction_label] + latent_vector.tolist()
        writer.writerow(row)
        show_images(
            image_path,
            reconstructed_image,
            save_path=f"reconstructed_{image_path.split('/')[-1]}",
        )


print(f"Predictions and latent vectors saved to {output_csv}")
print("Script completed.")
