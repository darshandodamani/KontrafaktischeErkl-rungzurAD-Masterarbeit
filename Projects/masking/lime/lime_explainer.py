import sys
import os
from PIL import Image as PilImage
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from lime import lime_image
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

# Adjust the Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

# Import the VariationalEncoder from the autoencoder module
from encoder import VariationalEncoder


# Function to load the image
def load_image(image_path):
    return PilImage.open(image_path).convert("RGB")


# Function to transform the image
def transform_image(image, transform):
    return transform(image).unsqueeze(0)


# Load the VAE encoder model
def load_vae_encoder():
    model = VariationalEncoder(latent_dims=256)
    model.load_state_dict(
        torch.load("/home/selab/darshan/git-repos/model/var_encoder_model.pth")
    )
    model.eval()
    return model


# Function for LIME to make predictions on the latent space
def batch_predict(images, model, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Convert each image from NumPy array to PIL Image, apply the transform, and stack them
    batch = torch.stack([transform(PilImage.fromarray(image)) for image in images]).to(
        device
    )

    with torch.no_grad():
        z = model(batch)
        # Simulate classification (for demonstration, sum of latent vector)
        logits = z.sum(dim=1)
        probs = F.softmax(logits, dim=0).unsqueeze(0).repeat(batch.size(0), 1)

    return probs.cpu().numpy()


# Main function to perform LIME explanation
def main():
    image_path = "/home/selab/darshan/git-repos/dataset/town7_dataset/train/town7_000260.png"  # Adjust to your image path

    img = load_image(image_path)

    model = load_vae_encoder()

    transform = transforms.Compose(
        [
            transforms.Resize((160, 80)),  # Resize to match your model's input
            transforms.ToTensor(),
        ]
    )

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(img),
        lambda x: batch_predict(x, model, transform),
        top_labels=1,
        hide_color=0,
        num_samples=1000,
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=False, num_features=5, hide_rest=False
    )
    img_boundry = mark_boundaries(temp / 255.0, mask)

    plt.imshow(img_boundry)
    plt.title(f"Top Label: {top_label}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
