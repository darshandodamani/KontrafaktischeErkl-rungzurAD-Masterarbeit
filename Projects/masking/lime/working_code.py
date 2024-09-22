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

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  the Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder


def load_image(image_path):
    return PilImage.open(image_path).convert("RGB")


def load_vae():
    encoder = VariationalEncoder(latent_dims=256).to(device)  # Send encoder to device
    encoder.load_state_dict(
        torch.load("/home/selab/darshan/git-repos/model/var_encoder_model.pth")
    )
    encoder.eval()

    decoder = Decoder(latent_dims=256).to(device)  # Send decoder to device
    decoder.load_state_dict(
        torch.load("/home/selab/darshan/git-repos/model/decoder_model.pth")
    )
    decoder.eval()

    return encoder, decoder


def batch_predict_latent(images, encoder, transform):
    encoder.to(device)

    batch = torch.stack([transform(PilImage.fromarray(image)) for image in images]).to(
        device
    )

    with torch.no_grad():
        mu, logvar, z = encoder(batch)
        logits = z.sum(dim=1)
        probs = F.softmax(logits, dim=0).unsqueeze(0).repeat(batch.size(0), 1)

    return probs.cpu().numpy()


def apply_latent_space_mask(latent_vector, mask_indices):
    masked_latent_vector = latent_vector.clone()
    masked_latent_vector[:, mask_indices] = 0
    return masked_latent_vector


def main():
    image_path = (
        "/home/selab/darshan/git-repos/dataset/town7_dataset/train/town7_000275.png"
    )

    img = load_image(image_path)

    encoder, decoder = load_vae()

    transform = transforms.Compose(
        [
            transforms.Resize((160, 80)),
            transforms.ToTensor(),
        ]
    )

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(img),
        lambda x: batch_predict_latent(x, encoder, transform),
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

    # Now apply the mask to the latent space
    mu, logvar, latent_vector = encoder(transform(img).unsqueeze(0).to(device))

    mask_indices = [0, 1, 2, 3, 4]
    masked_latent_vector = apply_latent_space_mask(latent_vector, mask_indices)

    masked_image = decoder(masked_latent_vector)

    masked_image_np = masked_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    plt.imshow(masked_image_np)
    plt.title("Reconstructed Image with Masked Latent Space")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
