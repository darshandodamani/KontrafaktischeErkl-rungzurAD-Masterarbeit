# location: Projects/masking/lime/single_image_and_lime_based_masking.py
from calendar import c
import dis
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import signal
import sys
import os
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd


# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels

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


# 1. Read the CSV file to get the label for a specific image
def get_actual_label(csv_file, image_path):
    # Extract image name from the path
    image_name = os.path.basename(image_path)

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Find the label corresponding to the image name
    label_row = df[df["image_filename"] == image_name]
    if label_row.empty:
        raise ValueError(f"Image {image_name} not found in the CSV file.")

    # Extract label (assuming the label column is named 'label')
    label = label_row["label"].values[0]

    # Convert label to numerical format (STOP=0, GO=1)
    actual_label = 0 if label == "STOP" else 1
    return actual_label


# Test the function for multiple images
image_path = "dataset/town7_dataset/test/town7_009628.png"
csv_file = "dataset/town7_dataset/test/labeled_test_data_log.csv"

# Get the label from the CSV
actual_label = get_actual_label(csv_file, image_path)

# Print the image and its corresponding label
print(
    f"Image {image_path} is selected and its label is: {'STOP' if actual_label == 0 else 'GO'}"
)

# save the image in the plot and print its size and label also in the plot
image = Image.open(image_path)
plt.imshow(image)
plt.title(
    f"Image: {os.path.basename(image_path)}\nLabel: {'STOP' if actual_label == 0 else 'GO'}\nSize: {image.size}"
)
plt.savefig("original_image.png")
# plt.show()


# 2. Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    print(f"Input Image Original shape: {np.array(image).shape}")
    transform = transforms.Compose(
        [
            transforms.Resize((80, 160)),  # Assuming CARLA image size
            transforms.ToTensor(),
        ]
    )
    transformed_image = transform(image)
    transformed_image_tensor = (
        transformed_image
        if isinstance(transformed_image, torch.Tensor)
        else torch.tensor(transformed_image)
    )
    print(
        f"Input Image Transformed shape: {transformed_image_tensor.cpu().numpy().shape}"
    )
    return transformed_image_tensor.unsqueeze(0).to(device)  # Add batch dimension


# save the preprocessed_image in the plot and print its size and label also in the plot
image = preprocess_image(image_path)
plt.imshow(image.cpu().detach().squeeze().numpy().transpose(1, 2, 0))
plt.title(
    f"Preprocessed Image: {os.path.basename(image_path)}\nLabel: {'STOP' if actual_label == 0 else 'GO'}\nSize: {image.shape[2:]}"
)
plt.savefig("preprocessed_image.png")
# plt.show()


# Send the image to the encoder and get the latent vector
image = preprocess_image(image_path)
with torch.no_grad():
    latent_vector = encoder(image)[2]
    print(f"Latent vector shape: {latent_vector.shape}")
    print(f"Latent vector before masking:\n {latent_vector}")

    # Get the classifier prediction
    prediction = classifier(latent_vector)
    print(f"Prediction: {prediction}")
    print(f"Prediction shape: {prediction.shape}")

    # Get the predicted class
    predicted_class = torch.argmax(prediction, dim=1).item()
    predicted_label = "STOP" if predicted_class == 0 else "GO"
    print(f"Predicted Label: {predicted_label}")

    # compare the actual and predicted labels
    print(f"Actual Label: {'STOP' if actual_label == 0 else 'GO'}")

    # Decode the latent vector to get the reconstructed image
    reconstructed_image = decoder(latent_vector)
    print(f"Reconstructed Image shape: {reconstructed_image.shape}")

    # Resize the reconstructed image to match the original input dimensions before calculating MSE loss
    reconstructed_image_resized = F.interpolate(
        reconstructed_image,
        size=image.shape[2:],  # Resize to match the input image size
        mode="bilinear",  # Interpolation mode
        align_corners=False,
    )

    # Now compute the MSE loss between the resized reconstructed image and the original input image
    reconstruction_loss = F.mse_loss(reconstructed_image_resized, image).item()
    print(f"Reconstruction Loss Before: {reconstruction_loss}")

    # Visualize the original and reconstructed images
    original_image = image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_image = (
        reconstructed_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_image)
    axs[0].set_title(f"Original Image\nActual: {'STOP' if actual_label == 0 else 'GO'}")

    axs[1].imshow(reconstructed_image)
    axs[1].set_title(
        f"Reconstructed Image without Masking\nLoss: {reconstruction_loss:.4f}\n Predicted: {predicted_label}"
    )

    plt.savefig("reconstructed_image_without_masking.png")
    # plt.show()


# 3. Masking
# Apply LIME on the latent space
def apply_lime_on_latent_space(latent_vector, classifier):
    latent_vector_np = latent_vector.cpu().numpy().flatten()
    print("--------------------")
    print(f"Latent vector shape: {latent_vector_np.shape}")

    # LIME Tabular Explainer
    explainer = LimeTabularExplainer(
        latent_vector_np.reshape(1, -1),
        feature_names=[f"latent_dim_{i}" for i in range(latent_vector_np.shape[0])],
        class_names=["STOP", "GO"],
        discretize_continuous=False,
    )
    # Explain the instance
    explanation = explainer.explain_instance(
        latent_vector_np,
        lambda x: F.softmax(
            classifier(torch.tensor(x).view(-1, 128).float().to(device)), dim=1
        )
        .cpu()
        .detach()
        .numpy(),
        num_features=5,  # Adjust number of features as needed
    )

    return explanation


# 4. Mask latent features with different methods
def mask_latent_features(latent_vector, important_features, method="zero"):
    masked_latent = latent_vector.clone()
    print("--------------------")
    # Ensure that important_features contains the correct indices
    print(f"Features being masked: {important_features}")

    if method == "zero":
        # Set important features to 0
        print("Masking features by setting them to 0...")
        masked_latent[:, important_features] = 0

    elif method == "median":
        # Set important features to the median value
        median_val = torch.median(latent_vector).item()
        print(f"Meadian value: {median_val}")
        print(f"Masking features by setting them to median value {median_val}...")
        masked_latent[:, important_features] = median_val

    elif method == "random":
        # Set important features to random values (within the range of the latent vector)
        random_vals = torch.randn_like(masked_latent[:, important_features])
        print("Masking features by setting them to random values...")
        masked_latent[:, important_features] = random_vals

    print(f"Masked Latent Vector ({method}):\n", masked_latent)
    return masked_latent


# 5. Function to save and visualize images
# Function to save and visualize images before and after masking
def save_images(
    original_image,
    reconstructed_image_before_masking,
    reconstructed_image_after_masking,
    reconstruction_loss_before,
    reconstruction_loss_after,
    method,
    actual_label,
    predicted_label_before,
    predicted_label_after,
    image_name,
):
    # Ensure the results directory exists
    results_dir = "plots/reconstruction"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Convert tensors to numpy arrays for visualization
    original_image_np = (
        original_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    )
    reconstructed_before_np = (
        reconstructed_image_before_masking.cpu()
        .detach()
        .squeeze()
        .numpy()
        .transpose(1, 2, 0)
    )
    reconstructed_after_np = (
        reconstructed_image_after_masking.cpu()
        .detach()
        .squeeze()
        .numpy()
        .transpose(1, 2, 0)
    )

    # Create a figure and save the original image separately
    plt.figure(figsize=(5, 5))
    plt.imshow(original_image_np)
    plt.title(f"Original Image")
    # plt.title(f"Original Image\nLabel: {'STOP' if actual_label == 0 else 'GO'}")
    plt.savefig(f"{results_dir}/original_{image_name}.png")
    plt.close()  # Close the figure to free memory

    # Create a figure and save the reconstructed image before masking separately
    plt.figure(figsize=(5, 5))
    plt.imshow(reconstructed_before_np)
    plt.title(
        f"Reconstructed Before Masking\nLoss: {reconstruction_loss_before:.4f}\nPred: {predicted_label_before}"
    )
    plt.savefig(f"{results_dir}/before_masking_{image_name}.png")
    plt.close()

    # Create a figure and save the reconstructed image after masking separately
    plt.figure(figsize=(5, 5))
    plt.imshow(reconstructed_after_np)
    plt.title(
        f"Reconstructed After Masking ({method})\nLoss: {reconstruction_loss_after:.4f}\nPred: {predicted_label_after}"
    )
    plt.savefig(f"{results_dir}/after_masking_{method}_{image_name}.png")
    plt.close()

    # Optionally, you can save all three images in one figure as well
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    axs[0].imshow(original_image_np)
    axs[0].set_title(f"Original Image\nLabel: {'STOP' if actual_label == 0 else 'GO'}")

    # Plot reconstructed image before masking
    axs[1].imshow(reconstructed_before_np)
    axs[1].set_title(
        f"Reconstructed Before Masking\nLoss: {reconstruction_loss_before:.4f}\nPred: {predicted_label_before}"
    )

    # Plot reconstructed image after masking
    axs[2].imshow(reconstructed_after_np)
    axs[2].set_title(
        f"Reconstructed After Masking ({method})\nLoss: {reconstruction_loss_after:.4f}\nPred: {predicted_label_after}"
    )

    # Save the combined figure
    plt.savefig(f"{results_dir}/{method}_reconstruction_combined_{image_name}.png")
    plt.close()


# Define the global results table
results = []


# Append the results for each image and masking method
def append_results(
    image_name,
    image_size,
    recon_loss_before,
    recon_loss_after,
    masking_method,
    num_features,
    important_features,
    important_feature_values_before,
    important_feature_values_after,
    prediction_change,
):
    results.append(
        {
            "Image Name": image_name,
            "Image Size": str(image_size),  # Format the image size as a string
            "Reconstruction Loss Before Masking": round(
                recon_loss_before, 6
            ),  # Limit the precision for losses
            "Reconstruction Loss After Masking": round(recon_loss_after, 6),
            "Masking Method": masking_method,  # Method used for masking
            "Number of Features Masked": num_features,  # Number of important features masked
            "Important Features Masked": important_features,  # List of important features
            "Important Features Values Before Masking": important_feature_values_before,  # Values before masking
            "Important Features Values After Masking": important_feature_values_after,  # Values after masking
            "Prediction Changed After Masking": prediction_change,  # True/False if prediction changed
        }
    )


# Process and append results for each masking method
def process_results(
    image_name,
    image_size,
    recon_loss_before,
    recon_loss_after,
    method,
    important_features,
    original_prediction,
    masked_prediction,
    masked_latent_vector,
):
    # Check if the prediction changed
    prediction_changed = original_prediction != masked_prediction

    # Number of important features identified
    num_features = len(important_features)

    # Append the result to the table
    append_results(
        image_name=image_name,
        image_size=image_size,
        recon_loss_before=recon_loss_before,
        recon_loss_after=recon_loss_after,
        masking_method=method,
        num_features=num_features,
        important_features=important_features,
        important_feature_values_before=latent_vector[:, important_features]
        .cpu()
        .numpy()
        .tolist(),
        important_feature_values_after=masked_latent_vector[:, important_features]
        .cpu()
        .numpy()
        .tolist(),
        prediction_change=prediction_changed,
    )


# Main function to test a single image with LIME and masking
def test_single_image(image_path, csv_file):
    actual_label = get_actual_label(csv_file, image_path)
    print(
        f"Image {image_path} is selected and its label is: {'STOP' if actual_label == 0 else 'GO'}"
    )
    image = preprocess_image(image_path)

    with torch.no_grad():
        # Step 1: Get the latent vector from the encoder
        latent_vector = encoder(image)[2]
        print(f"Original Latent Vector:\n {latent_vector}")

        # Step 2: Get original prediction from the classifier
        original_prediction = classifier(latent_vector)
        original_class = torch.argmax(original_prediction, dim=1).item()
        predicted_label = "STOP" if original_class == 0 else "GO"
        print(f"Original Prediction: {predicted_label}")

        # Step 3: Decode the latent vector to get the reconstructed image
        reconstructed_image = decoder(latent_vector)
        reconstructed_image_resized = F.interpolate(
            reconstructed_image,
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed_image_resized, image).item()
        print(f"Reconstruction Loss (Without Masking): {reconstruction_loss}")

        # Visualize the original and reconstructed images
        save_images(
            image,
            reconstructed_image,
            reconstructed_image,
            reconstruction_loss,
            reconstruction_loss,
            "original",
            actual_label,
            predicted_label,
            predicted_label,
            os.path.basename(image_path),
        )

        # Step 4: Apply LIME to identify important features
        explanation = apply_lime_on_latent_space(latent_vector, classifier)
        important_features = [
            int(feature.split("_")[-1]) for feature, _ in explanation.as_list()
        ]
        print(f"Important features identified by LIME: {important_features}")

        # Step 5: Mask the latent vector
        # for method in ["zero", "median", "random"]:
        for method in ["median"]:
            masked_latent_vector = mask_latent_features(
                latent_vector, important_features, method
            )
            masked_prediction = classifier(masked_latent_vector)
            masked_class = torch.argmax(masked_prediction, dim=1).item()
            masked_label_str = "STOP" if masked_class == 0 else "GO"
            print(f"Masked Prediction ({method}): {masked_label_str}")

            # Step 6: Decode the masked latent vector to get the masked reconstructed image
            masked_reconstructed_image = decoder(masked_latent_vector)
            masked_reconstructed_image_resized = F.interpolate(
                masked_reconstructed_image,
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            masked_reconstruction_loss = F.mse_loss(
                masked_reconstructed_image_resized, image
            ).item()
            print(
                f"Reconstruction Loss after Masking with ({method}) method: {masked_reconstruction_loss}"
            )

            save_images(
                image,
                reconstructed_image,
                masked_reconstructed_image,
                reconstruction_loss,
                masked_reconstruction_loss,
                method,
                actual_label,
                predicted_label,
                masked_label_str,
                os.path.basename(image_path),
            )

            # Step 7: Check if prediction changed after masking
            if masked_class != original_class:
                print(
                    f"Prediction changed after masking with {method}. Generating counterfactual explanation."
                )

                # Step 8: Re-encode the masked reconstructed image to get new latent vector
                re_encoded_latent_vector = encoder(masked_reconstructed_image_resized)[
                    2
                ]
                print(f"Re-encoded Latent Vector: {re_encoded_latent_vector}")

                # Step 9: Classify the re-encoded latent vector
                re_encoded_prediction = classifier(re_encoded_latent_vector)
                re_encoded_class = torch.argmax(re_encoded_prediction, dim=1).item()
                re_encoded_label = "STOP" if re_encoded_class == 0 else "GO"
                print(f"Re-encoded Prediction: {re_encoded_label}")

                # Compare the re-encoded prediction with the original prediction
                if re_encoded_class != original_class:
                    print(
                        f"Counterfactual explanation found! Re-encoded prediction differs from original."
                    )
                else:
                    print(f"Re-encoded prediction remains the same as the original.")
            else:
                print(
                    f"No prediction change after masking with {method}. No counterfactual explanation generated."
                )

            # Adding result to the table
            process_results(
                image_name=os.path.basename(image_path),
                image_size=image.shape[2:],  # Shape of the input image
                recon_loss_before=reconstruction_loss,
                recon_loss_after=masked_reconstruction_loss,
                method=method,
                important_features=important_features,
                original_prediction=original_class,
                masked_prediction=masked_class,
                masked_latent_vector=masked_latent_vector,
            )


# After testing, print and save the results to a CSV
def print_and_save_results():
    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)
    print("\nFinal Interpretation Results:")
    print(results_df)

    # Save the DataFrame to a CSV file
    results_df.to_csv("plots/reconstruction/masking_analysis_results.csv", index=False)


# Test the function
test_single_image(
    "dataset/town7_dataset/test/town7_009628.png",
    "dataset/town7_dataset/test/labeled_test_data_log.csv",
)

# to print and save results
print_and_save_results()