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
    label_row = df[df['image_filename'] == image_name]
    if label_row.empty:
        raise ValueError(f"Image {image_name} not found in the CSV file.")
    
    # Extract label (assuming the label column is named 'label')
    label = label_row['label'].values[0]
    
    # Convert label to numerical format (STOP=0, GO=1)
    actual_label = 0 if label == "STOP" else 1
    return actual_label

# Test the function
# image_path = "dataset/town7_dataset/test/town7_009628.png"
image_path = "dataset/town7_dataset/test/town7_008508.png"
csv_file = "dataset/town7_dataset/test/labeled_test_data_log.csv"

# Get the label from the CSV
actual_label = get_actual_label(csv_file, image_path)

# Print the image and its corresponding label
print(f"Image {image_path} is selected and its label is: {'STOP' if actual_label == 0 else 'GO'}")

#save the image in the plot and print its size and label also in the plot
image = Image.open(image_path)
plt.imshow(image)
plt.title(f"Image: {os.path.basename(image_path)}\nLabel: {'STOP' if actual_label == 0 else 'GO'}\nSize: {image.size}")
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
    transformed_image_tensor = transformed_image if isinstance(transformed_image, torch.Tensor) else torch.tensor(transformed_image)
    print(f"Input Image Transformed shape: {transformed_image_tensor.cpu().numpy().shape}")
    return transformed_image_tensor.unsqueeze(0).to(device)  # Add batch dimension

#save the preprocessed_image in the plot and print its size and label also in the plot
image = preprocess_image(image_path)
plt.imshow(image.cpu().detach().squeeze().numpy().transpose(1, 2, 0))
plt.title(f"Preprocessed Image: {os.path.basename(image_path)}\nLabel: {'STOP' if actual_label == 0 else 'GO'}\nSize: {image.shape[2:]}")
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
        mode="bilinear",       # Interpolation mode
        align_corners=False
    )
    
    # Now compute the MSE loss between the resized reconstructed image and the original input image
    reconstruction_loss = F.mse_loss(reconstructed_image_resized, image).item()
    print(f"Reconstruction Loss Before: {reconstruction_loss}")
    
    # Visualize the original and reconstructed images
    original_image = image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_image = reconstructed_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_image)
    axs[0].set_title(f"Original Image\nActual: {'STOP' if actual_label == 0 else 'GO'}")
    
    axs[1].imshow(reconstructed_image)
    axs[1].set_title(f"Reconstructed Image without Masking\nLoss: {reconstruction_loss:.4f}\n Predicted: {predicted_label}")
    
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
        discretize_continuous=False
    )
    # Explain the instance
    explanation = explainer.explain_instance(
        latent_vector_np,
        lambda x: F.softmax(classifier(torch.tensor(x).view(-1, 128).float().to(device)), dim=1).cpu().detach().numpy(),
        num_features= 5 # Adjust number of features as needed
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
def save_images(original_image, reconstructed_image, reconstruction_loss, method, actual_label, predicted_label, image_name):
    # Ensure the results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    original_image_np = original_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    reconstructed_image_np = reconstructed_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_image_np)
    axs[0].set_title(f"Original Image (Label: {'STOP' if actual_label == 0 else 'GO'})")
    axs[1].imshow(reconstructed_image_np)
    axs[1].set_title(f"Reconstructed Image ({method})\nLoss: {reconstruction_loss:.4f} (Pred: {predicted_label})")
    
    # Save the image in the results directory
    plt.savefig(f"{results_dir}/{method}_reconstruction_{image_name}.png")
    # plt.show()
    
    
# Main function to test a single image with LIME and masking
def test_single_image(image_path, csv_file):
    actual_label = get_actual_label(csv_file, image_path)
    print(f"Image {image_path} is selected and its label is: {'STOP' if actual_label == 0 else 'GO'}")
    image = preprocess_image(image_path)

    with torch.no_grad():
        latent_vector = encoder(image)[2]
        print(f"Original Latent Vector:\n {latent_vector}")
        original_prediction = classifier(latent_vector)
        original_class = torch.argmax(original_prediction, dim=1).item()
        predicted_label = "STOP" if original_class == 0 else "GO"
        print(f"Original Prediction: {predicted_label}")

        reconstructed_image = decoder(latent_vector)
        reconstructed_image_resized = F.interpolate(reconstructed_image, size=image.shape[2:], mode="bilinear", align_corners=False)
        reconstruction_loss = F.mse_loss(reconstructed_image_resized, image).item()
        print(f"Reconstruction Loss (Without Masking): {reconstruction_loss}")

        save_images(image, reconstructed_image, reconstruction_loss, "original", actual_label, predicted_label, os.path.basename(image_path))

        explanation = apply_lime_on_latent_space(latent_vector, classifier)
        important_features = [int(feature.split("_")[-1]) for feature, _ in explanation.as_list()]
        print(f"Important features identified by LIME: {important_features}")

        # for method in ["zero", "median", "random"]:
        for method in ["zero"]:
            masked_latent_vector = mask_latent_features(latent_vector, important_features, method)
            masked_prediction = classifier(masked_latent_vector)
            masked_class = torch.argmax(masked_prediction, dim=1).item()
            masked_label_str = "STOP" if masked_class == 0 else "GO"
            print(f"Masked Prediction ({method}): {masked_label_str}")

            masked_reconstructed_image = decoder(masked_latent_vector)
            masked_reconstructed_image_resized = F.interpolate(masked_reconstructed_image, size=image.shape[2:], mode="bilinear", align_corners=False)
            masked_reconstruction_loss = F.mse_loss(masked_reconstructed_image_resized, image).item()
            print(f"Reconstruction Loss after Masking with ({method}) method: {masked_reconstruction_loss}")

            save_images(image, masked_reconstructed_image, masked_reconstruction_loss, method, actual_label, masked_label_str, os.path.basename(image_path))

test_single_image("dataset/town7_dataset/test/town7_008508.png", "dataset/town7_dataset/test/labeled_test_data_log.csv")

# # Main function to test single image with LIME and masking
# def test_single_image(image_path, actual_label):
#     image = preprocess_image(image_path)
#     with torch.no_grad():
#         # Step 1: Get the latent vector from the encoder
#         latent_vector = encoder(image)[2]
#         print(f"Original Latent Vector:\n {latent_vector}")
        
#         # Step 2: Get original prediction from the classifier
#         original_prediction = classifier(latent_vector)
#         original_class = torch.argmax(original_prediction, dim=1).item()
#         original_label_str = "STOP" if original_class == 0 else "GO"
#         print(f"Original Prediction: {original_label_str}")
        
#         print("--------------------")
        
#         # Step 3: Apply LIME to get important features
#         explanation = apply_lime_on_latent_space(latent_vector, classifier)
#         print("Important features identified by LIME:")
#         important_features = [int(feature.split("_")[-1]) for feature, _ in explanation.as_list()]
#         print(important_features)
        
#         # Step 4: Masking the latent features and checking classification
#         for method in ["median"]:
#         #for method in ["zero", "median", "random"]:
#             masked_latent_vector = mask_latent_features(latent_vector, important_features, method)
            
#             # Step 5: Get prediction after masking
#             masked_prediction = classifier(masked_latent_vector)
#             masked_class = torch.argmax(masked_prediction, dim=1).item()
#             masked_label_str = "STOP" if masked_class == 0 else "GO"
#             print(f"Masked Prediction ({method}): {masked_label_str}")
            
#             # Step 6: Decode the masked latent vector into an image
#             reconstructed_image = decoder(masked_latent_vector)
            
#             # Step 7: Compute losses
#             # Resize the reconstructed image to match the original input dimensions before calculating MSE loss
#             reconstructed_image_resized = F.interpolate(
#                 reconstructed_image,
#                 size=image.shape[2:],  # Resize to match the input image size
#                 mode="bilinear",       # Interpolation mode
#                 align_corners=False
#             )

#             # Now compute the MSE loss between the resized reconstructed image and the original input image
#             reconstruction_loss = F.mse_loss(reconstructed_image_resized, image).item()
#             classification_diff = (original_class != masked_class)

#             print(f"Reconstruction Loss ({method}): {reconstruction_loss}")
#             print(f"Classification Change ({method}): {classification_diff}")

#             # Visualize the original and reconstructed images
#             show_images(image, reconstructed_image, reconstruction_loss, method)


    

# # Function to test a single image with loss calculation and display
# def test_single_image(image_path, actual_label):
#     # Preprocess the image
#     image = preprocess_image(image_path)

#     # Pass image through the encoder to get latent representation
#     with torch.no_grad():
#         latent_vector = encoder(image)[2]  # Get the latent vector z

#         # Pass latent vector through the decoder to reconstruct the image
#         reconstructed_image = decoder(latent_vector)

#         # Calculate reconstruction loss (e.g., MSE Loss)
#         reconstruction_loss = F.mse_loss(reconstructed_image, image)

#         # Get classifier prediction
#         prediction = classifier(latent_vector)

#         # Print actual label and predicted label
#         predicted_class = torch.argmax(prediction, dim=1).item()

#         if predicted_class == 0:
#             predicted_label = "STOP"
#         else:
#             predicted_label = "GO"

#         # Print actual vs predicted for cross-verification
#         actual_label_str = "STOP" if actual_label == 0 else "GO"
#         print(f"Image: {image_path}")
#         print(f"Actual: {actual_label_str}, Predicted: {predicted_label}")
#         print(f"Reconstruction Loss: {reconstruction_loss.item()}")

#         return image, reconstructed_image, reconstruction_loss.item(), predicted_label


# # Function to display the original and reconstructed images
# def show_images(original_image, reconstructed_image, image_path, reconstruction_loss):
#     # Convert tensors to numpy arrays for display
#     original_image = original_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
#     reconstructed_image = (
#         reconstructed_image.cpu().detach().squeeze().numpy().transpose(1, 2, 0)
#     )

#     # Plot images side by side
#     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#     axs[0].imshow(original_image)
#     axs[0].set_title("Original Image")
#     axs[1].imshow(reconstructed_image)
#     axs[1].set_title(f"Reconstructed Image\nLoss: {reconstruction_loss:.4f}")

#     #plt.savefig(f"plots/reconstruction/reconstruction_{image_path.split('/')[-1]}.png")
#     plt.show()


# # Global list to store reconstruction losses
# reconstruction_losses = []


# def plot_loss():
#     """Plot the reconstruction loss after the testing process is interrupted or completed."""
#     plt.figure(figsize=(10, 5))
#     plt.plot(reconstruction_losses, label="Reconstruction Loss")
#     plt.xlabel("Image Index")
#     plt.ylabel("Loss")
#     plt.title("Reconstruction Loss for Each Image")
#     plt.legend()
#     plt.savefig("plots/reconstruction_loss_over_batch.png")
#     plt.show()


# # Signal handler to handle interruption (Ctrl+C) and plot the graph
# def signal_handler(sig, frame):
#     print("Process interrupted. Saving the plot of reconstruction loss...")
#     plot_loss()
#     sys.exit(0)


# signal.signal(signal.SIGINT, signal_handler)  # Catch interruption signal (Ctrl+C)


# # Testing on batch of images
# def test_on_batch(test_loader):
#     try:
#         for batch_idx, (images, labels, image_paths) in enumerate(test_loader):
#             images = images.to(device)
#             labels = labels.to(device)

#             # Pass through encoder and classifier
#             latent_vectors = encoder(images)[2]
#             predictions = classifier(latent_vectors)
#             predicted_classes = torch.argmax(predictions, dim=1)

#             for i in range(images.size(0)):
#                 reconstructed_image = decoder(latent_vectors[i : i + 1])

#                 # Resize the reconstructed image to match the original input dimensions
#                 reconstructed_image = F.interpolate(
#                     reconstructed_image,
#                     size=images[i : i + 1].shape[2:],
#                     mode="bilinear",
#                     align_corners=False,
#                 )

#                 actual_label = labels[i].item()
#                 predicted_label = predicted_classes[i].item()

#                 # Calculate reconstruction loss
#                 reconstruction_loss = F.mse_loss(reconstructed_image, images[i : i + 1])

#                 # Print the actual and predicted class for the image
#                 actual_label_str = "STOP" if actual_label == 0 else "GO"
#                 predicted_label_str = "STOP" if predicted_label == 0 else "GO"
#                 print(f"Image: {image_paths[i]}")
#                 print(f"Actual: {actual_label_str}, Predicted: {predicted_label_str}")
#                 print(f"Reconstruction Loss: {reconstruction_loss.item()}")

#                 # Store the reconstruction loss for plotting
#                 reconstruction_losses.append(reconstruction_loss.item())

#                 # Visualize the original and reconstructed image
#                 show_images(
#                     images[i],
#                     reconstructed_image,
#                     image_paths[i],
#                     reconstruction_loss.item(),
#                 )

#     except Exception as e:
#         print(f"Error encountered during testing: {e}")
#     finally:
#         # Always plot the graph when the function exits
#         plot_loss()


# # Load test data and test the batch
# test_dataset = CustomImageDatasetWithLabels(
#     img_dir="dataset/town7_dataset/test/",
#     csv_file="dataset/town7_dataset/test/labeled_test_data_log.csv",
#     transform=transforms.Compose([transforms.ToTensor()]),
# )
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# # Test the batch
# test_on_batch(test_loader)
