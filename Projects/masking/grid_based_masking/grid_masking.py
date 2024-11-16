import os
import sys
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
import seaborn as sns
from PIL import Image
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse, peak_signal_noise_ratio as psnr
from sewar.full_ref import vifp, uqi

# Add Python path to include the directory where 'encoder.py' is located
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "autoencoder"))
)

from encoder import VariationalEncoder
from decoder import Decoder
from classifier import ClassifierModel as Classifier

# Paths to models
encoder_path = "model/epochs_500_latent_128_town_7/var_encoder_model.pth"
decoder_path = "model/epochs_500_latent_128_town_7/decoder_model.pth"
classifier_path = "model/epochs_500_latent_128_town_7/classifier_final.pth"

# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VariationalEncoder(latent_dims=128, num_epochs=100).to(device)
decoder = Decoder(latent_dims=128, num_epochs=100).to(device)
classifier = Classifier().to(device)

encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
classifier.load_state_dict(torch.load(classifier_path, map_location=device, weights_only=True))

encoder.eval()
decoder.eval()
classifier.eval()

# Manually select an image instead of choosing randomly
test_dir = 'dataset/town7_dataset/train/'
image_filename = 'town7_011990.png'
image_path = os.path.join(test_dir, image_filename)

# Load and preprocess the image
transform = transforms.Compose([
    transforms.Resize((80, 160)),
    transforms.ToTensor(),
])
image = Image.open(image_path).convert('RGB')
input_image = transform(image).unsqueeze(0).to(device)

# Step 1: Pass the input image through encoder, decoder, and classifier
latent_vector = encoder(input_image)[2]
reconstructed_input_image = decoder(latent_vector)
input_image_predicted_label = classifier(latent_vector)

# Convert the reconstructed image tensor to a PIL image for plotting
reconstructed_image_pil = transforms.ToPILImage()(reconstructed_input_image.squeeze(0).cpu())

# Print the predicted label
predicted_label = torch.argmax(input_image_predicted_label, dim=1).item()
predicted_class = "STOP" if predicted_label == 0 else "GO"
print(f'Input Image Predicted Label: {predicted_class}')

# apply grid based masking on the Inut Image
def grid_masking(input_image, grid_size=(10, 10), mask_value=0, pos=0):
      # Mask the first detected object
    masked_image = input_image.clone().squeeze().permute(1, 2, 0).cpu().numpy()
    
    # compute x and y coordinates for the grid field identified by pos
    x, y = grid_size
    x_pos = pos % x
    y_pos = pos // x
    x_size = masked_image.shape[1] // x
    y_size = masked_image.shape[0] // y
    x_start = x_pos * x_size
    y_start = y_pos * y_size
    x_end = x_start + x_size
    y_end = y_start + y_size
    masked_image[y_start:y_end, x_start:x_end] = mask_value
    


    # Convert masked image back to tensor
    return transforms.ToTensor()(masked_image).unsqueeze(0).to(device)

min_confidende = 0.0

for grid in [(10, 5), (4, 2)]:
    # iterate pver the grid positions
    for pos in range(grid[0] * grid[1]):
        grid_based_masked_image = grid_masking(input_image, grid_size=grid, pos=pos)

        # send the grid based masked image to the encoder, decoder, and classifier
        latent_vector_after_masking = encoder(grid_based_masked_image)[2]
        reconstructed_image_after_masking = decoder(latent_vector_after_masking)
        predicted_class_after_masking = classifier(latent_vector_after_masking)

        # Convert the reconstructed image tensor to a PIL image for plotting
        reconstructed_image_after_masking_pil = transforms.ToPILImage()(reconstructed_image_after_masking.squeeze(0).cpu())

        # Print the predicted label after masking
        predicted_label_after_masking = torch.argmax(predicted_class_after_masking, dim=1).item()
        
        # compute confidence of predicted class after masking
        confidence = F.softmax(predicted_class_after_masking, dim=1)[0]
    
        
        predicted_class_after_masking = "STOP" if predicted_label_after_masking == 0 else "GO"
        if predicted_class_after_masking != predicted_class:
            print(f'Counterfactual explanation generated at grid position {pos}')
            print(f'Grid Position: {pos}, Confidence: {confidence}')
            # check if confidence of counterfactual is high
            if confidence[0] > min_confidende or confidence[1] > min_confidende:
                # break out of outer loop
                break
            #break # stop if counterfactual explanation is generated
    if predicted_class_after_masking != predicted_class:
        break

print(f'Reconstructed Image after Masking Predicted Label: {predicted_class_after_masking}')

# plot the original image, reconstructed image, and reconstructed image after grid-based masking
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_image_pil)
plt.title("Reconstructed Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(reconstructed_image_after_masking_pil)
plt.title("Reconstructed Image after Masking")
plt.axis('off')

# save the plot
plt.savefig("plots/grid_based_masking_images/grid_based_masking.png")

# save the grid based masked image
grid_based_masked_image_pil = transforms.ToPILImage()(grid_based_masked_image.squeeze(0).cpu())
grid_based_masked_image_pil.save("plots/grid_based_masking_images/grid_based_masked_image.png")



if predicted_class_after_masking != predicted_class:
    print("Counterfactual explanation is generated.")
else:
    print("Counterfactual explanation is not generated.")
    
# Plot the original image, reconstructed image, and reconstructed image after grid-based masking
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
