import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from vae import VariationalAutoencoder, CustomImageDatasetWithLabels
from classifier import ClassifierModel  # Import the classifier model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the trained VAE model
vae_model = VariationalAutoencoder(latent_dims=256).to(device)
vae_model.load_state_dict(torch.load('model/var_autoencoder.pth', map_location=device))
vae_model.eval()

# Instantiate the classifier model
classifier = ClassifierModel(input_size=256, hidden_size=128, output_size=2).to(device)

# Load the classifier state dictionary
classifier.load_state_dict(torch.load('model/classifier.pth', map_location=device))
classifier.eval()  # Set the classifier to evaluation mode

# Load the test dataset
data_dir = 'dataset/town7_dataset/test/'
csv_file = 'dataset/town7_dataset/test/labeled_test_data_log.csv'
data_transforms = transforms.Compose([transforms.ToTensor()])

test_dataset = CustomImageDatasetWithLabels(img_dir=data_dir, csv_file=csv_file, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Collect predictions and true labels
all_preds = []
all_labels = []

for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)

    # Encode images into latent space
    latent_vectors = vae_model.encoder(images)

    # Get classifier predictions on latent space
    with torch.no_grad():
        outputs = classifier(latent_vectors)
        preds = torch.argmax(outputs, dim=1)  # Get the predicted class (0 for STOP, 1 for GO)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Compute Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Visualize the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['STOP', 'GO'], yticklabels=['STOP', 'GO'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate TP, FP, TN, FN from the Confusion Matrix
tn, fp, fn, tp = conf_matrix.ravel()  # Extract TN, FP, FN, TP from the confusion matrix

# Calculate other performance metrics (optional)
accuracy = accuracy_score(all_labels, all_preds)
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
