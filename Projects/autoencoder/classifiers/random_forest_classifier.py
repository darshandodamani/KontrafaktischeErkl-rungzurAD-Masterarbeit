import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn import svm, ensemble
from vae import (
    LATENT_SPACE,
    NUM_EPOCHS,
    VariationalAutoencoder,
    CustomImageDatasetWithLabels,
)
from torchvision.transforms import transforms
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Hyperparameters
input_size = 128  # Latent space size from VAE (ensure it matches the latent space size of the trained VAE)
num_epochs = 20  # Random Forest is not epoch-based, but for consistency, we'll use 1 epoch for evaluation
learning_rate = 0.001  # Not used in Random Forest, but present for consistency
batch_size = 128  # Batch size for processing the dataset

# Load the trained VAE model
vae_model = VariationalAutoencoder(latent_dims=input_size, num_epochs=NUM_EPOCHS).to(device)  # Instantiate the VAE model with latent dimensions of 128
vae_model.load_state_dict(
    torch.load(
        "model/epochs_500_latent_128_town_7/var_autoencoder.pth",  # Load the trained VAE model state dictionary
        map_location=device,  # Map the model to the appropriate device (CPU or GPU)
        weights_only=True,
    )
)
print("VAE model loaded successfully!")
vae_model.eval()  # Set the model to evaluation mode

# Load the test dataset
data_dir = "dataset/town7_dataset/train/"  # Directory containing the test images
csv_file = "dataset/town7_dataset/train/labeled_train_data_log.csv"  # CSV file containing the labels for the test images
data_transforms = transforms.Compose([transforms.ToTensor()])  # Transform to convert images to tensors

# Create the test dataset and data loader
test_dataset = CustomImageDatasetWithLabels(
    img_dir=data_dir, csv_file=csv_file, transform=data_transforms
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Load the test dataset in batches of 128

# Extract latent vectors and corresponding labels from the test dataset
latent_vectors = []
labels = []

with torch.no_grad():  # Disable gradient computation for inference
    for images, label_batch, _ in test_loader:
        images = images.to(device)  # Move images to the appropriate device

        # Encode images into latent space using the VAE encoder
        _, _, latent_space = vae_model.encoder(images)
        latent_vectors.append(latent_space.cpu().numpy())  # Convert latent vectors to numpy and append
        labels.append(label_batch.cpu().numpy())  # Convert labels to numpy and append

latent_vectors = np.vstack(latent_vectors)  # Stack all latent vectors into a single numpy array
labels = np.hstack(labels)  # Stack all labels into a single numpy array

# Print total number of samples in the dataset
print(f"Total number of samples in the dataset: {len(latent_vectors)}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(latent_vectors, labels, test_size=0.2, random_state=42)  # Split data into 80% training and 20% test

# Print total number of samples in training and test set
print(f"Number of samples in training set: {len(X_train)}")
print(f"Number of samples in test set: {len(X_test)}")

# Train Random Forest Model
rf_model = ensemble.RandomForestClassifier(n_estimators=100, random_state=42)  # Instantiate the Random Forest model with 100 trees
rf_model.fit(X_train, y_train)  # Train the model on the training data
y_pred_rf = rf_model.predict(X_test)  # Predict labels for the test set

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred_rf)  # Calculate accuracy of the model
precision = precision_score(y_test, y_pred_rf, average='weighted')  # Calculate precision of the model
recall = recall_score(y_test, y_pred_rf, average='weighted')  # Calculate recall of the model
f1 = f1_score(y_test, y_pred_rf, average='weighted')  # Calculate F1 score of the model

# Print evaluation metrics
print(f"Random Forest - Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)  # Generate the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["STOP", "GO"], yticklabels=["STOP", "GO"])  # Plot the confusion matrix as a heatmap
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Confusion Matrix")
plt.savefig("plots/rf_confusion_matrix.png")  # Save the confusion matrix plot
plt.show()  # Display the plot

# ROC Curve and AUC
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1], pos_label=1)  # Calculate false positive rate and true positive rate for the ROC curve
roc_auc = auc(fpr, tpr)  # Calculate the area under the ROC curve

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")  # Plot the ROC curve
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Plot the random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) - Random Forest")
plt.legend(loc="lower right")
plt.savefig("plots/rf_roc_curve.png")  # Save the ROC curve plot
plt.show()  # Display the plot

# Confusion Matrix Details
print(f"Confusion Matrix:\n{conf_matrix}")  # Print the confusion matrix
print(f"True Positives (TP): {conf_matrix[1][1]}")  # True Positives: Correctly predicted "GO"
print(f"False Positives (FP): {conf_matrix[0][1]}")  # False Positives: Incorrectly predicted "GO"
print(f"True Negatives (TN): {conf_matrix[0][0]}")  # True Negatives: Correctly predicted "STOP"
print(f"False Negatives (FN): {conf_matrix[1][0]}")  # False Negatives: Incorrectly predicted "STOP"
print(f"Accuracy: {accuracy:.4f}")  # Print accuracy
print(f"Precision: {precision:.4f}")  # Print precision
print(f"Recall: {recall:.4f}")  # Print recall
print(f"F1 Score: {f1:.4f}")  # Print F1 score
print(f"ROC AUC: {roc_auc:.4f}")  # Print Area Under the ROC Curve (AUC)

# Save the trained Random Forest model
import joblib
joblib.dump(rf_model, "model/epochs_500_latent_128/rf_model.pkl")  # Save the trained Random Forest model

# Save all the evaluation metrics and plots in the plots/classifier_plots directory
os.makedirs("plots/classifier_plots/", exist_ok=True)  # Create the directory if it doesn't exist
plt.savefig("plots/classifier_plots/rf_confusion_matrix.png")  # Save the confusion matrix plot
plt.savefig("plots/classifier_plots/rf_roc_curve.png")  # Save the ROC curve plot
print("Random Forest model saved successfully!")  # Print success message

# Note: Random Forest does not have epoch-based training, so train_losses and train_accuracies are not applicable.
# Save the placeholder training plots and loss and accuracy in the classifier_training_loss_accuracy.png
plt.figure(figsize=(12, 5))  # Create a figure for plotting
plt.subplot(1, 2, 1)  # Create a subplot for training loss
plt.plot([], label="Training Loss")  # Placeholder for training loss
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1, 2, 2)  # Create a subplot for training accuracy
plt.plot([], label="Training Accuracy")  # Placeholder for training accuracy
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("classifier_training_loss_accuracy.png")  # Save the placeholder training loss and accuracy plot
