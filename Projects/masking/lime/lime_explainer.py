import json
import torch
from torchvision import models, transforms
from PIL import Image as PilImage
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np

# Function to load an image
def load_image(image_path):
    return PilImage.open(image_path).convert('RGB')

# Function to load ImageNet class names
def load_class_names(json_path):
    with open(json_path, 'r') as read_file:
        class_idx = json.load(read_file)
    return [class_idx[str(k)][1] for k in range(len(class_idx))]

# Define the model (pre-trained ResNet-50 in this example)
def create_model():
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model.eval()
    return model

# Function to transform the image
def transform_image(image, transform):
    return transform(image).unsqueeze(0)

# Prediction function for LIME
def batch_predict(images, model, transform):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = torch.stack([transform(image) for image in images]).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.cpu().numpy()

# Main function to perform LIME explanation
def main():
    image_path = "../carla_images/CARLA_Camera_screenshot_07.07.2024.png"
    json_path = "../data/imagenet_class_index.json"

    # Load the image and class names
    img = load_image(image_path)
    idx2label = load_class_names(json_path)
    model = create_model()
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(img), 
                                             batch_predict, 
                                             top_labels=5, 
                                             hide_color=0, 
                                             num_samples=1000,
                                             model=model, 
                                             transform=transform)

    # Display explanation for the top label
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_label, positive_only=False, num_features=10, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)
    
    plt.imshow(img_boundry)
    plt.title(f"Class: {idx2label[top_label]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
