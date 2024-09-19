# 🚗 Enhancing Autonomous Driving Explainability: A Counterfactual Approach Using deep generative model, Masking, and inpainting techniques

This thesis focuses on improving the **explainability** of autonomous driving decisions using **counterfactual analysis**. We employ a **Variational Autoencoder (VAE)** for dimensionality reduction and **Local Interpretable Model-agnostic Explanations (LIME)** to highlight important latent space features affecting the car’s STOP or GO decisions.

---

## 🧠 Thesis Overview

Modern AI-driven autonomous driving systems often act as **black boxes**, making it difficult to understand the reasoning behind their decisions. This thesis tackles the black-box issue by providing interpretable **counterfactual explanations**. The main goals are:
  
- Train a **VAE** to encode images from the **CARLA simulator** into a **latent space** representation.
- Use **LIME** to identify critical features in the latent space that influence the car's decisions.
- Modify the latent space by **masking** important features, then reconstruct the image with the VAE decoder to generate **counterfactual explanations**.
- Compare predictions between the **original** and **counterfactual** images (STOP or GO) to provide **interpretable insights**.

---

## ✨ Key Features

- **Dataset**: Collected using the CARLA driving simulator (Town 7) with metadata like throttle, brake, and steering values.
- **VAE**: Learns a latent space from the images, enabling effective image reconstruction.
- **Classifier**: Learns to predict **STOP** or **GO** decisions from the latent vectors.
- **LIME**: Explains the classifier's decision by highlighting critical latent space features.
- **Counterfactual Explanations**: Shows what changes in the input would lead to a different decision by the model.

---

## 📊 System Workflow Diagram

![System Workflow](visualization_output/methodology.png)  
*This diagram shows how the data flows through the VAE, classifier, and LIME to generate counterfactual explanations.*

---

## 📁 Dataset Collection

The dataset used for this thesis was collected by driving a car in **CARLA Town 7**, a simulated driving environment. Each session captures the following:

- **Image**: Snapshots of the driving environment.
- **Steering**: Steering angle of the car.
- **Throttle**: Throttle input value.
- **Brake**: Brake input value.

### 🔧 How to Collect Data

1. Activate the virtual environment using **venv**:
   ```bash
   source venv/bin/activate
