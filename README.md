# Enhancing Autonomous Driving Explainability: A Counterfactual Approach Using Variational Auto Encoder and Local Interpretable Model-agnostic Explanations-Based Masking

This project aims to enhance the explainability of **autonomous driving decisions** using a **counterfactual approach**. By leveraging a **Variational Autoencoder (VAE)** and **Local Interpretable Model-agnostic Explanations (LIME)**, the system generates counterfactual explanations, revealing how certain features in the driving environment affect the car’s decisions to either **STOP** or **GO**.

## Project Overview

In autonomous driving, decisions such as when to stop or continue are made by AI models, which often act as black-box systems. This project addresses the **black-box issue** by providing **explainability** through counterfactuals. The core approach involves:

- Training a **VAE** to encode images from the driving environment (CARLA simulator) into a **latent space** representation.
- Using **LIME** to identify critical features in the latent space that influence the autonomous driving decisions.
- Modifying or **masking** the important latent features and reconstructing the image using the **VAE decoder** to generate **counterfactual explanations**.
- Evaluating the difference in the decision (STOP/GO) between the original and counterfactual images to provide interpretable insights.

### Key Features
- **Dataset**: Collected from the CARLA driving simulator (Town 7) with throttle, brake, and steering data.
- **VAE Training**: Compresses images into a latent space and reconstructs them.
- **Classifier**: Predicts whether the car should STOP or GO based on latent vectors.
- **LIME for Explainability**: Highlights important features in the latent space and generates counterfactuals by modifying them.
- **Counterfactual Explanation**: Explains why the model decided to STOP or GO and what would have changed the decision.

---

### Image: System Workflow Diagram

![System Workflow](visualization_output/flow_chat.png)  
*This diagram explains the process of how the data flows through the VAE, classifier, and LIME to generate counterfactual explanations.*

---

## Dataset Collection

The dataset used for this project is collected by driving a car in the **CARLA Town 7 environment**. Each driving session collects the following data for each image frame:
- **Image**: A snapshot of the driving environment.
- **Steering**: Steering angle of the car.
- **Throttle**: Throttle value (speed input).
- **Brake**: Brake input value.

### How to Collect Data

1. Activate the virtual environment using **venv**:
   ```bash
   source venv/bin/activate
