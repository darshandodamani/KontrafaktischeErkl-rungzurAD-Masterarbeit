# A Counterfactual Explanation Approach Using Deep Generative Models

This repository contains the implementation of the research presented in my thesis, which focuses on improving the **explainability** of autonomous driving decisions through **counterfactual explanations**. By leveraging **Deep Generative Models**, this work aims to provide clear insights into the decision-making processes of autonomous vehicles, especially in critical driving scenarios.

# Overview
Autonomous driving systems make complex decisions based on various inputs from their environment. While these decisions are often accurate, there is a growing need to **understand why** a particular action was taken or why an alternative action wasn't chosen. To address this, we use **counterfactual explanations**, which help in determining what minimal changes in the input would alter the model's decision.

# About the Project
This work aims to develop a solution for autonomous driving that can explain the decision-making process using counterfactual explanations. The project involves the following components:

- **CARLA Environment Setup**: Setting up the CARLA simulator to create a realistic driving environment.
- **Variational Autoencoder (VAE)**: A deep generative model used to encode high-dimensional driving data into a latent space, enabling counterfactual analysis.
- **Counterfactual Explanation Generation**: Using feature masking, inpainting, and reconstruction to generate alternative scenarios and understand the model's decisions.

We have used **CARLA (version 0.9.15)** as our urban driving simulator to collect datasets for training and generating counterfactual explanations.

# Prerequisites
- **CARLA Version**: `0.9.15` (Urban Simulator)
- **Python Version**: `3.7` is recommended for compatibility.
- **Additional Maps**: We have focused on **Town07** and **Town06**. Please download the additional maps alongside the CARLA server and copy them into the main CARLA directory to ensure seamless operation.

# Setting Up the Project
1. **Clone the Repository**: Clone this repository to get started.
2. **Create Virtual Environment**: We recommend creating a Python virtual environment for this project:
   ```bash
   python -m venv venv
   ```
   Activate the virtual environment:
   ```bash
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install Dependencies**: Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download CARLA Server**: Download **CARLA server (0.9.15)** and **additional maps** from the [official CARLA repository](https://github.com/carla-simulator/carla/releases).
5. **Start CARLA Server**: Make sure to start the CARLA server before running the client.
   ```bash
   ./CarlaUE4.sh
   ```

# Dataset Collection
Currently, the dataset is collected from **Town07** and **Town06** environments in CARLA. This dataset will be used for training the deep generative model and generating counterfactual explanations.

### Running Dataset Collection
To collect the dataset, use the following command:
```bash
python collect_images.py --output_dir dataset/town7_dataset --town_name Town07 --image_size 160 80
```
- **Parameters**:
  - `--output_dir`: Directory where the collected dataset will be stored.
  - `--town_name`: Specify the CARLA town (e.g., Town07, Town06).
  - `--image_size`: Specify the width and height of the collected images.

# Labeling and Splitting the Dataset
Once the dataset is collected, the next steps are **labeling** and **splitting** the dataset for training and testing purposes.

### Labeling the Dataset
The dataset is labeled based on the throttle and brake values collected from the vehicle in CARLA. Images are labeled as `STOP` or `GO` based on thresholds for brake and throttle:
- **STOP**: If `brake > 0.1` or `throttle < 0.2`.
- **GO**: Otherwise.

To label the dataset, use the following command:
```bash
python label_dataset.py --data_path <path_to_dataset>
```
- Example:
  ```bash
  python label_dataset.py --data_path ../dataset/town7_dataset
  ```

### Splitting the Dataset
To prepare the dataset for training, it needs to be split into **training** and **testing** subsets while maintaining balanced classes (`STOP` and `GO`). The splitting can be done using the command below:
```bash
python split_dataset.py --data_path <path_to_dataset> --train_ratio 0.8
```
- Example:
  ```bash
  python split_dataset.py --data_path ../dataset/town7_dataset --train_ratio 0.8
  ```
- **Parameters**:
  - `--train_ratio`: Ratio of the data to use for training (default: 0.8).

# Directory Structure
After collecting, labeling, and splitting the dataset, your project directory should look like this:
```
dataset/
  town7_dataset/
    labeled_data_log.csv
    train/
      train_data_log.csv
      ... (train images)
    test/
      test_data_log.csv
      ... (test images)
plots/
  dataset_images/
    label_distribution.png
    dataset_split.png
```

# Built With
- **Python** - Programming language
- **PyTorch** - Open source machine learning framework
- **CARLA** - An urban driving simulator
- **TensorBoard** - Visualization toolkit