# Projects/autoencoder/classifier.py

import torch  # noqa: F401
import torch.nn as nn
from torchviz import make_dot
import torch
from graphviz import Digraph


# Define a simple feedforward classifier
class ClassifierModel(nn.Module):
    """
    Initializes the ClassifierModel.

    Args:
        input_size (int, optional): The size of the input features. Default is 128.
        hidden_size (int, optional): The size of the hidden layers. Default is 128.
        output_size (int, optional): The size of the output layer. Default is 2.
        dropout_rate (float, optional): The dropout rate for regularization. Default is 0.5.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        bn1 (nn.BatchNorm1d): Batch normalization for the first hidden layer.
        fc2 (nn.Linear): The second fully connected layer.
        bn2 (nn.BatchNorm1d): Batch normalization for the second hidden layer.
        fc3 (nn.Linear): The third fully connected layer.
        bn3 (nn.BatchNorm1d): Batch normalization for the third hidden layer.
        fc4 (nn.Linear): The output fully connected layer.
        leaky_relu (nn.LeakyReLU): Leaky ReLU activation function.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(
        self, input_size=128, hidden_size=128, output_size=2, dropout_rate=0.5
    ):
        super(ClassifierModel, self).__init__()  # Initialize the nn.Module class
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Add batch normalization

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.fc4 = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.

        The forward pass includes:
        - Fully connected layer followed by batch normalization, Leaky ReLU activation, and dropout.
        - Repeated for three layers.
        - Final fully connected layer without activation (raw logits).
        """
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Add dropout after activation

        # Layer 2
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        # Layer 3
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)

        # Output layer (raw logits, so no activation)
        x = self.fc4(x)

        return x
