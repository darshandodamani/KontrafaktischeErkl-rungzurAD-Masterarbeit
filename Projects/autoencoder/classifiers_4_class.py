# Projects/autoencoder/classifier.py
import torch.nn as nn


# Define a simple feedforward classifier
class ClassifierModel(nn.Module):
    def __init__(
        self, input_size=128, hidden_size=128, output_size=4, dropout_rate=0.5
    ):
        super(ClassifierModel, self).__init__()  # Initialize the nn.Module class
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization for Layer 1

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)  # Batch normalization for Layer 2

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)  # Batch normalization for Layer 3

        self.fc4 = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
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
