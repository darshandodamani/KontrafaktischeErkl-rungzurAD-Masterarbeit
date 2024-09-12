# Projects/autoencoder/classifier.py

import torch  # noqa: F401
import torch.nn as nn


# Define a simple feedforward classifier
class ClassifierModel(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=2):
        super(ClassifierModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(
            x
        )  # No activation here, we will use CrossEntropyLoss that expects raw logits
        return x
