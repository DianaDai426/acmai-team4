import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x1 images.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 7, padding=3)
        self.pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(10, 20, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(20, 35, 5, padding=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(196, 300)
        self.fc2 = nn.Linear(300, 1000)
        self.fc3 = nn.Linear(1000, 5004)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.sigmoid(x)
        return x
