import torch
import torch.nn as nn
import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x1 images.
    """

    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True, force_reload=True)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.resnet.eval()

        # self.conv1 = nn.Conv2d(1, 10, 7, padding=3)
        # self.pool1 = nn.MaxPool2d(4, 4)
        # self.conv2 = nn.Conv2d(10, 20, 5, padding=2)
        # self.pool2 = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(20, 35, 5, padding=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6860, 6000)
        # self.fc2 = nn.Linear(2000, 4000)
        self.fc2 = nn.Linear(6000, 5005) #should be 5005 with new whale and small num of whales
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad:
             x = self.resnet(x)
        
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool2(F.relu(self.conv3(x)))

        # x = self.flatten(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # # x = F.relu(self.fc3(x))
        # x = self.sigmoid(x)
        return x
