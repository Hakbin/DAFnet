import torch
from torch import nn


class LeNet5(nn.Module):
    """
    Class for a Lenet5 classifier.
    """
    def __init__(self):
        """
        Class initializer.
        """
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.max_pool1 = nn.MaxPool2d((2, 2), 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.max_pool2 = nn.MaxPool2d((2, 2), 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Forward propagation.
        """
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.max_pool1(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.max_pool2(out)
        out = self.conv3(out)
        out = torch.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        return out

