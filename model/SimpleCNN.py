import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.nn = nn.Sequential(nn.Linear(8 * 32 * 32, 512), nn.ReLU(), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.pooling(torch.relu(self.conv1(x)))
        x = self.pooling(torch.relu(self.conv2(x)))
        x = self.pooling(torch.relu(self.conv3(x)))
        x = x.view(-1, 8 * 32 * 32)
        x = self.nn(x)
        return x
