import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    """Tiny CNN for MNIST."""
    def __init__(self):
        super(Cnn, self).__init__()

        # conv stack
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        # linear head
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        # block 1
        x = self.conv1(x) 
        x = F.relu(x)
        x = self.pool(x)

        # block 2
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.pool(x)
        
        # flatten and classify
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
