import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    """
    A Convolutional Neural Network (CNN) designed for classifying 28x28 grayscale images
    like the MNIST dataset. It uses two Conv/Pool blocks followed by two fully connected layers.
    """
    def __init__(self):
        """
        Initializes the layers (components) of the CNN architecture.
        """
        super(Cnn, self).__init__()
        
        # --- Feature Extraction Layers ---
        
        # Conv1: Input (1 channel/grayscale) -> Output (10 feature maps)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        
        # Max Pooling Layer: Reduces spatial dimensions by half (Kernel/Stride=2).
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv2: Input (10 feature maps) -> Output (20 complex feature maps)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        # --- Classification Layers (Fully Connected) ---
        
        # FC1: Input size is 4*4*20=320 after pooling -> 50 neurons
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        
        # FC2: Output layer (50 neurons) -> 10 classes (digits 0-9)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        """
        Defines the computational flow of the data through the network.
        
        Args:
            x (torch.Tensor): Input tensor (Batch, 1, 28, 28).
            
        Returns:
            torch.Tensor: Output logits (Batch, 10).
        """
        # --- BLOCK 1 ---
        # Output: 24x24x10 (28 - 5 + 1 = 24)
        x = self.conv1(x) 
        x = F.relu(x)
        x = self.pool(x) # Output: 12x12x10 (Halved by 2x2 pooling)

        # --- BLOCK 2 ---
        # Output: 8x8x20 (12 - 5 + 1 = 8)
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.pool(x) # Output: 4x4x20 (Halved by 2x2 pooling)
        
        # --- FLATTENING (Bridge to FC Layers) ---
        # Reshape to a 1D vector of size 320 for the linear layer
        x = x.view(x.size(0), -1) 
        
        # --- CLASSIFICATION HEAD ---
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) # Final output (Logits)
        
        return x