import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 architecture.
    We use nn.ModuleList to allow executing layers one by one 
    simulating the distributed split.
    """
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # We break the model down into granular "layers" suitable for offloading
        self.layers = nn.ModuleList([
            # Layer 0: Conv1
            nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            # Layer 1: Conv2
            nn.Sequential(
                nn.Conv2d(6, 16, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ),
            # Layer 2: Flatten (Helper layer for consistent sequential execution)
            nn.Flatten(),
            # Layer 3: FC1
            nn.Sequential(
                nn.Linear(16 * 5 * 5, 120),
                nn.ReLU()
            ),
            # Layer 4: FC2
            nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU()
            ),
            # Layer 5: Output
            nn.Linear(84, 10) 
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
