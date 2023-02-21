import torch
import torch.nn as nn
import torch.nn.functional as F


class GoPolicyNetwork(nn.Module):
    def __init__(self):
        super(GoPolicyNetwork, self).__init__()

        # Convolutional layers
        self.conv = nn.ModuleList()

        # First conv layer
        self.conv.append(
            nn.Conv2d(48, 192, kernel_size=(5, 5), stride=1, padding=(2, 2))
        )

        # Conv layers 2 - 12
        for __ in range(2, 13):
            self.conv.append(
                nn.Conv2d(192, 192, kernel_size=(3, 3), stride=1, padding=(1, 1))
            )

        # Final conv layer
        self.conv.append(nn.Conv2d(192, 1, kernel_size=(1, 1), stride=1))

        # Bias
        self.bias = nn.Parameter(torch.zeros(1, 361))

    def forward(self, x):
        # Pass the input through the convolutional layers
        for layer in self.conv:
            x = F.relu(layer(x))

        # Flatten
        x = x.view(-1, 19 * 19)

        # Bias
        x = x + self.bias

        return x
