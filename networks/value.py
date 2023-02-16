import torch.nn as nn
import torch.nn.functional as F


class GoValueNetwork(nn.Module):
    def __init__(self):
        super(GoValueNetwork, self).__init__()

        # Convolutional layers
        self.conv = nn.ModuleList()

        # First conv layer
        self.conv.append(
            nn.Conv2d(49, 192, kernel_size=(5, 5), stride=1, padding=(2, 2))
        )

        # Conv layers 2 - 13
        for __ in range(2, 14):
            self.conv.append(
                nn.Conv2d(192, 192, kernel_size=(3, 3), stride=1, padding=(1, 1))
            )

        # Final conv layer
        self.conv.append(nn.Conv2d(192, 1, kernel_size=(1, 1), stride=1))

        # Fully connected layer
        self.fc1 = nn.Linear(19 * 19, 256, bias=True)
        self.fc2 = nn.Linear(256, 1, bias=True)

    def forward(self, x):
        # Pass the input through the convolutional layers
        for layer in self.conv:
            x = F.relu(layer(x))

        # Flatten
        x = x.view(-1, 19 * 19)

        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))

        return x
