import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv192(nn.Module):
    """
    Proposed in Deep Mind's 2016 paper "Mastering the Game of Go with Deep Neural Networks and Tree Search"
    Used in the version of Alpha Go (Alpha Go Fan) that defeated the 3 times reigning European champion Fan Hui
    """

    def __init__(self):
        super(Conv192, self).__init__()

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

        # Softmax activation
        x = F.softmax(x, dim=1)

        return x


class Conv256(nn.Module):
    """
    Not officially published but mentioned in both Deep Mind's 2016 paper
    "Mastering the Game of Go with Deep Neural Networks and Tree Search",
    and their 2017 paper "Mastering the Game of Go without Human Knowledge",
    it is used in the version of Alpha Go (Alpha Go Lee) that defeated Lee Sedol,
    18 times world champion
    """

    def __init__(self):
        super(Conv256, self).__init__()

        # Convolutional layers
        self.conv = nn.ModuleList()

        # First conv layer
        self.conv.append(
            nn.Conv2d(48, 256, kernel_size=(5, 5), stride=1, padding=(2, 2))
        )

        # Conv layers 2 - 12
        for __ in range(2, 13):
            self.conv.append(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=(1, 1))
            )

        # Final conv layer
        self.conv.append(nn.Conv2d(256, 1, kernel_size=(1, 1), stride=1))

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

        # Softmax activation
        x = F.softmax(x, dim=1)

        return x
