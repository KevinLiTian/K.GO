import torch
import torch.nn as nn
import torch.nn.functional as F


class GoRolloutNetwork(nn.Module):
    def __init__(self):
        super(GoRolloutNetwork, self).__init__()
        self.conv = nn.Conv2d(4, 1, kernel_size=(1, 1), stride=1)

        # Bias
        self.bias = nn.Parameter(torch.zeros(361))

    def forward(self, x):
        x = self.conv(x)

        # Flatten
        x = x.view(-1, 19 * 19)

        # Bias
        x = x + self.bias(x)

        # Softmax activation
        x = F.softmax(x, dim=1)
        return x
