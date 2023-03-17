import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_features=256)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = residual + x
        x = F.relu(x)
        return x


class ResidualTower(nn.Module):
    def __init__(self, num_residual_blocks, in_channels):
        super(ResidualTower, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(num_features=256)

        self.residual_blocks = nn.ModuleList(
            [ResidualBlock() for _ in range(num_residual_blocks)]
        )

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for block in self.residual_blocks:
            x = block(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self):
        super(PolicyHead, self).__init__()

        self.conv = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(num_features=2)
        self.fc = nn.Linear(in_features=19 * 19 * 2, out_features=19 * 19)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, 19 * 19 * 2)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    def __init__(self):
        super(ValueHead, self).__init__()

        self.conv = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(num_features=1)
        self.fc1 = nn.Linear(in_features=19 * 19, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, 19 * 19)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class DualResnet(nn.Module):
    """
    Proposed in Deep Mind's 2017 paper "Mastering the Game of Go
    without Human Knowledge". Consisting of a convolutional layer
    followed by a residual tower of 19/39 blocks, then seperated into
    two heads, policy and value.
    """

    def __init__(self, num_residual_blocks=19, in_channels=17):
        super(DualResnet, self).__init__()

        self.residual_tower = ResidualTower(num_residual_blocks, in_channels)
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def forward(self, x):
        x = self.residual_tower(x)
        policy_output = self.policy_head(x)
        value_output = self.value_head(x)
        return policy_output, value_output
