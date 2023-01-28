import torch.nn as nn
import torch.nn.functional as F


class GoPolicyNetwork(nn.Module):
    def __init__(self):
        super(GoPolicyNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            20, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)
        )
        self.conv2 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv3 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv4 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv5 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv6 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv7 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv8 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv9 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv10 = nn.Conv2d(
            192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(192 * 19 * 19, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 361, bias=True)

    def forward(self, x):
        # Pass the input through the convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))

        # Flatten the output
        x = x.view(-1, 192 * 19 * 19)

        # Pass the output through the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
