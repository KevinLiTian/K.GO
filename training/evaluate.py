import os

import torch
import torch.nn.functional as F
import numpy as np

from networks.policy import GoPolicyNetwork

# Data directory
DATA_FILES = [f"{i}_{i+200}.npz" for i in range(0, 160000, 200)]


def policy_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check = torch.load("./checkpoints/checkpoint_0.pth")

    data = np.load(os.path.join("./dataset", DATA_FILES[0]))
    board_states = torch.from_numpy(data["board_states"][:1000, :48]).to(device)
    moves = torch.from_numpy(data["moves"][:1000]).long().to(device)

    model = GoPolicyNetwork().to(device)
    model.load_state_dict(check["model_state_dict"])

    output = model(board_states)

    # Check accuracy
    # Find the indices of the maximum values along the second dimension
    output = F.softmax(output, dim=1)
    _, preds = torch.max(output, dim=1)

    # Check if the maximum values match the corresponding labels
    matches = torch.eq(preds, moves)

    # Count the number of matches
    num_matches = torch.sum(matches)
    print(num_matches / len(board_states))
