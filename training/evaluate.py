import os

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from networks.policy import GoPolicyNetwork
from training.policy_train import parse_file

CHECKPOINT_DIR = "./checkpoints"


def policy_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check = torch.load(f"{CHECKPOINT_DIR}/checkpoint_2_400.pth")

    board_states, moves, __, __ = parse_file("./dataset/val/160000_160200.npz", [], [])

    model = GoPolicyNetwork().to(device)
    model.load_state_dict(check["model_state_dict"])

    criterion = CrossEntropyLoss()

    correct_count = 0
    total_loss = 0
    total_epoch = 0
    for board, move in zip(board_states, moves):
        board, move = board.to(device), move.to(device)
        output = model(board)
        total_loss += float(criterion(output, move.long()))

        # Check accuracy
        index_of_max = output.argmax(axis=1)
        match = index_of_max == move
        correct_count += match.sum()
        total_epoch += 1

    avg_loss = total_loss / total_epoch
    print(f"Loss: {avg_loss}, Accuracy: {correct_count / (len(board_states) * 16)}")
