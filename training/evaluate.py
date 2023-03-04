import torch
from torch.nn import NLLLoss

from networks.policy import GoPolicyNetwork
from training.policy_train import parse_file

CHECKPOINT_DIR = "./checkpoints"


def policy_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check = torch.load(f"{CHECKPOINT_DIR}/checkpoint_0_50.pth")

    board_states, moves, __, __ = parse_file("./dataset/val/160000_160200.npz", [], [])

    model = GoPolicyNetwork().to(device)
    model.load_state_dict(check["model_state_dict"])

    criterion = NLLLoss()

    correct_count = 0
    total_loss = 0
    total_epoch = 0
    for board, move in zip(board_states, moves):
        board, move = board.to(device), move.to(device)
        output = model(board)
        total_loss += float(criterion(torch.log(output), move))

        # Check accuracy
        state_max = output.argmax(axis=1)
        match = state_max == move
        correct_count += match.sum()
        total_epoch += 1

    avg_loss = total_loss / total_epoch
    print(f"Loss: {avg_loss}, Accuracy: {correct_count / (len(board_states) * 16)}")
