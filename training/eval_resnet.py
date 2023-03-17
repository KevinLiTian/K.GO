import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from networks.resnet import DualResnet
from training.utils import parse_file

TRAIN_SET = [f"./zero_dataset/{i}_{i+400}.npz" for i in range(0, 800, 400)]


def resnet_eval(model_path):
    print(f"Start evaluating {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = DualResnet().to(device)
    check = torch.load(model_path)
    model.load_state_dict(check["model_state_dict"])
    model.eval()

    policy_criterion = CrossEntropyLoss()
    value_criterion = MSELoss()

    train_acc, train_policy_loss, train_value_loss = 0, 0, 0

    total_epoch = 0
    total_boards = 0
    remaining_boards, remaining_moves, remaining_results = [], [], []
    for file in TRAIN_SET:
        (
            board_states,
            moves,
            results,
            remaining_boards,
            remaining_moves,
            remaining_results,
        ) = parse_file(
            file,
            remaining_boards,
            remaining_moves,
            remaining_results,
            batch_size=64,
            features=17,
        )
        total_boards += len(board_states) * 64

        for board, move, result in zip(board_states, moves, results):
            board, move, result = board.to(device), move.to(device), result.to(device)
            policy_out, value_out = model(board)

            train_policy_loss += policy_criterion(policy_out, move).item()
            train_value_loss += value_criterion(value_out, result).item()

            # Check accuracy
            policy_out = F.softmax(policy_out, dim=1)
            state_max = policy_out.argmax(axis=1)
            match = state_max == move
            train_acc += match.sum()
            total_epoch += 1

    train_policy_loss = train_policy_loss / total_epoch
    train_value_loss = train_value_loss / total_epoch
    train_acc = train_acc / total_boards
    print(
        f"Policy loss: {train_policy_loss}, Value loss: {train_value_loss}, Accuracy: {train_acc}"
    )
