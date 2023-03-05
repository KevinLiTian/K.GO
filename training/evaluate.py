import pandas as pd

import torch
from torch.nn import NLLLoss

from networks.policy import GoPolicyNetwork
from training.policy_train import parse_file

CHECKPOINT_DIR = "./checkpoints"
GRAPH_FILE = "./training/train.csv"


def policy_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check = torch.load(f"{CHECKPOINT_DIR}/checkpoint_0.pth")

    train_board_states, train_moves, __, __ = parse_file("./dataset/0_200.npz", [], [])
    val_board_states, val_moves, __, __ = parse_file(
        "./dataset/val/160000_160200.npz", [], []
    )

    model = GoPolicyNetwork().to(device)
    model.load_state_dict(check["model_state_dict"])

    criterion = NLLLoss()

    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0
    total_epoch = 0

    for board, move in zip(train_board_states, train_moves):
        board, move = board.to(device), move.to(device)
        output = model(board)
        train_loss += float(criterion(torch.log(output), move))

        # Check accuracy
        state_max = output.argmax(axis=1)
        match = state_max == move
        train_acc += match.sum()
        total_epoch += 1

    train_loss = train_loss / total_epoch
    train_acc = train_acc / (len(train_board_states) * 16)
    print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")

    total_epoch = 0
    for board, move in zip(val_board_states, val_moves):
        board, move = board.to(device), move.to(device)
        output = model(board)
        val_loss += float(criterion(torch.log(output), move))

        # Check accuracy
        state_max = output.argmax(axis=1)
        match = state_max == move
        val_acc += match.sum()
        total_epoch += 1

    val_loss = val_loss / total_epoch
    val_acc = val_acc / (len(val_board_states) * 16)
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

    # Read the existing CSV file into a DataFrame
    df = pd.read_csv(GRAPH_FILE)

    # Assuming your existing DataFrame is called 'df'
    new_row = pd.DataFrame(
        {
            "Steps (K)": [df["Steps (K)"].iloc[-1] + 100],
            "Training loss": train_loss,
            "Training accuracy": float(train_acc),
            "Validation loss": val_loss,
            "Validation accuracy": float(val_acc),
        }
    )

    # Combine existing DataFrame with new row using concat()
    df = pd.concat([df, new_row], ignore_index=True)

    # Write the updated DataFrame back to the CSV file
    df.to_csv(GRAPH_FILE, index=False)
