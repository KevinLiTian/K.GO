import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn import NLLLoss

from networks.policy import Conv192
from training.sl_policy import parse_file

CHECKPOINTS_DIR = "checkpoints"
GRAPH_FILE = "./training/train.csv"

TRAIN_SET = [f"./dataset/{i}_{i+200}.npz" for i in range(0, 1000, 200)]
VAL_SET = [f"./dataset/val/{i}_{i+200}.npz" for i in range(160000, 161000, 200)]


def policy_evaluate(model_path):
    print(f"Start evaluating {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = Conv192().to(device)
    check = torch.load(model_path)
    model.load_state_dict(check["model_state_dict"])

    criterion = NLLLoss()

    train_acc, train_loss = 0, 0
    val_acc, val_loss = 0, 0

    total_epoch = 0
    total_boards = 0
    remaining_boards, remaining_moves = [], []
    for file in TRAIN_SET:
        board_states, moves, remaining_boards, remaining_moves = parse_file(
            file, remaining_boards, remaining_moves
        )
        total_boards += len(board_states) * 16

        for board, move in zip(board_states, moves):
            board, move = board.to(device), move.to(device)
            output = model(board)
            train_loss += float(criterion(torch.log(output), move))

            # Check accuracy
            state_max = output.argmax(axis=1)
            match = state_max == move
            train_acc += match.sum()
            total_epoch += 1

    train_loss = train_loss / total_epoch
    train_acc = train_acc / total_boards
    print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc}")

    total_epoch = 0
    total_boards = 0
    remaining_boards, remaining_moves = [], []
    for file in VAL_SET:
        board_states, moves, remaining_boards, remaining_moves = parse_file(
            file, remaining_boards, remaining_moves
        )
        total_boards += len(board_states) * 16

        for board, move in zip(board_states, moves):
            board, move = board.to(device), move.to(device)
            output = model(board)
            val_loss += float(criterion(torch.log(output), move))

            # Check accuracy
            state_max = output.argmax(axis=1)
            match = state_max == move
            val_acc += match.sum()
            total_epoch += 1

    val_loss = val_loss / total_epoch
    val_acc = val_acc / total_boards
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


def policy_valuate_all():
    checkpoints = []

    # Loop over files in the directory
    for file_name in os.listdir(CHECKPOINTS_DIR):
        # Check if the file matches the pattern
        match = re.match(r"checkpoint_(\d+)_(\d+)", file_name)
        if match:
            epoch = int(match.group(1))
            batch = int(match.group(2))
            # Check if the batch number is in the range 50-750
            if 50 <= batch <= 750:
                checkpoints.append((epoch, batch, file_name))
        else:
            # Check if the file is a special checkpoint
            match = re.match(r"checkpoint_(\d+)", file_name)
            if match:
                epoch = int(match.group(1))
                checkpoints.append((epoch, None, file_name))

    # Sort the checkpoints in the desired order
    checkpoints.sort(key=lambda x: (x[0], x[1] or 751, x[2]))

    # Loop over the checkpoints
    for __, __, file_name in checkpoints:
        model_path = os.path.join(CHECKPOINTS_DIR, file_name)
        policy_evaluate(model_path)


def plot_policy_curves():
    # Load data from CSV file
    data = pd.read_csv("./training/train.csv")

    # Set the x-axis values
    steps = data["Steps (K)"]

    # Set the y-axis values for the first subplot
    train_loss = data["Training loss"]
    val_loss = data["Validation loss"]

    # Set the y-axis values for the second subplot
    train_acc = data["Training accuracy"]
    val_acc = data["Validation accuracy"]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the data in the first subplot
    ax1.plot(steps, train_loss, label="Training Loss")
    ax1.plot(steps, val_loss, label="Validation Loss")
    ax1.set_xlabel("Training Steps (10^3)")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.grid(True)
    ax1.legend()

    # Plot the data in the second subplot
    ax2.plot(steps, train_acc, label="Training Accuracy")
    ax2.plot(steps, val_acc, label="Validation Accuracy")
    ax2.set_xlabel("Training Steps (10^3)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.grid(True)
    ax2.legend()

    # Adjust the spacing between the subplots
    fig.subplots_adjust(hspace=0.5)

    # Show the plot
    plt.show()
