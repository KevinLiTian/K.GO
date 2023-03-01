import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss

from networks.policy import GoPolicyNetwork

# Hyperparameters
NUM_EPOCH = 200
BATCH_SIZE = 16
LR = 0.003

# Data directory
DATA_FILES = [f"{i}_{i+200}.npz" for i in range(0, 160000, 200)]

# Checkpoint directory
CHECKPOINT_DIR = "./checkpoints"


def parse_file(file_path, remaining_boards, remaining_moves):
    """Parse a .npz file into batches of tensors"""
    data = np.load(file_path)
    board_states = data["board_states"]
    moves = data["moves"]

    if len(remaining_boards) != 0:
        board_states = np.concatenate([board_states, remaining_boards], axis=0)
        moves = np.concatenate([moves, remaining_moves], axis=0)

    perm = np.random.permutation(board_states.shape[0])
    board_states, moves = board_states[perm], moves[perm]

    board_states_batch = []
    moves_batch = []

    num_batches = board_states.shape[0] // BATCH_SIZE
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        board_states_batch.append(
            torch.from_numpy(board_states[start_idx:end_idx, :48])
        )
        moves_batch.append(torch.from_numpy(moves[start_idx:end_idx]).long())

    if board_states.shape[0] % BATCH_SIZE != 0:
        remaining_boards = board_states[num_batches * BATCH_SIZE :]
        remaining_moves = moves[num_batches * BATCH_SIZE :]
    else:
        remaining_boards, remaining_moves = [], []

    return board_states_batch, moves_batch, remaining_boards, remaining_moves


def train(resume=False):
    # Setup device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if resume:
        checkpoint_files = os.listdir(CHECKPOINT_DIR)
        latest_checkpoint = max(
            checkpoint_files,
            key=lambda x: os.path.getctime(os.path.join(CHECKPOINT_DIR, x)),
        )
        print(f"Resume from {latest_checkpoint}")
        checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, latest_checkpoint))

        model = GoPolicyNetwork().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=LR)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler = StepLR(optimizer, step_size=80000000, gamma=0.5)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "file_count" in checkpoint:
            file_count = checkpoint["file_count"]
            cur_epoch = checkpoint["epoch"]
        else:
            file_count = 0
            cur_epoch = checkpoint["epoch"] + 1

    else:
        model = GoPolicyNetwork().to(device)
        criterion = CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=80000000, gamma=0.5)
        file_count = 0
        cur_epoch = 0

    # Training utilities
    remaining_boards, remaining_moves = [], []

    # Main train loop
    print(f"Start epoch {cur_epoch}, file count {file_count}")
    for epoch in range(cur_epoch, NUM_EPOCH):
        if file_count == len(DATA_FILES):
            file_count = 0
        for file in DATA_FILES[file_count:]:
            (
                board_states_batch,
                moves_batch,
                remaining_boards,
                remaining_moves,
            ) = parse_file(
                os.path.join("./dataset", file), remaining_boards, remaining_moves
            )

            for board_states, moves in zip(board_states_batch, moves_batch):
                optimizer.zero_grad()
                board_states, moves = board_states.to(device), moves.to(device)

                outputs = model(board_states)
                loss = criterion(outputs, moves)

                loss.backward()
                optimizer.step()
                scheduler.step()

            file_count += 1
            print(f"Files finished: {file_count}/{len(DATA_FILES)}, Loss: {loss}")

            if file_count % 100 == 0:
                checkpoint = {
                    "file_count": file_count,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(
                    checkpoint, f"{CHECKPOINT_DIR}/checkpoint_{epoch}_{file_count}.pth"
                )

        print(f"Epoch {epoch + 1}/{NUM_EPOCH} finished")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(checkpoint, f"{CHECKPOINT_DIR}/checkpoint_{epoch}.pth")
