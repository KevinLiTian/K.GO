import os

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss

from networks.policy import GoPolicyNetwork

NUM_EPOCH = 200
BATCH_SIZE = 16
LR = 0.003

DATA_FILES = [f"{i}_{i+200}.npz" for i in range(0, 160000, 200)]


def parse_file(file_path, remaining_boards, remaining_moves):
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
        moves_batch.append(torch.from_numpy(moves[start_idx:end_idx]))

    if board_states.shape[0] % BATCH_SIZE != 0:
        remaining_boards = board_states[num_batches * BATCH_SIZE :]
        remaining_moves = moves[num_batches * BATCH_SIZE :]
    else:
        remaining_boards, remaining_moves = [], []

    return board_states_batch, moves_batch, remaining_boards, remaining_moves


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = GoPolicyNetwork().to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=80000000, gamma=0.5)

    remaining_boards, remaining_moves = [], []
    for epoch in range(NUM_EPOCH):
        file_count = 0
        for file in DATA_FILES:
            (
                board_states_batch,
                moves_batch,
                remaining_boards,
                remaining_moves,
            ) = parse_file(
                os.path.join("./dataset", file), remaining_boards, remaining_moves
            )

            for board_states, moves in zip(board_states_batch, moves_batch):
                board_states, moves = board_states.to(device), moves.to(device)
                output = model(board_states)
                loss = criterion(output, moves.long())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            file_count += 1
            print(f"Files finished: {file_count}/{len(DATA_FILES)}, Loss: {loss}")

            if file % 100 == 0:
                checkpoint = {
                    "file_count": file_count,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                }
                torch.save(
                    checkpoint, f"./checkpoints/checkpoint_{epoch}_{file_count}.pth"
                )

        print(f"Epoch {epoch + 1}/{NUM_EPOCH}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, f"./checkpoints/checkpoint_{epoch}.pth")
