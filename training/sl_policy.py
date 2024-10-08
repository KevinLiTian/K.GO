import os

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.nn import NLLLoss

from training.utils import parse_file

# Hyperparameters
NUM_EPOCH = 30
BATCH_SIZE = 16
LR = 0.003

# Data directory
DATA_FILES = [
    os.path.join("dataset", f"{i}_{i+200}.npz") for i in range(0, 160000, 200)
]

# Checkpoint directory
CHECKPOINT_DIR = "./checkpoints"


def train(model, resume):
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

        model = model().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        criterion = NLLLoss()
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
        model = model().to(device)
        criterion = NLLLoss()
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
                __,
                remaining_boards,
                remaining_moves,
                __,
            ) = parse_file(
                file,
                remaining_boards,
                remaining_moves,
                remaining_results=[],
                batch_size=BATCH_SIZE,
                features=48,
            )

            for board_states, moves in zip(board_states_batch, moves_batch):
                # Forward pass
                board_states, moves = board_states.to(device), moves.to(device)
                outputs = model(board_states)
                loss = criterion(torch.log(outputs), moves)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Steps
                optimizer.step()
                scheduler.step()

            file_count += 1
            print(f"Files finished: {file_count}/{len(DATA_FILES)}, Loss: {loss}")

            if file_count % 50 == 0 and file_count != 800:
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
