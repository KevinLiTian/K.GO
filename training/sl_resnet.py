import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

from networks.resnet import DualResnet
from training.utils import parse_file

# Data directory
DATA_FILES = [
    os.path.join("zero_dataset", f"{i}_{i+200}.npz") for i in range(0, 160000, 200)
]

# Checkpoint directory
CHECKPOINT_DIR = "./checkpoints"

NUM_EPOCH = 30
BATCH_SIZE = 2048
LR = 0.1
MOMENTUM = 0.9
VALUE_WEIGHT = 0.01
REG_WEIGHT = 1e-4


def train(resume):
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

        model = DualResnet().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Setup optimizer
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Setup learning rate annealing
        milestones = [200000, 400000, 600000, 700000]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "file_count" in checkpoint:
            file_count = checkpoint["file_count"]
            cur_epoch = checkpoint["epoch"]
        else:
            file_count = 0
            cur_epoch = checkpoint["epoch"] + 1

    else:
        model = DualResnet().to(device)

        # Setup optimizer
        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

        # Setup learning rate annealing
        milestones = [200000, 400000, 600000, 700000]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        file_count = 0
        cur_epoch = 0

    # Setup loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()

    # Training utilities
    remaining_boards, remaining_moves, remaining_results = [], [], []

    # Main train loop
    print(f"Start epoch {cur_epoch}, file count {file_count}")
    model.train()
    for epoch in range(cur_epoch, NUM_EPOCH):
        if file_count == len(DATA_FILES):
            file_count = 0
        for file in DATA_FILES[file_count:]:
            (
                board_states_batch,
                moves_batch,
                results_batch,
                remaining_boards,
                remaining_moves,
                remaining_results,
            ) = parse_file(
                file,
                remaining_boards,
                remaining_moves,
                remaining_results,
                BATCH_SIZE,
                features=49,
            )

            total_policy_loss = 0.0
            total_value_loss = 0.0
            count = 0
            for board_states, moves, results in zip(
                board_states_batch, moves_batch, results_batch
            ):
                # Forward pass
                board_states, moves, results = (
                    board_states.to(device),
                    moves.to(device),
                    results.to(device),
                )
                policy_output, value_output = model(board_states)
                policy_loss = policy_criterion(policy_output, moves)
                value_loss = value_criterion(value_output, results)
                reg_loss = sum(p.pow(2).sum() for p in model.parameters())
                loss = policy_loss + VALUE_WEIGHT * value_loss + REG_WEIGHT * reg_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Steps
                optimizer.step()
                scheduler.step()

                # Progress report
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                count += 1

            file_count += 1
            print(
                f"Files finished: {file_count}/{len(DATA_FILES)}, Policy loss: {total_policy_loss / count}, Value loss: {total_value_loss / count}"
            )

            if file_count % 100 == 0 and file_count != 800:
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
