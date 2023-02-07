import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import numpy as np

from policy import GoPolicyNetwork

EPOCH = 10
LR = 0.003
BATCH_SIZE = 16

PATH = "./dataset"
BOARDS_FILE = "board_states_batches.npz"
MOVES_FILE = "moves_batches.npz"
CHECKPOINT_DIR = "./checkpoints"


def train():
    # To device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Policy Network
    PolicyNetwork = GoPolicyNetwork().to(device)

    # Loss & Optim
    criterion = CrossEntropyLoss()
    optimizer = SGD(PolicyNetwork.parameters(), lr=LR)

    # Decreaes total of 4 times
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1500000, gamma=0.5)

    # Dataset
    dataset = sorted(
        [d for d in os.scandir(PATH) if d.is_dir()], key=lambda x: int(x.name)
    )

    # Training
    for epoch in range(EPOCH):
        for sub_dir in dataset:
            sub_path = sub_dir.path
            print(sub_path)

            board_states_batches = np.load(f"{sub_path}/{BOARDS_FILE}")["array"]
            moves_batches = np.load(f"{sub_path}/{MOVES_FILE}")["array"]

            for board_states_batch, moves_batch in zip(
                board_states_batches, moves_batches
            ):
                board_states_batch, moves_batch = (
                    torch.from_numpy(board_states_batch).to(device),
                    torch.from_numpy(moves_batch).to(device),
                )

                outputs = PolicyNetwork(board_states_batch)
                loss = criterion(outputs, moves_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "PolicyNetwork_state_dict": PolicyNetwork.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpoint_file,
        )


if __name__ == "__main__":
    train()
