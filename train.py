import pickle
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

from policy import GoPolicyNetwork
from preprocessing import GoDataset

EPOCH = 10
LR = 0.003
BATCH_SIZE = 16


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PolicyNetwork = GoPolicyNetwork().to(device)
    criterion = CrossEntropyLoss()
    optimizer = SGD(PolicyNetwork.parameters(), lr=LR)

    # Decreaes total of 4 times
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1500000, gamma=0.5)

    # Dataset
    dataset = GoDataset()

    print("========== Start training policy network ==========")
    for epoch in range(EPOCH):
        for idx, (board_states, moves) in enumerate(dataset):
            board_states_batches = [
                torch.stack(board_states[i : i + BATCH_SIZE])
                for i in range(0, len(board_states), BATCH_SIZE)
            ]

            moves_batches = [
                torch.stack(moves[i : i + BATCH_SIZE])
                for i in range(0, len(moves), BATCH_SIZE)
            ]

            for board_state_batch, moves_batch in zip(
                board_states_batches, moves_batches
            ):
                # Data to GPU device
                board_state_batch, moves_batch = (
                    board_state_batch.to(device),
                    moves_batch.to(device),
                )

                outputs = PolicyNetwork(board_state_batch)
                loss = criterion(outputs, moves_batch.argmax(dim=1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            if idx % 1000 == 0:
                print(f"Games: {idx}/{len(dataset)}, Loss: {loss}")

        print(f"Epoch: {epoch + 1}/{EPOCH}, Loss: {loss}")

    torch.save(
        PolicyNetwork.state_dict(),
        f"./models/loss_{loss}.pth",
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )


if __name__ == "__main__":
    train()
