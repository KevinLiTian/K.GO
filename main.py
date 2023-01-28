import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
import pickle

from policy import GoPolicyNetwork
from dataset import GoDataset

EPOCH = 100
LR = 0.1
BATCH = 32


def train():
    PolicyNetwork = GoPolicyNetwork()
    dataloader = DataLoader(GoDataset(), batch_size=BATCH, shuffle=True, drop_last=True)
    criterion = CrossEntropyLoss()
    optimizer = Adam(PolicyNetwork.parameters(), lr=LR)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.75)

    print("========== Start training policy network ==========")
    for epoch in range(EPOCH):
        for batch_num, (board_states, moves) in enumerate(dataloader):
            outputs = PolicyNetwork(board_states)
            loss = criterion(outputs, moves.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_num % 100 == 0:
                print(
                    f"Batches completed: {batch_num}/{len(dataloader)} Loss: {loss} Learning rate: {scheduler.get_last_lr()}"
                )

        print(f"Epoch {epoch + 1}/{EPOCH}: Loss {loss}")

    torch.save(
        PolicyNetwork.state_dict(),
        f"./models/lr_{LR}_batch_{BATCH}_loss_{loss}.pth",
        pickle_protocol=pickle.HIGHEST_PROTOCOL,
    )


if __name__ == "__main__":
    train()
