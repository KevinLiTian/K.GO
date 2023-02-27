import torch

from networks.policy import GoPolicyNetwork

CHECKPOINT_DIR = "./checkpoints"

net1 = GoPolicyNetwork()
net1.load_state_dict(
    torch.load(f"{CHECKPOINT_DIR}/checkpoint_1_200.pth")["model_state_dict"]
)

net2 = GoPolicyNetwork()
net2.load_state_dict(
    torch.load(f"{CHECKPOINT_DIR}/checkpoint_1_300.pth")["model_state_dict"]
)


# Compare the state dictionaries of the models
num_different_params = 0
for k1, v1 in net1.state_dict().items():
    v2 = net2.state_dict()[k1]
    num_different_params += torch.sum(torch.ne(v1, v2))

print(
    f"There are {num_different_params} different parameters between the two state dictionaries."
)
