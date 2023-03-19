import torch

from networks.resnet import DualResnet
from Go.features import get_board_history
import Go.GameState as go
from AI.MCTS import MCTS


class MCTSPlayer:
    def __init__(self, checkpoint, game: go.GameState):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = DualResnet().to(self.device)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.net = self.net.eval()

        self.mcts = MCTS(self.net)
        self.game = game

    def get_move(self):
        return self.mcts.search(self.game, temperature=1)

    def update_root(self, action):
        for child in self.mcts.root.children:
            if child.action == action:
                self.mcts.root = child
                self.mcts.root.parent = None
