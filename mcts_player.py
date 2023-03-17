import torch

from Go.GameState import GameState
from AI.MCTS import MCTS, TreeNode
from networks.resnet import DualResnet
from Go.features import get_board_history


class MCTSPlayer:
    def __init__(self, checkpoint, game):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = DualResnet().to(self.device)
        self.net.eval()
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])

        self.mcts = MCTS(self.net)
        self.game = game

    def get_move(self):
        node = TreeNode()
        print(self.mcts.search(self.game, node, 1))


p = MCTSPlayer("./checkpoints/checkpoint_0_125.pth", GameState())
p.game.do_move((3, 3))
p.get_move()
