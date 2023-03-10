import torch

from networks.policy import Conv192
from training.process import create_board_state


class GreedyPolicyPlayer:
    def __init__(self, checkpoint):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = Conv192().to(self.device)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])

    def get_move(self, state):
        # list with sensible moves
        sensible_moves = [move for move in state.get_legal_moves(include_eyes=False)]
        if len(sensible_moves) > 0:
            board_states = create_board_state(state)[:48]
            board_states = torch.from_numpy(board_states).to(self.device)
            move_probs = self.policy(board_states)
            idx = torch.argmax(move_probs)

            row = int(idx / 19)
            col = int(idx % 19)
            return (row, col)

        return None
