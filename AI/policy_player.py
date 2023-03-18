import torch
import numpy as np

from networks.policy import Conv192
from training.process import create_board_state
import Go.GameState as go


class GreedyPolicyPlayer:
    def __init__(self, checkpoint, game: go.GameState):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = Conv192().to(self.device)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])

        self.game = game

    def get_move(self):
        # list with sensible moves
        sensible_moves = [
            move for move in self.game.get_legal_moves(include_eyes=False)
        ]
        if len(sensible_moves) > 0:
            board_states = create_board_state(self.game)[:48]
            board_states = torch.from_numpy(board_states).to(self.device)
            move_probs = self.policy(board_states)
            idx = torch.argmax(move_probs)

            row = int(idx / 19)
            col = int(idx % 19)
            if (row, col) not in sensible_moves:
                return go.PASS_MOVE
            return (row, col)
        return go.PASS_MOVE


class ProbabilisticPolicyPlayer:
    def __init__(self, checkpoint, game: go.GameState):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = Conv192().to(self.device)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])

        self.game = game

    def get_move(self):
        # list with sensible moves
        sensible_moves = [
            move for move in self.game.get_legal_moves(include_eyes=False)
        ]
        if len(sensible_moves) > 0:
            board_states = create_board_state(self.game)[:48]
            board_states = torch.from_numpy(board_states).to(self.device)
            move_probs = self.policy(board_states)

            # Select possible moves at random
            idx = torch.multinomial(move_probs, num_samples=1)

            row = int(idx / 19)
            col = int(idx % 19)
            if (row, col) not in sensible_moves:
                return go.PASS_MOVE
            return (row, col)
        return go.PASS_MOVE

    def get_moves(self, states):
        """Batch version of get_move. A list of moves is returned (one per state)"""
        sensible_move_lists = [
            [move for move in st.get_legal_moves(include_eyes=False)] for st in states
        ]

        board_states = []
        for state in states:
            board_states.append(create_board_state(state)[:48])
        board_states = np.stack(board_states, axis=0)
        board_states = torch.from_numpy(board_states).to(self.device)
        all_moves_distributions = self.policy(board_states)

        move_list = [None] * len(states)
        for i, move_probs in enumerate(all_moves_distributions):
            if len(sensible_move_lists[i]) == 0:
                move_list[i] = go.PASS_MOVE

            else:
                idx = torch.multinomial(move_probs, num_samples=1)

                row = int(idx / 19)
                col = int(idx % 19)

                if (row, col) not in sensible_move_lists[i]:
                    move_list[i] = go.PASS_MOVE
                else:
                    move_list[i] = (row, col)
        return move_list
