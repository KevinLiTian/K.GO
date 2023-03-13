import math

import torch
import numpy as np

import Go.GameState as go
from Go.features import get_sensibleness
from training.process import create_board_state

C_PUCT = 0.5
NUM_SIM = 1600


class Node:
    def __init__(self, state: go.GameState, parent=None, action_taken=None, prior=0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.Q = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        return self.Q + C_PUCT * child.prior * (
            math.sqrt(self.visit_count) / (child.visit_count + 1)
        )

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                move = (int(action / 19), int(action % 19))
                child_state.do_move(move)
                child = Node(child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        self.Q = self.value_sum / self.visit_count

        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def search(self, state: go.GameState):
        root = Node(state)

        for __ in range(NUM_SIM):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            if state.is_end_of_game:
                value = state.get_winner()
            else:
                policy, value = self.model(
                    torch.from_numpy(create_board_state(node.state)).unsqueeze(0)
                )

                # Policy distribution
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = get_sensibleness(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                # Value scalar
                value = value.item()

                # Expand leaf node with policy
                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(19 * 19)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    