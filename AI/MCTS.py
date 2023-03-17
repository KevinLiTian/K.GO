import math
from copy import deepcopy

import torch
import numpy as np

import Go.GameState as go
from Go.features import get_board_history, get_all_legal

C_PUCT = 5
NUM_SIM = 1600


class TreeNode:
    """Represents a board state and stores statistics for actions at that state.
    Attributes:
        Nsa: An integer for visit count.
        Wsa: A float for the total action value.
        Qsa: A float for the mean action value.
        Psa: A float for the prior probability of reaching this node.
        action: A tuple(row, column) of the prior move of reaching this node.
        children: A list which stores child nodes.
        child_psas: A vector containing child probabilities.
        parent: A TreeNode representing the parent node.
    """

    def __init__(self, parent=None, action=None, psa=0.0):
        """Initializes TreeNode with the initial statistics and data."""
        self.Nsa = 0
        self.Wsa = 0.0
        self.Qsa = 0.0
        self.Psa = psa
        self.action = action
        self.children = []
        self.parent = parent

    def is_not_leaf(self):
        """Checks if a TreeNode is a leaf.
        Returns:
            A boolean value indicating if a TreeNode is a leaf.
        """
        if len(self.children) > 0:
            return True
        return False

    def select_child(self):
        """Selects a child node based on the AlphaZero PUCT formula.
        Returns:
            A child TreeNode which is the most promising according to PUCT.
        """
        Nsa = np.array([child.Nsa for child in self.children])
        Qsa = np.array([child.Qsa for child in self.children])
        Psa = np.array([child.Psa for child in self.children])
        uct = Qsa + Psa * C_PUCT * np.sqrt(self.Nsa) / (1 + Nsa)
        highest_index = np.argmax(uct)
        return self.children[highest_index]

    def expand_node(self, game: go.GameState, psa_vector):
        """Expands the current node by adding valid moves as children.
        Args:
            game: An object containing the game state.
            psa_vector: A list containing move probabilities for each move.
        """
        valid_moves = game.get_legal_moves()
        for move in valid_moves:
            action = deepcopy(move)
            idx = game.size * action[0] + action[1]
            self.add_child_node(parent=self, action=action, psa=psa_vector[0, idx])

    def add_child_node(self, parent, action, psa=0.0):
        """Creates and adds a child TreeNode to the current node.
        Args:
            parent: A TreeNode which is the parent of this node.
            action: A tuple(row, column) of the prior move to reach this node.
            psa: A float representing the raw move probability for this node.
        Returns:
            The newly created child TreeNode.
        """
        child_node = TreeNode(parent=parent, action=action, psa=psa)
        self.children.append(child_node)
        return child_node

    def back_prop(self, v):
        """Update the current node's statistics based on the game outcome.
        Args:
            v: A float representing the network value of this state.
        """
        self.Nsa += 1
        self.Wsa = self.Wsa + v
        self.Qsa = self.Wsa / self.Nsa


class MCTS:
    """Represents a Monte Carlo Tree Search Algorithm.
    Attributes:
        root: A TreeNode representing the board state and its statistics.
        game: An object containing the game state.
        net: An object containing the neural network.
    """

    def __init__(self, net):
        """Initializes TreeNode with the TreeNode, board and neural network."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root = None
        self.game = None
        self.net = net

    @torch.no_grad()
    def search(self, game: go.GameState, node: TreeNode, temperature):
        """MCTS loop to get the best move which can be played at a given state.
        Args:
            game: An object containing the game state.
            node: A TreeNode representing the board state and its statistics.
            temperature: A float to control the level of exploration.
        Returns:
            A child node representing the best move to play at this state.
        """
        self.root = node
        self.game = game

        for i in range(NUM_SIM):
            node = self.root
            game = self.game.copy()

            # Go down to a leaf
            while node.is_not_leaf():
                node = node.select_child()
                game.do_move(node.action)

            # Get move probabilities and values from the network for this state.
            board_state = (
                torch.from_numpy(get_board_history(game)).unsqueeze(0).to(self.device)
            )
            psa_vector, v = self.net(board_state)
            psa_vector = torch.softmax(psa_vector, dim=1).cpu().numpy()
            v = float(v)

            # Add Dirichlet noise to the psa_vector of the root node.
            if node.parent is None:
                psa_vector = self.add_dirichlet_noise(game, psa_vector)

            # Remove invalid moves probabilities
            valid_moves = get_all_legal(game)
            psa_vector = psa_vector * valid_moves

            # Renormalize
            psa_vector_sum = psa_vector.sum()
            if psa_vector_sum > 0:
                psa_vector /= psa_vector_sum

            # Expand
            node.expand_node(game=game, psa_vector=psa_vector)

            # Check for end game
            while node is not None:
                v = -v
                node.back_prop(v)
                node = node.parent

        # Select the move with a temperature param
        highest_nsa, highest_idx = 0, 0
        for idx, child in enumerate(self.root.children):
            if child.Nsa > highest_nsa:
                highest_nsa = child.Nsa
                highest_idx = idx

        return self.root.children[highest_idx].action

    def add_dirichlet_noise(self, game: go.GameState, psa_vector):
        """Add Dirichlet noise to the psa_vector of the root node.
        This is for additional exploration.
        Args:
            game: An object containing the game state.
            psa_vector: A probability vector.
        Returns:
            A probability vector which has Dirichlet noise added to it.
        """

        # generate noise vector from Dirichlet distribution
        eta = np.random.dirichlet([0.03] * game.size * game.size)

        p_exp = 0.75 * psa_vector + 0.25 * eta
        p_exp /= np.sum(p_exp)
        return p_exp
