import os

import numpy as np
import torch
from sgfmill import sgf

from Go.Go import Go

BATCH_SIZE = 16
START = 0
END = 10000

DATA_PATH = "data"
PATH = f"./dataset"


class GoDataset:
    def __init__(self):
        # 142010 KGS games by
        # - One of the players is 7 dan or stronger
        # - Both players 6 dan
        # - No handicap games included
        self.games = []

        # Read from data folder
        for dirpath, __, filenames in os.walk(DATA_PATH):
            filenames.sort()
            for file in filenames:
                if file.endswith("sgf"):
                    file_path = os.path.join(dirpath, file)
                    self.games.append(file_path)

    def __getitem__(self, index):
        return parse_game(self.games[index])

    def __len__(self):
        return len(self.games)


def parse_game(game):
    with open(game, "rb") as file:
        game = sgf.Sgf_game.from_bytes(file.read())

    # Initialize game states
    board_states = []
    moves = []
    board = Go()

    # Read moves one by one
    main_sequence = game.get_main_sequence()
    length = len(main_sequence)
    for idx, node in enumerate(main_sequence):
        move = node.get_move()
        if move[0] is not None and move[1] is not None:
            print(f"Moves processed: {idx}/{length}")
            row, col = move[1]

            # Add board state
            board_states.append(create_board_state(board))

            # Make move
            board.make_move(row, col)

            # Add move
            move = torch.zeros(19, 19)
            move[row, col] = 1
            moves.append(torch.argmax(move.flatten()))

    return board_states, moves


def create_board_state(board: Go):
    """
    48 feature planes
    - Stone colours (3)
    - Constant 1 plane (1)
    - Turns since (8)
    - Liberties (8)
    - Capture size (8)
    - Self atari size (8)
    - Liberty after (8)
    - Ladder capture (1)
    - Ladder escape (1)
    - Sensibleness (1)
    - Const 0 plane (1)
    """

    # Stone colour each (1x19x19)
    stones = torch.zeros((3, 19, 19))
    player = board.turn
    opponent = 2 if player == 1 else 1
    stones[0, :, :] = torch.Tensor(board.board == player)
    stones[1, :, :] = torch.Tensor(board.board == opponent)
    stones[2, :, :] = torch.Tensor(board.board == 0)

    # Const 1s (1x19x19)
    const_one = torch.ones((1, 19, 19))

    # 8 channel feature planes (8x19x19)
    turns_since = one_hot_representation(board.turns_since)
    liberties = one_hot_representation(board.liberties)
    capture_size = board.get_capture_size()
    self_atari_size = board.get_self_atari_size()
    liberties_after = board.get_liberties_after()

    # Ladders (2x19x19)
    ladder_capture = board.get_ladder_capture()
    ladder_escape = board.get_ladder_escape()

    # Sensibleness (1x19x19)
    sensibleness = board.get_sensibleness()

    # Const 0s (1x19x19)
    const_zero = torch.zeros((1, 19, 19))

    # Player's colour (Used in value network only)
    player_colour = torch.ones((1, 19, 19)) * (board.turn == 1)

    return torch.cat(
        [
            stones,
            const_one,
            turns_since,
            liberties,
            capture_size,
            self_atari_size,
            liberties_after,
            ladder_capture,
            ladder_escape,
            sensibleness,
            const_zero,
            player_colour,
        ],
        dim=0,
    )


def one_hot_representation(board):
    planes = torch.zeros(8, 19, 19, dtype=torch.float32)

    for i in range(19):
        for j in range(19):
            val = board[i, j]
            if val < 8:
                planes[val - 1, i, j] = 1
            else:
                planes[7, i, j] = 1

    return planes

def make_moves():
    go = Go()
    go.make_move(3, 3)
    go.make_move(3, 4)
    go.make_move(2, 4)
    go.make_move(0, 0)
    go.make_move(4, 3)
    go.make_move(0, 1)
    return go

if __name__ == "__main__":
    import cProfile
    go = make_moves()

    cProfile.run("create_board_state(go)")
    # dataset = GoDataset()

    # for idx in range(START, END):
        # board_states, moves = dataset[idx]

        # if len(moves) % BATCH_SIZE != 0:
        #     board_states = board_states[: -(len(moves) % BATCH_SIZE)]
        #     moves = moves[: -(len(moves) % BATCH_SIZE)]

        # board_states_batches = [
        #     torch.stack(board_states[i : i + BATCH_SIZE])
        #     for i in range(0, len(board_states), BATCH_SIZE)
        # ]

        # moves_batches = [
        #     torch.stack(moves[i : i + BATCH_SIZE])
        #     for i in range(0, len(moves), BATCH_SIZE)
        # ]

        # board_states_batches = [np.array(tensor) for tensor in board_states_batches]
        # moves_batches = [np.array(tensor) for tensor in moves_batches]

        # path = f"{PATH}/{idx}"
        # if not os.path.exists(path):
        #     os.makedirs(path)

        # np.savez_compressed(f"{path}/board_states_batches", array=board_states_batches)
        # np.savez_compressed(f"{path}/moves_batches", array=moves_batches)
