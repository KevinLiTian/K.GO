import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sgfmill import sgf

from Go.Go import Go

DATA_PATH = "data"
BATCH_SIZE = 16
START = 50
END = 60
PATH = f"./dataset"


class GoDataset(Dataset):
    def __init__(self):
        # 66236 games by 4p - 9p players
        self.games = []

        # Read from data folder
        print("========== Start preprocessing game files ==========")
        num_games_processed = 0
        for dirpath, __, filenames in os.walk(DATA_PATH):
            filenames.sort()
            for file in filenames:
                if file.endswith("sgf"):
                    # Progress report
                    num_games_processed += 1
                    if num_games_processed % 10000 == 0:
                        print(f"{num_games_processed}/115623 games processed")

                    file_path = os.path.join(dirpath, file)
                    with open(file_path, "rb") as file:
                        game = sgf.Sgf_game.from_bytes(file.read())

                    # Discard handicap games
                    try:
                        if game.get_handicap() is not None:
                            continue
                    except ValueError:
                        continue

                    # Discard setup stone games (座子)
                    root = game.get_root()
                    if root.has_setup_stones():
                        continue

                    # Discard amateur games and low dan games
                    try:
                        black_rank = root.get("BR")
                        white_rank = root.get("WR")
                    except ValueError:
                        continue

                    if len(black_rank) != 2 or len(white_rank) != 2:
                        continue
                    elif black_rank[1] != "p" or white_rank[1] != "p":
                        continue
                    elif int(black_rank[0]) < 4 or int(white_rank[0]) < 4:
                        continue

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
            colour, (row, col) = move

            # Add board state
            board_states.append(create_board_state(board, colour))

            # Make move
            board.make_move(row, col)

            # Add move
            move = torch.zeros(19, 19)
            move[row, col] = 1
            moves.append(move.flatten())

    return board_states, moves


def create_board_state(board: Go, colour):
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
    if colour == "b":
        player_stones = torch.Tensor(board.black).unsqueeze(0)
        opponent_stones = torch.Tensor(board.white).unsqueeze(0)
    else:
        player_stones = torch.Tensor(board.white).unsqueeze(0)
        opponent_stones = torch.Tensor(board.black).unsqueeze(0)

    empty = torch.Tensor(board.empty).unsqueeze(0)

    # Const 1s (1x19x19)
    const_one = torch.ones((1, 19, 19))

    # 8 channel feature planes (8x19x19)
    turns_since = one_hot_representation(board.turns_since)
    liberties = one_hot_representation(board.liberties)
    capture_size = board.get_capture_size()
    self_atari_size = board.get_self_atari_size()
    liberties_after = board.get_liberties_after()

    # Ladder (1x19x19)
    ladder_capture = board.get_ladder_capture()
    ladder_escape = board.get_ladder_escape()

    # Sensibleness (1x19x19)
    sensibleness = board.get_sensibleness()

    # Const 0s (1x19x19)
    const_zero = torch.zeros((1, 19, 19))

    return torch.cat(
        [
            player_stones,
            opponent_stones,
            empty,
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


if __name__ == "__main__":
    dataset = GoDataset()

    for idx in range(START, END):
        board_states, moves = dataset[idx]

        if len(moves) % BATCH_SIZE != 0:
            board_states = board_states[: -(len(moves) % BATCH_SIZE)]
            moves = moves[: -(len(moves) % BATCH_SIZE)]

        board_states_batches = [
            torch.stack(board_states[i : i + BATCH_SIZE])
            for i in range(0, len(board_states), BATCH_SIZE)
        ]

        moves_batches = [
            torch.stack(moves[i : i + BATCH_SIZE])
            for i in range(0, len(moves), BATCH_SIZE)
        ]

        board_states_batches = [np.array(tensor) for tensor in board_states_batches]
        moves_batches = [np.array(tensor) for tensor in moves_batches]

        path = f"{PATH}/{idx}"
        if not os.path.exists(path):
            os.makedirs(path)

        np.savez_compressed(f"{path}/board_states_batches", array=board_states_batches)
        np.savez_compressed(f"{path}/moves_batches", array=moves_batches)
