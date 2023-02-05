import os

import torch
from torch.utils.data import Dataset
from sgfmill import sgf

from go import Go

DATA_PATH = "data"


class GoDataset(Dataset):
    def __init__(self):
        # 45532 games by 6p - 9p players
        self.games = []

        # Read from data folder
        print("========== Start preprocessing game files ==========")
        num_games_processed = 0
        for dirpath, __, filenames in os.walk(DATA_PATH):
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
                    elif int(black_rank[0]) < 6 or int(white_rank[0]) < 6:
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
    for node in game.get_main_sequence():
        move = node.get_move()
        if move[0] is not None and move[1] is not None:
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
    20 feature planes
    - Stone colours (3)
    - Constant 1 plane (1)
    - Turns since (8)
    - Liberties (8)
    """
    # Stone colour
    if colour == "b":
        player_stones = torch.Tensor(board.black).unsqueeze(0)
        opponent_stones = torch.Tensor(board.white).unsqueeze(0)
    else:
        player_stones = torch.Tensor(board.white).unsqueeze(0)
        opponent_stones = torch.Tensor(board.black).unsqueeze(0)

    empty = torch.Tensor(board.empty).unsqueeze(0)

    # Const Ones
    const_one = torch.ones(19, 19).unsqueeze(0)

    # Other feature planes (8x19x19)
    turns_since = one_hot_representation(board.turns_since)
    liberties = one_hot_representation(board.liberties)

    return torch.cat(
        [player_stones, opponent_stones, empty, const_one, turns_since, liberties],
        dim=0,
    )


def one_hot_representation(board):
    planes = torch.zeros(8, 19, 19, dtype=torch.float32)

    for i in range(19):
        for j in range(19):
            val = board[i][j]
            if val < 8:
                planes[val - 1, i, j] = 1
            else:
                planes[7, i, j] = 1

    return planes
