import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset
from sgfmill import sgf

from go import Go

DATA_PATH = "data"
SAMPLES = 3000


class GoDataset(Dataset):
    def __init__(self):
        self.board_states = []
        self.moves = []
        self.load_data()

    def __getitem__(self, index):
        return self.board_states[index], self.moves[index]

    def __len__(self):
        return len(self.board_states)

    def load_data(self):
        print("========== Start loading games data ==========")
        games = []

        # Read from data folder
        for dirpath, __, filenames in os.walk(DATA_PATH):
            for file in filenames:
                if file.endswith("sgf"):
                    file_path = os.path.join(dirpath, file)
                    games.append(file_path)

        # Use a ThreadPoolExecutor to parse the SGF files simultaneously
        with ThreadPoolExecutor() as executor:
            results = [
                executor.submit(parse_game, game)
                for game in random.sample(games, SAMPLES)
            ]
            completed = 0
            for future in as_completed(results):
                boards, moves = future.result()
                self.board_states.extend(boards)
                self.moves.extend(moves)

                # Keep track of progress
                completed += 1
                if completed % 100 == 0:
                    print(f"{completed}/{SAMPLES} games completed loading")


def parse_game(game):
    with open(game, "rb") as file:
        game = sgf.Sgf_game.from_bytes(file.read())

    # Discard handicap games
    try:
        if game.get_handicap() is not None:
            return [], []
    except ValueError:
        return [], []

    # Discard setup stone games (座子)
    root = game.get_root()
    if root.has_setup_stones():
        return [], []

    # Discard amateur games
    try:
        black_rank = root.get("BR")
        white_rank = root.get("WR")
    except ValueError:
        return [], []

    if len(black_rank) != 2 or len(white_rank) != 2:
        return [], []
    elif black_rank[1] != "p" or white_rank[1] != "p":
        return [], []

    # Discard games that are too short
    if len(game.get_main_sequence()) < 100:
        return [], []

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
    turns_since = binary_representation(board.turns_since)
    liberties = binary_representation(board.liberties)

    return torch.cat(
        [player_stones, opponent_stones, empty, const_one, turns_since, liberties],
        dim=0,
    )


def binary_representation(board):
    # Convert the input list into a PyTorch Tensor
    int_tensor = torch.tensor(board, dtype=torch.int32)

    # Create an empty tensor to store the binary representation
    bin_tensor = torch.zeros((8, 19, 19), dtype=torch.int8)

    for bit in range(8):
        # Shift the bits to the right and extract the k-th bit
        bin_tensor[bit] = (int_tensor >> bit) & 1

    return bin_tensor
