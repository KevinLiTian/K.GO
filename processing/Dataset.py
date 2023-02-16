import torch
from torch.utils.data import Dataset
import numpy as np


class NpzDataset(Dataset):
    def __init__(self, dir_list, batch_size=16):
        self.dir_list = dir_list
        self.batch_size = batch_size

    def __len__(self):
        return (
            sum([np.load(f"{d}/board.npz")["data"].shape[0] for d in self.dir_list])
            // self.batch_size
        )

    def __getitem__(self, index):
        start_index = index * self.batch_size
        boards = []
        moves = []
        for d in self.dir_list:
            board = np.load(f"{d}/board.npz")["data"]
            move = np.load(f"{d}/move.npz")["data"]
            if len(boards) + board.shape[0] >= self.batch_size:
                boards.extend(board[: self.batch_size - len(boards)])
                moves.extend(move[: self.batch_size - len(boards)])
                break
            else:
                boards.extend(board)
                moves.extend(move)
        return torch.from_numpy(np.array(boards)), torch.from_numpy(np.array(moves))
