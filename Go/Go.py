import torch
import numpy as np

from Go.GoBase import GoBase, MiniGo
from Go.util import find_adjacent_cells, find_stone_group, diagonals


class Go(GoBase):
    def __init__(self):
        # Game board
        self.board = np.zeros((19, 19), dtype=int)

        # Board of only black/white/empty
        self.black = np.zeros((19, 19), dtype=int)
        self.white = np.zeros((19, 19), dtype=int)
        self.empty = np.ones((19, 19), dtype=int)

        # Turns since (each intersection)
        self.turns_since = np.zeros((19, 19), dtype=int)

        # Liberties (each intersection)
        self.liberties = np.zeros((19, 19), dtype=int)

        # Black - 1, White - 2
        self.turn = 1

        # Determine KO
        self.hashes = []

        # Optimize
        self.legal_moves_cache = None
        self.legal_eyes_cache = None

    def is_eyeish(self, position, owner):
        """returns whether the position is empty and is surrounded by all stones of 'owner'"""
        (x, y) = position
        if self.board[x, y] != 0:
            return False

        for (nx, ny) in find_adjacent_cells(x, y):
            if self.board[nx, ny] != owner:
                return False
        return True

    def is_eye(self, position, owner, stack=[]):
        (x, y) = position
        opponent = 2 if owner == 1 else 1

        if not self.is_eyeish(position, owner):
            return False

        # (as in Fuego/Michi/etc) ensure that num "bad" diagonals is 0 (edges) or 1
        # where a bad diagonal is an opponent stone or an empty non-eye space
        num_bad_diagonal = 0

        # if in middle of board, 1 bad neighbor is allowable; zero for edges and corners
        allowable_bad_diagonal = 1 if len(find_adjacent_cells(x, y)) == 4 else 0

        for row, col in diagonals(position):
            # opponent stones count against this being eye
            if self.board[row, col] == opponent:
                num_bad_diagonal += 1

            # empty spaces (that aren't themselves eyes) count against it too
            # the 'stack' keeps track of where we've already been to prevent
            # infinite loops of recursion
            elif self.board[row, col] == 0 and (row, col) not in stack:
                stack.append(position)
                if not self.is_eye((row, col), owner, stack):
                    num_bad_diagonal += 1
                stack.pop()

            # at any point, if we've surpassed # allowable, we can stop
            if num_bad_diagonal > allowable_bad_diagonal:
                return False

        return True

    def get_legal_moves(self, include_eyes=True):
        if self.legal_moves_cache is not None:
            if include_eyes:
                return self.legal_moves_cache + self.legal_eyes_cache
            else:
                return self.legal_moves_cache

        self.legal_moves_cache = []
        self.legal_eyes_cache = []

        for x in range(19):
            for y in range(19):
                if self.is_legal(x, y):
                    if not self.is_eye((x, y), self.turn):
                        self.legal_moves_cache.append((x, y))
                    else:
                        self.legal_eyes_cache.append((x, y))

        return self.get_legal_moves(include_eyes)

    def get_groups_around(self, row, col):
        groups_around = []
        groups = self.find_groups()
        for (x, y) in find_adjacent_cells(row, col):
            if self.board[x, y] != 0:
                groups_around.append(groups[find_stone_group(x, y, groups)])

        return groups_around

    def get_capture_size(self):
        planes = torch.zeros((8, 19, 19))
        for (x, y) in self.get_legal_moves():
            # multiple disconnected groups may be captured. hence we loop over
            # groups and count sizes if captured.
            n_captured = 0
            for neighbor_group in self.get_groups_around(x, y):
                # if the neighboring group is opponent stones and they have
                # one liberty, it must be (x,y) and we are capturing them
                # (note suicide and ko are not an issue because they are not
                # legal moves)
                (gx, gy) = neighbor_group[0]
                if (self.find_group_liberty(neighbor_group) == 1) and (
                    self.board[gx, gy] != self.turn
                ):
                    n_captured += len(neighbor_group)

            planes[min(n_captured, 7), x, y] = 1
        return planes

    def get_self_atari_size(self):
        planes = torch.zeros((8, 19, 19))
        for (x, y) in self.get_legal_moves():
            self_atari_size = 0
            game_copy = MiniGo(self)
            game_copy.make_move(x, y)
            groups = game_copy.find_groups()
            cur_stone_group = find_stone_group(x, y, groups)
            for idx, group in enumerate(groups):
                # This move will self atari
                if game_copy.find_group_liberty(group) == 1 and cur_stone_group == idx:
                    self_atari_size += len(group)

            if self_atari_size != 0:
                planes[min(self_atari_size - 1, 7), x, y] = 1

        return planes

    def get_liberties_after(self):
        planes = torch.zeros((8, 19, 19))
        for (x, y) in self.get_legal_moves():
            game_copy = MiniGo(self)
            game_copy.make_move(x, y)

            groups = game_copy.find_groups()
            cur_stone_group = find_stone_group(x, y, groups)
            liberty = game_copy.find_group_liberty(groups[cur_stone_group])
            planes[min(7, liberty - 1), x, y] = 1

        return planes

    def get_stone_lib_pos(self, row, col):
        groups = self.find_groups()
        group = groups[find_stone_group(row, col, groups)]

        liberty = []
        visited = set()

        for (row, col) in group:
            for (x, y) in find_adjacent_cells(row, col):
                if self.board[x, y] == 0 and not (x, y) in visited:
                    visited.add((x, y))
                    liberty.append((x, y))

        return group, liberty

    def get_ladder_capture(self):
        plane = torch.zeros((1, 19, 19))
        for x, y in self.get_legal_moves():
            plane[0, x, y] = self.is_ladder_capture((x, y))

        return plane

    def get_ladder_escape(self):
        plane = torch.zeros((1, 19, 19))
        for x, y in self.get_legal_moves():
            plane[0, x, y] = self.is_ladder_escape((x, y))

        return plane

    def get_sensibleness(self):
        plane = torch.zeros((1, 19, 19))
        for x, y in self.get_legal_moves(include_eyes=False):
            plane[0, x, y] = 1

        return plane

    def remove_dead_groups(self, groups, cur_stone_group, dead_groups):
        for idx in dead_groups:
            if idx != cur_stone_group:
                for (row, col) in groups[idx]:
                    # Update black/white board
                    if self.board[row, col] == 1:
                        self.black[row, col] = 0

                    else:
                        self.white[row, col] = 0

                    # Update empty board
                    self.empty[row, col] = 1

                    # Remove stones on the board
                    self.board[row, col] = 0

                    # Set turns since of the dead stone intersection to 0
                    self.turns_since[row, col] = 0

    def update_turns_since(self):
        for row in range(19):
            for col in range(19):
                # +1 to every stone on the board
                if self.board[row, col] != 0:
                    self.turns_since[row, col] += 1

    def update_liberties(self):
        # Refresh to 0
        self.liberties = np.zeros((19, 19), dtype=int)

        groups = self.find_groups()
        for group in groups:
            liberty = self.find_group_liberty(group)
            for (row, col) in group:
                self.liberties[row, col] = liberty

    def make_move(self, row, col):
        # Update move to board
        self.board[row, col] = self.turn

        if self.turn == 1:
            self.black[row, col] = 1
        else:
            self.white[row, col] = 1

        self.empty[row, col] = 0

        # Switch turns
        self.turn = 2 if self.turn == 1 else 1

        # Find groups, current stones's group and dead groups
        groups = self.find_groups()
        cur_stone_group = find_stone_group(row, col, groups)
        dead_groups = self.find_dead_groups(groups)

        # Remove dead groups from board
        self.remove_dead_groups(groups, cur_stone_group, dead_groups)

        # Update turns since of the rest of the stones
        self.update_turns_since()

        # Update liberties of the rest of the stones
        self.update_liberties()

        # Add to hash
        self.hashes.append(self.board.tostring())
        if len(self.hashes) > 3:
            self.hashes.pop(0)

        # Remove cache
        self.legal_moves_cache = None
