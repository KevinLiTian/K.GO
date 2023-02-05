from copy import deepcopy

import torch


class Go:
    def __init__(self):
        # Game board
        self.board = construct_board(0)

        # Board of only black/white/empty
        self.black = construct_board(0)
        self.white = construct_board(0)
        self.empty = construct_board(1)

        # Turns since (each intersection)
        self.turns_since = construct_board(0)

        # Liberties (each intersection)
        self.liberties = construct_board(0)

        # Black - 1, White - 2
        self.turn = 1

        # Determine KO
        self.hashes = []

    def display(self):
        for row in range(19):
            for col in range(19):
                print(self.board[row][col], end="")
            print()
        print()

    """ Features """

    def is_legal(self, row, col):
        if self.board[row][col] != 0:
            return False
        if self.is_suicide(row, col):
            return False
        if self.is_ko(row, col):
            return False
        return True

    def is_ko(self, row, col):
        game_copy = deepcopy(self)
        game_copy.make_move(row, col)

        if str(game_copy.board) in self.hashes:
            return True

        return False

    def is_suicide(self, row, col):
        game_copy = deepcopy(self)
        game_copy.make_move(row, col)

        groups = game_copy.find_groups()
        cur_stone_group = find_stone_group(row, col, groups)
        dead_groups = game_copy.find_dead_groups(groups)

        # Only cur stone group is dead, suicide
        if len(dead_groups) == 1 and cur_stone_group in dead_groups:
            return True

        return False

    def is_eyeish(self, position, owner):
        """returns whether the position is empty and is surrounded by all stones of 'owner'"""
        (x, y) = position
        if self.board[x][y] != 0:
            return False

        for (nx, ny) in find_adjacent_cells(x, y):
            if self.board[nx][ny] != owner:
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
            if self.board[row][col] == opponent:
                num_bad_diagonal += 1

            # empty spaces (that aren't themselves eyes) count against it too
            # the 'stack' keeps track of where we've already been to prevent
            # infinite loops of recursion
            elif self.board[row][col] == 0 and (row, col) not in stack:
                stack.append(position)
                if not self.is_eye((row, col), owner, stack):
                    num_bad_diagonal += 1
                stack.pop()

            # at any point, if we've surpassed # allowable, we can stop
            if num_bad_diagonal > allowable_bad_diagonal:
                return False

        return True

    def get_legal_moves(self, include_eyes=True):
        legal_moves = []
        eyes = []

        for x in range(19):
            for y in range(19):
                if self.is_legal(x, y):
                    if not self.is_eye((x, y), self.turn):
                        legal_moves.append((x, y))
                    else:
                        eyes.append((x, y))

        if include_eyes:
            return legal_moves + eyes
        else:
            return legal_moves

    def get_groups_around(self, row, col):
        groups_around = []
        groups = self.find_groups()
        for (x, y) in find_adjacent_cells(row, col):
            if self.board[x][y] != 0:
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
                    self.board[gx][gy] != self.turn
                ):
                    n_captured += len(neighbor_group)

            planes[min(n_captured, 7), x, y] = 1
        return planes

    def get_self_atari_size(self):
        planes = torch.zeros((8, 19, 19))
        for (x, y) in self.get_legal_moves():
            self_atari_size = 0
            game_copy = deepcopy(self)
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
            game_copy = deepcopy(self)
            game_copy.make_move(x, y)

            liberty = game_copy.liberties[x][y]
            planes[min(7, liberty - 1), x, y] = 1

        return planes

    def get_stone_lib_pos(self, row, col):
        groups = self.find_groups()
        group = groups[find_stone_group(row, col, groups)]

        liberty = []
        visited = []

        for (row, col) in group:
            for (x, y) in find_adjacent_cells(row, col):
                if self.board[x][y] == 0 and not (x, y) in visited:
                    visited.append((x, y))
                    liberty.append((x, y))

        return group, liberty

    def is_ladder_capture(self, action, prey=None, remaining_attempts=80):
        """Check if moving at action results in a ladder capture, defined as being next
        to an enemy group with two liberties, and with no ladder_escape move afterward
        for the other player.
        If prey is None, check all adjacent groups, otherwise only the prey
        group is checked.  In the (prey is None) case, if this move is a ladder
        capture for any adjacent group, it's considered a ladder capture.
        Recursion depth between is_ladder_capture() and is_ladder_escape() is
        controlled by the remaining_attempts argument.  If it reaches 0, the
        move is assumed not to be a ladder capture.
        """

        row, col = action
        # ignore illegal moves
        if not self.is_legal(row, col):
            return False

        # if we haven't found a capture by a certain number of moves, assume it's worked.
        if remaining_attempts <= 0:
            return True

        hunter_player = self.turn
        prey_player = 2 if self.turn == 1 else 1

        if prey is None:
            # default case is to check all adjacent prey_player groups that
            # have 2 liberties
            potential_prey = []
            for group in self.get_groups_around(row, col):
                x, y = group[0]
                if (
                    self.board[x][y] == prey_player
                    and self.find_group_liberty(group) == 2
                ):
                    potential_prey.append((x, y))

        else:
            # we are checking a specific group (called from is_ladder_escape)
            potential_prey = [prey]

        for (prey_x, prey_y) in potential_prey:
            # attempt to capture the group at prey_x, prey_y in a ladder
            game_copy = deepcopy(self)
            game_copy.make_move(row, col)

            # we only want to check a limited set of possible escape moves:
            # - extensions from the remaining liberty of the prey group.
            # - captures of enemy groups adjacent to the prey group.
            group, possible_escapes = game_copy.get_stone_lib_pos(prey_x, prey_y)

            # Check if any hunter groups adjacent to the prey groups
            # are in atari.  Capturing these groups are potential escapes.
            for x, y in group:
                for (nx, ny) in find_adjacent_cells(x, y):
                    if game_copy.board[nx][ny] == hunter_player:
                        __, n_lib_pos = game_copy.get_stone_lib_pos(nx, ny)
                        if len(n_lib_pos) == 1:
                            possible_escapes.extend(n_lib_pos)

            if not any(
                game_copy.is_ladder_escape(
                    (escape_x, escape_y),
                    prey=(prey_x, prey_y),
                    remaining_attempts=(remaining_attempts - 1),
                )
                for (escape_x, escape_y) in possible_escapes
            ):
                # we found at least one group that could be captured in a
                # ladder, so this move is a ladder capture.
                return True

        # no ladder captures were found
        return False

    def is_ladder_escape(self, action, prey=None, remaining_attempts=80):
        """Check if moving at action results in a ladder escape, defined as being next
        to a current player's group with one liberty, with no ladder captures
        afterward.  Going from 1 to >= 3 liberties is counted as escape, or a
        move giving two liberties without a subsequent ladder capture.
        If prey is None, check all adjacent groups, otherwise only the prey
        group is checked.  In the (prey is None) case, if this move is a ladder
        escape for any adjacent group, this move is a ladder escape.
        Recursion depth between is_ladder_capture() and is_ladder_escape() is
        controlled by the remaining_attempts argument.  If it reaches 0, the
        move is assumed not to be a ladder capture.
        """

        row, col = action
        # ignore illegal moves
        if not self.is_legal(row, col):
            return False

        # if we haven't found an escape by a certain number of moves, give up.
        if remaining_attempts <= 0:
            return False

        prey_player = self.turn

        if prey is None:
            # default case is to check all adjacent groups that might be in a
            # ladder (i.e., with one liberty)
            potential_prey = []
            for group in self.get_groups_around(row, col):
                x, y = group[0]
                if (
                    self.board[x][y] == prey_player
                    and self.find_group_liberty(group) == 1
                ):
                    potential_prey.append((x, y))

        else:
            # we are checking a specific group (called from is_ladder_capture)
            potential_prey = [prey]

        # This move is an escape if it's an escape for any of the potential_prey
        for (prey_x, prey_y) in potential_prey:
            # make the move, see if the group at (prey_x, prey_y) has escaped,
            # defined as having >= 3 liberties, or 2 liberties and not
            # ladder_capture() being true when played on either of those
            # liberties.
            game_copy = deepcopy(self)
            game_copy.make_move(row, col)

            # if we have >= 3 liberties, we've escaped
            __, liberties = game_copy.get_stone_lib_pos(prey_x, prey_y)
            if len(liberties) >= 3:
                return True

            # if we only have 1 liberty, we've failed
            if len(liberties) == 1:
                # not an escape - check next group
                continue

            # The current group has two liberties.  It may still be in a ladder.
            # Check both liberties to see if they are ladder captures
            if any(
                game_copy.is_ladder_capture(
                    possible_capture,
                    prey=(prey_x, prey_y),
                    remaining_attempts=(remaining_attempts - 1),
                )
                for possible_capture in liberties
            ):
                # not an escape - check next group
                continue

            # reached two liberties that were no longer ladder-capturable
            return True

        # no ladder escape found
        return False

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

    """ Game Logic """

    def find_groups(self):
        groups = []
        visited = []

        # Check each cell of board
        for row in range(19):
            for col in range(19):
                # If there's an unvisited stone
                if self.board[row][col] != 0 and not (row, col) in visited:
                    visited.append((row, col))
                    group = [(row, col)]

                    # DFS
                    frontier = [(row, col)]
                    while len(frontier) != 0:
                        (x, y) = frontier.pop()
                        for (r, c) in find_adjacent_cells(x, y):
                            # Same colour and not visited
                            if (
                                self.board[r][c] == self.board[x][y]
                                and not (r, c) in visited
                            ):
                                group.append((r, c))
                                visited.append((r, c))
                                frontier.append((r, c))

                    groups.append(group)

        return groups

    def find_group_liberty(self, group):
        liberty = 0
        visited = []

        for (row, col) in group:
            for (x, y) in find_adjacent_cells(row, col):
                if self.board[x][y] == 0 and not (x, y) in visited:
                    visited.append((x, y))
                    liberty += 1

        return liberty

    def find_dead_groups(self, groups):
        dead_groups = []
        for idx, group in enumerate(groups):
            if self.find_group_liberty(group) == 0:
                dead_groups.append(idx)

        return dead_groups

    def remove_dead_groups(self, groups, cur_stone_group, dead_groups):
        for idx in dead_groups:
            if idx != cur_stone_group:
                for (row, col) in groups[idx]:
                    # Update black/white board
                    if self.board[row][col] == 1:
                        self.black[row][col] = 0

                    else:
                        self.white[row][col] = 0

                    # Update empty board
                    self.empty[row][col] = 1

                    # Remove stones on the board
                    self.board[row][col] = 0

                    # Set turns since of the dead stone intersection to 0
                    self.turns_since[row][col] = 0

    def update_turns_since(self):
        for row in range(19):
            for col in range(19):
                # +1 to every stone on the board
                if self.board[row][col] != 0:
                    self.turns_since[row][col] += 1

    def update_liberties(self):
        # Refresh to 0
        self.liberties = construct_board(0)

        groups = self.find_groups()
        for group in groups:
            liberty = self.find_group_liberty(group)
            for (row, col) in group:
                self.liberties[row][col] = liberty

    def make_move(self, row, col):
        # Update move to board
        self.board[row][col] = self.turn

        if self.turn == 1:
            self.black[row][col] = 1
        else:
            self.white[row][col] = 1

        self.empty[row][col] = 0

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
        self.hashes.append(str(self.board))
        if len(self.hashes) > 3:
            self.hashes.pop(0)


def construct_board(val):
    return [[val for __ in range(19)] for __ in range(19)]


def within_board(position):
    row, col = position
    return row >= 0 and row < 19 and col >= 0 and col < 19


def find_adjacent_cells(row, col):
    adjacent = []
    for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        if within_board((row + dx, col + dy)):
            adjacent.append((row + dx, col + dy))

    return adjacent


def find_stone_group(row, col, groups):
    for idx, group in enumerate(groups):
        if (row, col) in group:
            return idx


def diagonals(position):
    (x, y) = position
    return filter(
        within_board, [(x - 1, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1)]
    )
