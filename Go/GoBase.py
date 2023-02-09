from Go.util import find_adjacent_cells, find_stone_group


class GoBase:
    def display(self):
        for row in range(19):
            for col in range(19):
                print(self.board[row, col], end="")
            print()
        print()

    def is_legal(self, row, col):
        if self.board[row, col] != 0:
            return False
        if self.is_suicide(row, col):
            return False
        if self.ko == (row, col):
            return False
        return True

    def is_suicide(self, row, col):
        game_copy = MiniGo(self)
        game_copy.make_move(row, col)

        groups = game_copy.find_groups()
        cur_stone_group = find_stone_group(row, col, groups)
        dead_groups = game_copy.find_dead_groups(groups)

        # Only cur stone group is dead, suicide
        if len(dead_groups) == 1 and cur_stone_group in dead_groups:
            return True

        return False

    def find_groups(self):
        groups = []
        visited = set()

        # Check each cell of board
        for row in range(19):
            for col in range(19):
                # If there's an unvisited stone
                if self.board[row, col] != 0 and not (row, col) in visited:
                    visited.add((row, col))
                    group = [(row, col)]

                    # DFS
                    frontier = [(row, col)]
                    while len(frontier) != 0:
                        (x, y) = frontier.pop()
                        for (r, c) in find_adjacent_cells(x, y):
                            # Same colour and not visited
                            if (
                                self.board[r, c] == self.board[x, y]
                                and not (r, c) in visited
                            ):
                                group.append((r, c))
                                visited.add((r, c))
                                frontier.append((r, c))

                    groups.append(group)

        return groups

    def find_dead_groups(self, groups):
        dead_groups = []
        for idx, group in enumerate(groups):
            if self.find_group_liberty(group) == 0:
                dead_groups.append(idx)

        return dead_groups

    def find_group_liberty(self, group):
        liberty = 0
        visited = set()

        for (row, col) in group:
            for (x, y) in find_adjacent_cells(row, col):
                if self.board[x, y] == 0 and not (x, y) in visited:
                    visited.add((x, y))
                    liberty += 1

        return liberty

    def remove_dead_groups(self, groups, cur_stone_group, dead_groups):
        captured = 0
        for idx in dead_groups:
            if idx != cur_stone_group:
                for (row, col) in groups[idx]:
                    self.board[row, col] = 0
                    captured += 1

        return captured

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

        # if we haven't found a capture by a certain number of moves,
        # assume it's not a capture
        if remaining_attempts <= 0:
            return False

        hunter_player = self.turn
        prey_player = 2 if self.turn == 1 else 1

        if prey is None:
            # default case is to check all adjacent prey_player groups that
            # have 2 liberties
            potential_prey = []
            for group in self.get_groups_around(row, col):
                x, y = group[0]
                if (
                    self.board[x, y] == prey_player
                    and self.find_group_liberty(group) == 2
                ):
                    potential_prey.append((x, y))

        else:
            # we are checking a specific group (called from is_ladder_escape)
            potential_prey = [prey]

        for (prey_x, prey_y) in potential_prey:
            # attempt to capture the group at prey_x, prey_y in a ladder
            game_copy = MiniGo(self)
            game_copy.make_move(row, col)

            # we only want to check a limited set of possible escape moves:
            # - extensions from the remaining liberty of the prey group.
            # - captures of enemy groups adjacent to the prey group.
            group, possible_escapes = game_copy.get_stone_lib_pos(prey_x, prey_y)

            # Check if any hunter groups adjacent to the prey groups
            # are in atari.  Capturing these groups are potential escapes.
            for x, y in group:
                for (nx, ny) in find_adjacent_cells(x, y):
                    if game_copy.board[nx, ny] == hunter_player:
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
                    self.board[x, y] == prey_player
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
            game_copy = MiniGo(self)
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


class MiniGo(GoBase):
    """
    This is a minimal version of Go game states, used for copying partial board states
    to replace the use of deepcopy. Optimizing the runtime of copying a board state.
    """

    def __init__(self, game):
        self.board = game.board.copy()
        self.turn = game.turn

        self.ko = game.ko

    def make_move(self, row, col):
        # Update move to board
        self.board[row, col] = self.turn

        # Remove previous KO position
        self.ko = None

        # Switch turns
        self.turn = 2 if self.turn == 1 else 1

        # Find groups, current stones's group and dead groups
        groups = self.find_groups()
        cur_stone_group = find_stone_group(row, col, groups)
        dead_groups = self.find_dead_groups(groups)

        # Remove dead groups from board
        num_captured = self.remove_dead_groups(groups, cur_stone_group, dead_groups)

        # Check for KO
        if num_captured == 1:
            # If played stone is not captured, cannot be KO
            if cur_stone_group in dead_groups:
                # 2 dead groups
                # - Played stone
                # - captured stone

                # Remove played stone group, only captured remains
                dead_groups.remove(cur_stone_group)
                captured_idx = dead_groups[0]

                # Get dead stone position
                dead_stone = groups[captured_idx][0]

                # Get played stone's group positions
                cur_group = groups[cur_stone_group]

                # Only if the current stone group only has 1 stone and only has one liberty
                # is considered a KO situation
                if len(cur_group) == 1 and self.find_group_liberty(cur_group) == 1:
                    self.ko = dead_stone
