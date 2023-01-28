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

    def display(self):
        for row in range(19):
            for col in range(19):
                print(self.board[row][col], end="")
            print()
        print()

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


def construct_board(val):
    return [[val for __ in range(19)] for __ in range(19)]


def within_board(row, col):
    return row >= 0 and row < 19 and col >= 0 and col < 19


def find_adjacent_cells(row, col):
    adjacent = []
    for (dx, dy) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        if within_board(row + dx, col + dy):
            adjacent.append((row + dx, col + dy))

    return adjacent


def find_stone_group(row, col, groups):
    for idx, group in enumerate(groups):
        if (row, col) in group:
            return idx
