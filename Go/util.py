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
