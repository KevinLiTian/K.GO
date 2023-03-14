import numpy as np

import Go.GameState as go


def get_board(state: go.GameState):
    """A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
    always refers to the current player and plane 1 to the opponent
    """
    planes = np.zeros((3, state.size, state.size), dtype=np.float32)
    planes[0, :, :] = state.board == state.current_player  # own stone
    planes[1, :, :] = state.board == -state.current_player  # opponent stone
    planes[2, :, :] = state.board == go.EMPTY  # empty space
    return planes


def get_turns_since(state: go.GameState, maximum=8):
    """A feature encoding the age of the stone at each location up to 'maximum'

    Note:
    - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
    - EMPTY locations are all-zero features
    """
    planes = np.zeros((maximum, state.size, state.size), dtype=np.float32)
    for x in range(state.size):
        for y in range(state.size):
            if state.stone_ages[x][y] >= 0:
                planes[min(state.stone_ages[x][y], maximum - 1), x, y] = 1
    return planes


def get_liberties(state: go.GameState, maximum=8):
    """A feature encoding the number of liberties of the group connected to the stone at
    each location

    Note:
    - there is no zero-liberties plane; the 0th plane indicates groups in atari
    - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
    - EMPTY locations are all-zero features
    """
    planes = np.zeros((maximum, state.size, state.size), dtype=np.float32)
    for i in range(maximum):
        # single liberties in plane zero (groups won't have zero), double
        # liberties in plane one, etc
        planes[i, state.liberty_counts == i + 1] = 1
    # the "maximum-or-more" case on the backmost plane
    planes[maximum - 1, state.liberty_counts >= maximum] = 1
    return planes


def get_capture_size(state: go.GameState, maximum=8):
    """A feature encoding the number of opponent stones that would be captured by
    playing at each location, up to 'maximum'

    Note:
    - we currently *do* treat the 0th plane as "capturing zero stones"
    - the [maximum-1] plane is used for any capturable group of size
      greater than or equal to maximum-1
    - the 0th plane is used for legal moves that would not result in capture
    - illegal move locations are all-zero features

    """
    planes = np.zeros((maximum, state.size, state.size), dtype=np.float32)
    for (x, y) in state.get_legal_moves():
        # multiple disconnected groups may be captured. hence we loop over
        # groups and count sizes if captured.
        n_captured = 0
        for neighbor_group in state.get_groups_around((x, y)):
            # if the neighboring group is opponent stones and they have
            # one liberty, it must be (x,y) and we are capturing them
            # (note suicide and ko are not an issue because they are not
            # legal moves)
            (gx, gy) = next(iter(neighbor_group))
            if (state.liberty_counts[gx][gy] == 1) and (
                state.board[gx, gy] != state.current_player
            ):
                n_captured += len(state.group_sets[gx][gy])
        planes[min(n_captured, maximum - 1), x, y] = 1
    return planes


def get_self_atari_size(state: go.GameState, maximum=8):
    """A feature encoding the size of the own-stone group that is put into atari by
    playing at a location

    """
    planes = np.zeros((maximum, state.size, state.size), dtype=np.float32)

    for (x, y) in state.get_legal_moves():
        # make a copy of the liberty/group sets at (x,y) so we can manipulate them
        lib_set_after = set(state.liberty_sets[x][y])
        group_set_after = set()
        group_set_after.add((x, y))
        captured_stones = set()
        for neighbor_group in state.get_groups_around((x, y)):
            # if the neighboring group is of the same color as the current player
            # then playing here will connect this stone to that group
            (gx, gy) = next(iter(neighbor_group))
            if state.board[gx, gy] == state.current_player:
                lib_set_after |= state.liberty_sets[gx][gy]
                group_set_after |= state.group_sets[gx][gy]
            # if instead neighboring group is opponent *and about to be captured*
            # then we might gain new liberties
            elif state.liberty_counts[gx][gy] == 1:
                captured_stones |= state.group_sets[gx][gy]
        # add captured stones to liberties if they are neighboring the 'group_set_after'
        # i.e. if they will become liberties once capture is resolved
        if len(captured_stones) > 0:
            for (gx, gy) in group_set_after:
                # intersection of group's neighbors and captured stones will become liberties
                lib_set_after |= set(state._neighbors((gx, gy))) & captured_stones
        if (x, y) in lib_set_after:
            lib_set_after.remove((x, y))
        # check if this move resulted in atari
        if len(lib_set_after) == 1:
            group_size = len(group_set_after)
            # 0th plane used for size=1, so group_size-1 is the index
            planes[min(group_size - 1, maximum - 1), x, y] = 1
    return planes


def get_liberties_after(state: go.GameState, maximum=8):
    """A feature encoding what the number of liberties *would be* of the group connected to
    the stone *if* played at a location

    Note:
    - there is no zero-liberties plane; the 0th plane indicates groups in atari
    - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
    - illegal move locations are all-zero features
    """
    planes = np.zeros((maximum, state.size, state.size), dtype=np.float32)
    # note - left as all zeros if not a legal move
    for (x, y) in state.get_legal_moves():
        # make a copy of the set of liberties at (x,y) so we can add to it
        lib_set_after = set(state.liberty_sets[x][y])
        group_set_after = set()
        group_set_after.add((x, y))
        captured_stones = set()
        for neighbor_group in state.get_groups_around((x, y)):
            # if the neighboring group is of the same color as the current player
            # then playing here will connect this stone to that group and
            # therefore add in all that group's liberties
            (gx, gy) = next(iter(neighbor_group))
            if state.board[gx, gy] == state.current_player:
                lib_set_after |= state.liberty_sets[gx][gy]
                group_set_after |= state.group_sets[gx][gy]
            # if instead neighboring group is opponent *and about to be captured*
            # then we might gain new liberties
            elif state.liberty_counts[gx][gy] == 1:
                captured_stones |= state.group_sets[gx][gy]
        # add captured stones to liberties if they are neighboring the 'group_set_after'
        # i.e. if they will become liberties once capture is resolved
        if len(captured_stones) > 0:
            for (gx, gy) in group_set_after:
                # intersection of group's neighbors and captured stones will become liberties
                lib_set_after |= set(state._neighbors((gx, gy))) & captured_stones
        # (x,y) itself may have made its way back in, but shouldn't count
        # since it's clearly not a liberty after playing there
        if (x, y) in lib_set_after:
            lib_set_after.remove((x, y))
        planes[min(maximum - 1, len(lib_set_after) - 1), x, y] = 1
    return planes


def get_ladder_capture(state: go.GameState):
    """A feature wrapping GameState.is_ladder_capture()."""
    feature = np.zeros((1, state.size, state.size), dtype=np.float32)
    for (x, y) in state.get_legal_moves():
        feature[0, x, y] = state.is_ladder_capture((x, y))
    return feature


def get_ladder_escape(state: go.GameState):
    """A feature wrapping GameState.is_ladder_escape()."""
    feature = np.zeros((1, state.size, state.size), dtype=np.float32)
    for (x, y) in state.get_legal_moves():
        feature[0, x, y] = state.is_ladder_escape((x, y))
    return feature


def get_sensibleness(state: go.GameState):
    """A move is 'sensible' if it is legal and if it does not fill the current_player's own eye"""
    feature = np.zeros((1, state.size, state.size), dtype=np.float32)
    for (x, y) in state.get_legal_moves(include_eyes=False):
        feature[0, x, y] = 1
    return feature


def get_black_white(state: go.GameState, player):
    """A feature encoding WHITE BLACK on separate planes, but plane 0
    always refers to the input player and plane 1 to the opponent
    """
    planes = np.zeros((2, state.size, state.size), dtype=np.float32)
    planes[0, :, :] = state.board == player  # own stone
    planes[1, :, :] = state.board == -player  # opponent stone
    return planes


def get_board_history(state: go.GameState):
    """Get 19 x 19 x 17 board history, proposed in 'Mastering the Game of Go without Human Knowledge'"""
    planes = []

    # Enough historys
    if len(state.history) >= 8:
        new_gs = go.GameState()

        # Place handicaps
        if len(state.handicaps) != 0:
            new_gs.place_handicaps(state.handicaps)

        # Recreate up to last 8 moves
        for action in state.history[:-8]:
            new_gs.do_move(action)

        # Generate state for last 8 moves
        for action in state.history[-8:]:
            new_gs.do_move(action)
            planes.insert(0, get_black_white(new_gs, player=state.current_player))

    # Not enough, pad with all zeros
    else:
        new_gs = go.GameState()

        # Place handicaps
        if len(state.handicaps) != 0:
            new_gs.place_handicaps(state.handicaps)

        for action in state.history:
            new_gs.do_move(action)
            planes.insert(0, get_black_white(new_gs, player=state.current_player))

        num_missing = 8 - len(planes)
        if num_missing != 0:
            planes.append(
                np.zeros((num_missing * 2, state.size, state.size), dtype=np.float32)
            )

    # Player's colour
    planes.append(
        np.ones((1, state.size, state.size), dtype=np.float32)
        * (state.current_player == go.BLACK)
    )
    return np.concatenate(planes, axis=0)
