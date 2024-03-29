import os

import numpy as np
from sgfmill import sgf

from Go.features import (
    get_board,
    get_turns_since,
    get_liberties,
    get_capture_size,
    get_self_atari_size,
    get_liberties_after,
    get_ladder_capture,
    get_ladder_escape,
    get_sensibleness,
    get_board_history,
)
import Go.GameState as go
from Go.GameState import GameState


def parse_game(game, ZERO=False):
    with open(game, "rb") as file:
        sgf_object = sgf.Sgf_game.from_bytes(file.read())

    board_states = []
    moves = []
    results = []

    # Place setup stones
    root = sgf_object.get_root()
    setup = root.get_setup_stones()[0]
    gs = GameState()
    gs.place_handicaps(setup)

    # Set result label
    winner = sgf_object.get_winner()
    if winner == "b":
        result = 1
    elif winner == "w":
        result = -1
    else:
        raise Exception(f"{game} has no winner")

    main_sequence = sgf_object.get_main_sequence()
    for idx, node in enumerate(main_sequence):
        color, move = node.get_move()

        # Root node does not have colour
        if color is None:
            continue

        # A player passes, indicating game might be over
        if move is None and idx != 0:
            break

        # Add board state
        if not ZERO:
            board_states.append(create_board_state(gs))
        else:
            board_states.append(get_board_history(gs))

        # Make move
        color = go.BLACK if color == "b" else go.WHITE
        gs.do_move(move, color)

        # Add move
        move_encode = np.zeros((19, 19))
        row, col = move
        move_encode[row, col] = 1
        moves.append(np.float32(np.argmax(move_encode.ravel())))

        # Add winner
        results.append(result)

    return board_states, moves, results


def create_board_state(state: GameState):
    """
    49 feature planes
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
    - Player's colour (1)
    """

    # Stone colour each (1x19x19)
    stones = get_board(state)

    # Const 1s (1x19x19)
    const_one = np.ones((1, state.size, state.size), dtype=np.float32)

    # 8 channel feature planes (8x19x19)
    turns_since = get_turns_since(state)
    liberties = get_liberties(state)
    capture_size = get_capture_size(state)
    self_atari_size = get_self_atari_size(state)
    liberties_after = get_liberties_after(state)

    # Ladders (2x19x19)
    ladder_capture = get_ladder_capture(state)
    ladder_escape = get_ladder_escape(state)

    # Sensibleness (1x19x19)
    sensibleness = get_sensibleness(state)

    # Const 0s (1x19x19)
    const_zero = np.zeros((1, state.size, state.size), dtype=np.float32)

    # Player's colour (Used in value network only)
    player_colour = np.ones((1, state.size, state.size), dtype=np.float32) * (
        state.current_player == go.BLACK
    )

    return np.concatenate(
        [
            stones,
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
            player_colour,
        ],
        axis=0,
    )


def process(START, END):
    # 225833 KGS games by
    # - One of the players is 7 dan or stronger
    # - Both players 6 dan
    game_files = []

    # Read from data folder
    for dirpath, __, filenames in os.walk("./data"):
        filenames.sort()
        for file in filenames:
            if file.endswith("sgf"):
                file_path = os.path.join(dirpath, file)
                game_files.append(file_path)

    # Process
    for idx in range(START, END, 200):
        # Save to disk setup
        board_states_storage = []
        moves_storage = []
        results_storage = []

        start = idx
        end = idx + 200
        for game_id in range(start, end):
            # Get data from game
            board_states, moves, results = parse_game(game_files[game_id])

            # Store to temporary storage
            board_states_storage.extend(board_states)
            moves_storage.extend(moves)
            results = np.array(results, dtype=np.float32)
            results_storage.extend(results)

            # Progress report
            print(f"Game {game_id} processed")

        # Save to .npz
        path = f"./dataset/{start}_{end}.npz"
        np.savez_compressed(
            path,
            board_states=board_states_storage,
            moves=moves_storage,
            results=results_storage,
        )


def process_zero(START, END):
    # 225833 KGS games by
    # - One of the players is 7 dan or stronger
    # - Both players 6 dan
    game_files = []

    # Read from data folder
    for dirpath, __, filenames in os.walk("./data"):
        filenames.sort()
        for file in filenames:
            if file.endswith("sgf"):
                file_path = os.path.join(dirpath, file)
                game_files.append(file_path)

    # Process
    for idx in range(START, END, 400):
        # Save to disk setup
        board_states_storage = []
        moves_storage = []
        results_storage = []

        start = idx
        end = idx + 400
        for game_id in range(start, end):
            # Get data from game
            board_states, moves, results = parse_game(game_files[game_id], ZERO=True)

            # Store to temporary storage
            board_states_storage.extend(board_states)
            moves_storage.extend(moves)
            results_storage.extend(results)

            # Progress report
            print(f"Game {game_id} processed")

        # Save to .npz
        path = f"./zero_dataset/{start}_{end}.npz"
        np.savez_compressed(
            path,
            board_states=board_states_storage,
            moves=moves_storage,
            results=results_storage,
        )
