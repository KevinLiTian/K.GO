import random

import torch
import numpy as np

BOARD_TRANSFORMATIONS = [
    lambda feature: feature,
    lambda feature: np.rot90(feature, 1),
    lambda feature: np.rot90(feature, 2),
    lambda feature: np.rot90(feature, 3),
    lambda feature: np.fliplr(feature),
    lambda feature: np.flipud(feature),
    lambda feature: np.transpose(feature),
    lambda feature: np.fliplr(np.rot90(feature, 1)),
]


def one_hot_encode(value, transform):
    encode = np.zeros((1, 361), dtype=np.float32)
    encode[0, int(value)] = 1
    encode = encode.reshape((19, 19))
    encode = transform(encode).reshape((1, 361))
    return np.argmax(encode, axis=1)


def generate_batch(board_states, moves):
    states, actions = [], []
    for board_state, move in zip(board_states, moves):
        rand_int = random.randint(0, 7)
        transform = BOARD_TRANSFORMATIONS[rand_int]
        states.append(np.array([transform(plane) for plane in board_state]))
        actions.append(one_hot_encode(move, transform))

    states = torch.from_numpy(np.stack(states, axis=0))
    actions = torch.from_numpy(np.concatenate(actions, axis=0))
    return states, actions


def parse_file(
    file_path,
    remaining_boards,
    remaining_moves,
    remaining_results,
    batch_size,
    features,
):
    """Parse a .npz file into batches of tensors"""
    data = np.load(file_path)
    board_states = data["board_states"]
    moves = data["moves"]
    results = data["results"]

    if len(remaining_boards) != 0:
        board_states = np.concatenate([board_states, remaining_boards], axis=0)
        moves = np.concatenate([moves, remaining_moves], axis=0)
        results = np.concatenate([results, remaining_results], axis=0)

    perm = np.random.permutation(board_states.shape[0])
    board_states, moves, results = board_states[perm], moves[perm], results[perm]

    board_states_batch = []
    moves_batch = []
    results_batch = []

    num_batches = board_states.shape[0] // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        states, actions = generate_batch(
            board_states[start_idx:end_idx, :features], moves[start_idx:end_idx]
        )
        board_states_batch.append(states)
        moves_batch.append(actions)
        results_batch.append(results[start_idx:end_idx])

    if board_states.shape[0] % batch_size != 0:
        remaining_boards = board_states[num_batches * batch_size :]
        remaining_moves = moves[num_batches * batch_size :]
        remaining_results = results[num_batches * batch_size :]
    else:
        remaining_boards, remaining_moves, remaining_results = [], [], []

    return (
        board_states_batch,
        moves_batch,
        results_batch,
        remaining_boards,
        remaining_moves,
        remaining_results,
    )
