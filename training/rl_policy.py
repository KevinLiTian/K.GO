import torch
from torch.nn import NLLLoss

import Go.GameState as go
from training.process import create_board_state
from AI.policy_player import ProbabilisticPolicyPlayer


def make_learning_pair(st, mv):
    row, col = mv
    st_tensor = create_board_state(st)[:48]
    st_tensor = torch.from_numpy(st_tensor)
    move = 19 * row + col
    return st_tensor, move


def run_n_games(optimizer, lr, learner, opponent, num_games):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states = [go.GameState() for _ in range(num_games)]
    learner_net = learner.policy

    state_tensors = [[] for _ in range(num_games)]
    move_tensors = [[] for _ in range(num_games)]

    learner_won = [None] * num_games

    learner_color = [go.BLACK if i % 2 == 0 else go.WHITE for i in range(num_games)]
    odd_states = states[1::2]  # Games where learner is white
    moves = opponent.get_moves(odd_states)
    for st, mv in zip(odd_states, moves):
        st.do_move(mv)

    current = learner
    other = opponent
    idxs_to_unfinished_states = {i: states[i] for i in range(num_games)}
    while len(idxs_to_unfinished_states) > 0:
        moves = current.get_moves(idxs_to_unfinished_states.values())
        just_finished = []
        # Do each move to each state in order.
        for (idx, state), mv in zip(idxs_to_unfinished_states.items(), moves):
            # Order is important here. We must get the training pair on the unmodified state before
            # updating it with do_move.
            is_learnable = current is learner and mv is not go.PASS_MOVE
            if is_learnable:
                (st_tensor, move) = make_learning_pair(state, mv)
                state_tensors[idx].append(st_tensor)
                move_tensors[idx].append(move)
            state.do_move(mv)
            if state.is_end_of_game:
                learner_won[idx] = state.get_winner() == learner_color[idx]
                just_finished.append(idx)

        for idx in just_finished:
            del idxs_to_unfinished_states[idx]

        current, other = other, current

    criterion = NLLLoss()
    for st_tensor, mv_tensor, won in zip(state_tensors, move_tensors, learner_won):
        optimizer.param_groups[0]["lr"] = abs(lr) * (1 if won else -1)
        st_tensor = torch.stack(st_tensor).to(device)
        mv_tensor = torch.Tensor(mv_tensor).to(device)

        output = learner_net(st_tensor)
        loss = criterion(output, mv_tensor.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    wins = sum(state.get_winner() == pc for (state, pc) in zip(states, learner_color))
    print(f"Learner win rate: {float(wins) / num_games}")


def train():
    player = ProbabilisticPolicyPlayer("./networks/conv192/best.pth")
    opponent = ProbabilisticPolicyPlayer("./networks/conv192/best.pth")
    optimizer = torch.optim.SGD(player.policy.parameters(), lr=0.003)

    for idx in range(125):
        run_n_games(optimizer, 0.003, player, opponent, 4)
        print(f"Iterations finished: {idx}/125")

    checkpoint = {"model_state_dict": player.policy.state_dict()}
    torch.save(checkpoint, "./rl_pool/0.pth")
