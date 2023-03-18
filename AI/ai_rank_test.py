from AI.policy_player import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
import Go.GameState as go

g_win = 0
p_win = 0
for i in range(100):
    i += 1
    print(i)

    game = go.GameState()
    turn = go.BLACK
    checkpoint = "./networks/conv192/best.pth"
    g_player = GreedyPolicyPlayer(checkpoint, game)
    p_player = ProbabilisticPolicyPlayer(checkpoint, game)

    if i % 2 == 0:
        g_turn = go.BLACK
        p_turn = go.WHITE
    else:
        g_turn = go.WHITE
        p_turn = go.BLACK

    while not game.is_end_of_game:
        if g_turn == turn:
            move = g_player.get_move()
            game.do_move(move)
            turn = -turn

        elif p_turn == turn:
            move = p_player.get_move()
            game.do_move(move)
            turn = -turn

    winner = game.get_winner()
    if winner == g_turn:
        g_win += 1
    elif winner == p_turn:
        p_win += 1

print(f"Greedy player wins: {g_win}/100")
print(f"Prob player wins: {p_win}/100")
