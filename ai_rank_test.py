from AI.policy_player import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
import Go.GameState as go

win1 = 0
win2 = 0
for i in range(50):
    i += 1
    print(i)

    game = go.GameState()
    turn = go.BLACK
    checkpoint1 = "./networks/conv192/best.pth"
    checkpoint2 = "./networks/conv192/checkpoint_0.pth"
    player1 = GreedyPolicyPlayer(checkpoint1, game)
    player2 = ProbabilisticPolicyPlayer(checkpoint2, game)

    if i % 2 == 0:
        turn1 = go.BLACK
        turn2 = go.WHITE
    else:
        turn1 = go.WHITE
        turn2 = go.BLACK

    while not game.is_end_of_game:
        if turn1 == turn:
            move = player1.get_move()
            game.do_move(move)
            turn = -turn

        elif turn2 == turn:
            move = player2.get_move()
            game.do_move(move)
            turn = -turn

    winner = game.get_winner()
    if winner == turn1:
        win1 += 1
    elif winner == turn2:
        win2 += 1

print(f"Player 1 wins: {win1}/50")
print(f"Player 2 wins: {win2}/50")
