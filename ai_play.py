import json
import asyncio
import websockets

import Go.GameState as go
from AI.policy_player import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
from AI.mcts_player import MCTSPlayer


async def greedy_player(websocket):
    # Wait for the initial setting message
    ai_turn = await websocket.recv()
    if ai_turn == "b":
        ai_turn = go.BLACK
    elif ai_turn == "w":
        ai_turn = go.WHITE
    else:
        raise Exception("Unexpected Error: AI turn")

    # initialize game state
    game = go.GameState()
    turn = go.BLACK

    checkpoint = "./networks/conv192/best.pth"
    ai_player = GreedyPolicyPlayer(checkpoint, game)

    # continuously listen for incoming messages
    while True:
        if turn == ai_turn:
            move = ai_player.get_move()
            game.do_move(move)
            turn = -turn
            data = json.dumps(game.board.tolist())
            await websocket.send(data)

        move = await websocket.recv()
        move = json.loads(move)
        row, col = move[0], move[1]
        game.do_move((row, col))
        turn = -turn


async def prob_player(websocket):
    # Wait for the initial setting message
    ai_turn = await websocket.recv()
    if ai_turn == "b":
        ai_turn = go.BLACK
    elif ai_turn == "w":
        ai_turn = go.WHITE
    else:
        raise Exception("Unexpected Error: AI turn")

    # initialize game state
    game = go.GameState()
    turn = go.BLACK

    checkpoint = "./networks/conv192/best.pth"
    ai_player = ProbabilisticPolicyPlayer(checkpoint, game)

    # continuously listen for incoming messages
    while True:
        if turn == ai_turn:
            move = ai_player.get_move()
            game.do_move(move)
            turn = -turn
            data = json.dumps(game.board.tolist())
            await websocket.send(data)

        move = await websocket.recv()
        move = json.loads(move)
        row, col = move[0], move[1]
        game.do_move((row, col))
        turn = -turn


async def mcts_player(websocket):
    # Wait for the initial setting message
    ai_turn = await websocket.recv()
    if ai_turn == "b":
        ai_turn = go.BLACK
    elif ai_turn == "w":
        ai_turn = go.WHITE
    else:
        raise Exception("Unexpected Error: AI turn")

    # initialize game state
    game = go.GameState()
    turn = go.BLACK

    checkpoint = "./checkpoints/checkpoint_0_350.pth"
    ai_player = MCTSPlayer(checkpoint, game)

    # continuously listen for incoming messages
    while True:
        if turn == ai_turn:
            move, __ = ai_player.get_move()
            game.do_move(move)
            ai_player.update_root(move)
            turn = -turn
            data = json.dumps(game.board.tolist())
            await websocket.send(data)

        move = await websocket.recv()
        move = json.loads(move)
        row, col = move[0], move[1]
        game.do_move((row, col))
        turn = -turn


# create a WebSocket server and register the coroutines
prob_player_server = websockets.serve(prob_player, "localhost", 8080)
greedy_player_server = websockets.serve(greedy_player, "localhost", 8081)
mcts_player_server = websockets.serve(mcts_player, "localhost", 8082)

# start the event loop to listen for incoming connections
asyncio.get_event_loop().run_until_complete(greedy_player_server)
asyncio.get_event_loop().run_until_complete(prob_player_server)
asyncio.get_event_loop().run_until_complete(mcts_player_server)
print("Listening...")
asyncio.get_event_loop().run_forever()
