import os
import uuid
import requests

from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

from Go.GameState import GameState
from AI.policy_player import GreedyPolicyPlayer

# App Configuration
app = Flask(__name__, static_folder="frontend/build")
app.config["ENV"] = "development"
app.config["DEBUG"] = True
app.config["TESTING"] = True

CORS(app)


@app.route("/setup", methods=["POST"])
def setup():
    id = str(uuid.uuid4())
    api = "https://bohyly1o.api.sanity.io/v2021-06-07/data/mutate/production"
    headers = {
        "Authorization": "Bearer skK02BJIyy5V3e7G0CzSg2PYNFkklOfjOOnzYvf2winnQJSDVQ0wToE7a6UOHTfpVNqQfFN7trBVBO2mPTnVypooVqlOHh2kRY6VCEbar7nL75RI3JbSjsfIi0dKmcqWWAqQTluZgPOgMG3pVhwkuAgwRrG0f3ulqZfXJy6sewnd4is4KHqw"
    }
    payload = {
        "mutations": [{"createOrReplace": {"_id": id, "_type": "game", "history": []}}]
    }
    response = requests.post(api, json=payload, headers=headers)

    if response.status_code == 200:
        return jsonify({"id": id}), 200
    return response.json()


@app.route("/greedypolicy", methods=["POST"])
def greedy_policy():
    id = request.json["id"]
    client_move = request.json.get("move")

    api = f"https://bohyly1o.api.sanity.io/v2021-06-07/data/query/production?query=*[_id=='{id}']"
    headers = {
        "Authorization": "Bearer skK02BJIyy5V3e7G0CzSg2PYNFkklOfjOOnzYvf2winnQJSDVQ0wToE7a6UOHTfpVNqQfFN7trBVBO2mPTnVypooVqlOHh2kRY6VCEbar7nL75RI3JbSjsfIi0dKmcqWWAqQTluZgPOgMG3pVhwkuAgwRrG0f3ulqZfXJy6sewnd4is4KHqw"
    }
    response = requests.get(api, headers=headers)
    response = response.json()
    history = response["result"][0]["history"]

    board = GameState()
    for move in history:
        row, col = move["row"], move["col"]
        board.do_move((row, col))

    if client_move is not None:
        board.do_move((client_move[0], client_move[1]))

    player = GreedyPolicyPlayer(checkpoint="./networks/conv192/best.pth")
    player_move = player.get_move(board)

    if client_move is not None:
        history.append(
            {"_key": str(uuid.uuid4()), "row": client_move[0], "col": client_move[1]}
        )
    history.append(
        {"_key": str(uuid.uuid4()), "row": player_move[0], "col": player_move[1]}
    )

    api = "https://bohyly1o.api.sanity.io/v2021-06-07/data/mutate/production"
    payload = {"mutations": [{"patch": {"id": id, "set": {"history": history}}}]}
    response = requests.post(api, json=payload, headers=headers)

    if response.status_code == 200:
        return jsonify({"move": player_move}), 200
    return response.json()


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(threaded=True, port=5000)
