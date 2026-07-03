"""HTTP entrypoint for Battlesnake."""

from __future__ import annotations

import os

from flask import Flask, jsonify, request

from .engine import choose_move

app = Flask(__name__)


@app.get("/")
def index():
    return jsonify(
        {
            "apiversion": "1",
            "author": "risk-ai-challenge",
            "color": "#17A398",
            "head": "default",
            "tail": "default",
        }
    )


@app.post("/start")
def start():
    return jsonify({})


@app.post("/move")
def move():
    state = request.get_json(force=True)
    return jsonify({"move": choose_move(state)})


@app.post("/end")
def end():
    return jsonify({})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
