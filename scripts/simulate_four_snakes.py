"""Run a local four-snake deterministic scrimmage.

This does not need the Battlesnake CLI or Flask. It calls the move engine
directly and applies Battlesnake-like movement, food, health, body collision,
and head-to-head rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk_ai_challenge.engine import choose_move


MovePolicy = Callable[[dict], str]

MOVES = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class ScrimmageConfig:
    width: int = 11
    height: int = 11
    max_turns: int = 80
    food_target: int = 6
    seed: int = 42
    move_time_limit_ms: int = 35


def main() -> None:
    config = ScrimmageConfig()
    random.seed(config.seed)
    state = initial_state(config)
    policies = {
        snake["id"]: lambda game_state, limit=config.move_time_limit_ms: choose_move(
            game_state, time_limit_ms=limit
        )
        for snake in state["board"]["snakes"]
    }

    print_board(state)
    for turn in range(config.max_turns):
        if len(state["board"]["snakes"]) <= 1:
            break
        moves = choose_all_moves(state, policies)
        state = advance(state, moves, config)
        alive = ", ".join(s["id"] for s in state["board"]["snakes"]) or "none"
        print(f"turn={turn + 1:03d} moves={moves} alive={alive}")

    print_board(state)
    if state["board"]["snakes"]:
        winner = max(state["board"]["snakes"], key=lambda s: (s["length"], s["health"]))
        print(
            "winner="
            f"{winner['id']} length={winner['length']} health={winner['health']} "
            f"turns={state['turn']}"
        )
    else:
        print(f"winner=none turns={state['turn']}")


def initial_state(config: ScrimmageConfig) -> dict:
    snakes = [
        snake("alpha", [(1, 1), (1, 0), (0, 0)]),
        snake("bravo", [(9, 9), (9, 10), (10, 10)]),
        snake("charlie", [(1, 9), (0, 9), (0, 10)]),
        snake("delta", [(9, 1), (10, 1), (10, 0)]),
    ]
    state = {
        "game": {"id": "local-four-snake-scrimmage"},
        "turn": 0,
        "board": {
            "height": config.height,
            "width": config.width,
            "food": [{"x": 5, "y": 5}, {"x": 3, "y": 5}, {"x": 7, "y": 5}],
            "snakes": snakes,
        },
        "you": snakes[0],
    }
    refill_food(state, config)
    return state


def snake(snake_id: str, body: list[tuple[int, int]]) -> dict:
    return {
        "id": snake_id,
        "name": snake_id,
        "health": 100,
        "length": len(body),
        "head": point(body[0]),
        "body": [point(part) for part in body],
    }


def point(part: tuple[int, int]) -> dict:
    return {"x": part[0], "y": part[1]}


def choose_all_moves(state: dict, policies: dict[str, MovePolicy]) -> dict[str, str]:
    moves = {}
    for current in state["board"]["snakes"]:
        perspective = clone_state(state)
        perspective["you"] = current
        moves[current["id"]] = policies[current["id"]](perspective)
    return moves


def advance(state: dict, moves: dict[str, str], config: ScrimmageConfig) -> dict:
    moved = []
    eaten = set()
    food = {(f["x"], f["y"]) for f in state["board"]["food"]}
    head_counts: dict[tuple[int, int], int] = {}

    for current in state["board"]["snakes"]:
        direction = moves.get(current["id"], "up")
        dx, dy = MOVES.get(direction, MOVES["up"])
        old_head = current["body"][0]
        new_head = (old_head["x"] + dx, old_head["y"] + dy)
        ate = new_head in food
        new_body = [point(new_head)] + current["body"][:]
        if not ate:
            new_body.pop()
        next_snake = {
            **current,
            "health": 100 if ate else current["health"] - 1,
            "length": len(new_body),
            "head": new_body[0],
            "body": new_body,
        }
        moved.append(next_snake)
        if ate:
            eaten.add(new_head)
        head_counts[new_head] = head_counts.get(new_head, 0) + 1

    occupied_body = {
        (part["x"], part["y"])
        for current in moved
        for part in current["body"][1:]
    }
    survivors = []
    for current in moved:
        head = (current["head"]["x"], current["head"]["y"])
        if current["health"] <= 0:
            continue
        if not in_bounds(head, config):
            continue
        if head in occupied_body:
            continue
        if head_counts[head] > 1:
            contenders = [
                other
                for other in moved
                if (other["head"]["x"], other["head"]["y"]) == head
            ]
            longest = max(other["length"] for other in contenders)
            longest_count = sum(1 for other in contenders if other["length"] == longest)
            if current["length"] < longest or longest_count > 1:
                continue
        survivors.append(current)

    next_state = {
        "game": state["game"],
        "turn": state["turn"] + 1,
        "board": {
            "height": config.height,
            "width": config.width,
            "food": [point(f) for f in sorted(food - eaten)],
            "snakes": survivors,
        },
        "you": survivors[0] if survivors else {},
    }
    refill_food(next_state, config)
    return next_state


def refill_food(state: dict, config: ScrimmageConfig) -> None:
    occupied = {
        (part["x"], part["y"])
        for current in state["board"]["snakes"]
        for part in current["body"]
    }
    food = {(f["x"], f["y"]) for f in state["board"]["food"]}
    free = [
        (x, y)
        for x in range(config.width)
        for y in range(config.height)
        if (x, y) not in occupied and (x, y) not in food
    ]
    random.shuffle(free)
    while len(food) < config.food_target and free:
        food.add(free.pop())
    state["board"]["food"] = [point(f) for f in sorted(food)]


def in_bounds(part: tuple[int, int], config: ScrimmageConfig) -> bool:
    return 0 <= part[0] < config.width and 0 <= part[1] < config.height


def clone_state(state: dict) -> dict:
    return {
        "game": dict(state["game"]),
        "turn": state["turn"],
        "board": {
            "height": state["board"]["height"],
            "width": state["board"]["width"],
            "food": [dict(f) for f in state["board"]["food"]],
            "snakes": [clone_snake(s) for s in state["board"]["snakes"]],
        },
        "you": clone_snake(state["you"]) if state.get("you") else {},
    }


def clone_snake(current: dict) -> dict:
    return {
        **current,
        "head": dict(current["head"]),
        "body": [dict(part) for part in current["body"]],
    }


def print_board(state: dict) -> None:
    width = state["board"]["width"]
    height = state["board"]["height"]
    cells = [["." for _x in range(width)] for _y in range(height)]
    for food in state["board"]["food"]:
        cells[food["y"]][food["x"]] = "*"
    for current in state["board"]["snakes"]:
        marker = current["id"][0].upper()
        for part in current["body"][1:]:
            cells[part["y"]][part["x"]] = marker.lower()
        head = current["head"]
        cells[head["y"]][head["x"]] = marker
    print(f"board turn={state['turn']}")
    for y in range(height - 1, -1, -1):
        print(" ".join(cells[y]))


if __name__ == "__main__":
    main()
