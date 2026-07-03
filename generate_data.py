"""Generate imitation learning training data using the heuristic as teacher."""

import random
import numpy as np
from logic import choose_move_heuristic

WIDTH, HEIGHT = 11, 11
MOVE_TO_IDX = {"up": 0, "down": 1, "left": 2, "right": 3}
DIRS = [(0, 1), (0, -1), (-1, 0), (1, 0)]


def _random_body(occupied, min_len=3, max_len=10):
    length = random.randint(min_len, max_len)
    for _ in range(200):
        x = random.randint(1, WIDTH - 2)
        y = random.randint(1, HEIGHT - 2)
        if (x, y) not in occupied:
            break
    else:
        return None

    body = [(x, y)]
    occupied.add((x, y))

    for _ in range(length - 1):
        cx, cy = body[-1]
        cands = [
            (cx + dx, cy + dy)
            for dx, dy in DIRS
            if 0 <= cx + dx < WIDTH and 0 <= cy + dy < HEIGHT
            and (cx + dx, cy + dy) not in occupied
        ]
        if not cands:
            break
        nxt = random.choice(cands)
        body.append(nxt)
        occupied.add(nxt)

    return body if len(body) >= 3 else None


def _make_snake(snake_id, body, health=None):
    if health is None:
        health = random.randint(20, 100)
    return {
        "id": snake_id,
        "head": {"x": body[0][0], "y": body[0][1]},
        "body": [{"x": x, "y": y} for x, y in body],
        "length": len(body),
        "health": health,
    }


def _random_state():
    occupied = set()

    my_body = _random_body(occupied, min_len=3, max_len=10)
    if my_body is None:
        return None

    snakes = [_make_snake("me", my_body, health=random.randint(10, 100))]

    if random.random() > 0.25:
        en_body = _random_body(occupied, min_len=3, max_len=12)
        if en_body:
            snakes.append(_make_snake("enemy", en_body))

    food = []
    for _ in range(random.randint(1, 4)):
        for _ in range(30):
            fx, fy = random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1)
            if (fx, fy) not in occupied:
                food.append({"x": fx, "y": fy})
                occupied.add((fx, fy))
                break

    return {
        "turn": random.randint(0, 200),
        "board": {"width": WIDTH, "height": HEIGHT, "food": food, "snakes": snakes},
        "you": snakes[0],
    }


def encode_board(state):
    board = state["board"]
    you = state["you"]
    w, h = board["width"], board["height"]
    my_id = you["id"]
    my_len = you["length"]

    tensor = np.zeros((6, h, w), dtype=np.float32)

    hx, hy = you["head"]["x"], you["head"]["y"]
    tensor[0, hy, hx] = 1.0

    for seg in you["body"][1:]:
        tensor[1, seg["y"], seg["x"]] = 1.0

    for snake in board["snakes"]:
        if snake["id"] == my_id:
            continue
        ex, ey = snake["head"]["x"], snake["head"]["y"]
        tensor[2, ey, ex] = 1.0
        for seg in snake["body"][1:]:
            tensor[3, seg["y"], seg["x"]] = 1.0
        if snake["length"] >= my_len:
            for dx, dy in DIRS:
                nx, ny = ex + dx, ey + dy
                if 0 <= nx < w and 0 <= ny < h:
                    tensor[5, ny, nx] = 1.0

    for f in board["food"]:
        tensor[4, f["y"], f["x"]] = 1.0

    return tensor


def generate(n=50_000):
    X, y = [], []
    attempts = 0
    while len(X) < n:
        attempts += 1
        state = _random_state()
        if state is None:
            continue
        move = choose_move_heuristic(state)
        X.append(encode_board(state))
        y.append(MOVE_TO_IDX[move])
        if len(X) % 5000 == 0:
            print(f"  {len(X)}/{n}  (attempts={attempts})")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


if __name__ == "__main__":
    print("Generating 50 000 training states...")
    X, y = generate(50_000)
    np.savez("data.npz", X=X, y=y)
    print(f"Saved data.npz  X={X.shape}  y={y.shape}")
    dist = {m: int((y == i).sum()) for m, i in MOVE_TO_IDX.items()}
    print("Label distribution:", dist)
