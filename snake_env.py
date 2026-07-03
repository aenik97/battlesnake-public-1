"""Gymnasium Battlesnake environment: our CNN agent vs mixed opponents.

Opponent setup (configurable):
  - enemy index 0: ONNX model (model_v1.onnx) if available, else heuristic
  - enemy index 1+: always heuristic
"""

import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from logic import choose_move_heuristic, DIRECTIONS

MOVE_NAMES = ["up", "down", "left", "right"]
MOVE_IDX   = {"up": 0, "down": 1, "left": 2, "right": 3}
DIRS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_v1.onnx")


class BattlesnakeEnv(gym.Env):
    WIDTH  = 11
    HEIGHT = 11
    MAX_TURNS = 200

    def __init__(self, n_enemies: int = None, opponent_model_path: str = _DEFAULT_MODEL_PATH):
        """
        n_enemies: fixed number of enemies (None = random 1-3 each episode).
        opponent_model_path: path to ONNX model used for enemy 0.
                             Pass None to use heuristic for all enemies.
        """
        super().__init__()
        self.n_enemies_fixed = n_enemies
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(6, self.HEIGHT, self.WIDTH),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        # Load ONNX opponent if available
        self._onnx_session   = None
        self._onnx_input_name = None
        if opponent_model_path and os.path.exists(opponent_model_path):
            try:
                import onnxruntime as rt
                self._onnx_session    = rt.InferenceSession(opponent_model_path)
                self._onnx_input_name = self._onnx_session.get_inputs()[0].name
                print(f"[env] Loaded ONNX opponent from {opponent_model_path}")
            except Exception as e:
                print(f"[env] Could not load ONNX opponent: {e}")

        self._reset_state()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()

        n_enemies = self.n_enemies_fixed if self.n_enemies_fixed is not None \
            else random.randint(1, 3)

        occupied = set()
        x, y = self._random_free(occupied, margin=2)
        self.my_body   = [(x, y), (x, y), (x, y)]
        self.my_health = 100
        occupied.add((x, y))

        self.enemies = []
        for _ in range(n_enemies):
            x, y = self._random_free(occupied, margin=2)
            self.enemies.append({"body": [(x, y), (x, y), (x, y)], "health": 100})
            occupied.add((x, y))

        self._spawn_food(occupied, n=min(n_enemies + 2, 5))
        return self._observe(), {}

    def step(self, action):
        my_move  = MOVE_NAMES[action]
        en_moves = self._get_enemy_moves()

        food_set = set(self.food)

        # Apply our move
        my_dx, my_dy = DIRECTIONS[my_move]
        my_new_head  = (self.my_body[0][0] + my_dx, self.my_body[0][1] + my_dy)
        my_ate = my_new_head in food_set
        if my_ate:
            self.my_body   = [my_new_head] + self.my_body
            self.my_health = 100
        else:
            self.my_body   = [my_new_head] + self.my_body[:-1]
            self.my_health -= 1

        # Apply enemy moves
        new_enemies = []
        for en, en_move in zip(self.enemies, en_moves):
            dx, dy   = DIRECTIONS[en_move]
            new_head = (en["body"][0][0] + dx, en["body"][0][1] + dy)
            ate      = new_head in food_set
            if ate:
                new_body   = [new_head] + en["body"]
                new_health = 100
            else:
                new_body   = [new_head] + en["body"][:-1]
                new_health = en["health"] - 1
            new_enemies.append({"body": new_body, "health": new_health})
        self.enemies = new_enemies

        # Remove eaten food, maybe spawn
        eaten    = {self.my_body[0]} | {e["body"][0] for e in self.enemies}
        self.food = [f for f in self.food if f not in eaten]
        if len(self.food) < 1 or random.random() < 0.05:
            self._spawn_food()

        self.turn += 1

        # Detect deaths
        all_bodies  = [self.my_body] + [e["body"] for e in self.enemies]
        my_dead     = self._is_dead(self.my_body, all_bodies[1:], self.my_health)
        surviving   = []
        for i, en in enumerate(self.enemies):
            others = [all_bodies[0]] + [all_bodies[j + 1] for j in range(len(self.enemies)) if j != i]
            if not self._is_dead(en["body"], others, en["health"]):
                surviving.append(en)
        en_all_dead   = len(surviving) == 0
        self.enemies  = surviving

        # Head-to-head with remaining enemies
        if not my_dead:
            for en in self.enemies:
                if self.my_body[0] == en["body"][0] and len(self.my_body) <= len(en["body"]):
                    my_dead = True
                    break

        # Reward
        if my_dead:
            reward = -1.0
        elif en_all_dead:
            reward = 5.0
        else:
            reward = 0.01 + (1.0 if my_ate else 0.0)

        terminated = my_dead or en_all_dead or self.turn >= self.MAX_TURNS
        return self._observe(), reward, terminated, False, {}

    # ------------------------------------------------------------------
    # Enemy move selection
    # ------------------------------------------------------------------

    def _get_enemy_moves(self):
        moves = []
        for i, en in enumerate(self.enemies):
            other_bh = [(self.my_body, self.my_health)] + [
                (self.enemies[j]["body"], self.enemies[j]["health"])
                for j in range(len(self.enemies)) if j != i
            ]
            # Enemy 0 → ONNX model if loaded, else heuristic
            if i == 0 and self._onnx_session is not None:
                moves.append(self._model_move(en["body"], en["health"], other_bh))
            else:
                en_state = self._make_state(en["body"], en["health"], other_bh)
                moves.append(choose_move_heuristic(en_state))
        return moves

    def _model_move(self, pov_body, pov_health, other_bh):
        """Run ONNX model from pov snake's perspective, return best legal move."""
        tensor = self._encode_tensor(pov_body, pov_health, other_bh)
        logits = self._onnx_session.run(None, {self._onnx_input_name: tensor})[0][0]
        legal  = self._legal_moves_for(pov_body)
        return max(legal, key=lambda m: logits[MOVE_IDX[m]])

    def _legal_moves_for(self, body):
        hx, hy   = body[0]
        occupied = set(self.my_body) | set(body[1:])
        for e in self.enemies:
            if e["body"] is not body:
                occupied.update(e["body"])
        legal = [
            m for m, (dx, dy) in DIRECTIONS.items()
            if self._in_bounds(hx + dx, hy + dy)
            and (hx + dx, hy + dy) not in occupied
        ]
        return legal or ["up"]

    def _encode_tensor(self, pov_body, pov_health, other_bh):
        """Encode board as (1, 6, H, W) float32 — same channels as generate_data.py."""
        w, h     = self.WIDTH, self.HEIGHT
        my_len   = len(pov_body)
        tensor   = np.zeros((1, 6, h, w), dtype=np.float32)

        hx, hy = pov_body[0]
        if self._in_bounds(hx, hy):
            tensor[0, 0, hy, hx] = 1.0
        for x, y in pov_body[1:]:
            if self._in_bounds(x, y):
                tensor[0, 1, y, x] = 1.0

        for (ob, _oh) in other_bh:
            ex, ey = ob[0]
            if self._in_bounds(ex, ey):
                tensor[0, 2, ey, ex] = 1.0
                if len(ob) >= my_len:
                    for dx, dy in DIRS:
                        nx, ny = ex + dx, ey + dy
                        if self._in_bounds(nx, ny):
                            tensor[0, 5, ny, nx] = 1.0
            for x, y in ob[1:]:
                if self._in_bounds(x, y):
                    tensor[0, 3, y, x] = 1.0

        for fx, fy in self.food:
            tensor[0, 4, fy, fx] = 1.0

        return tensor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        self.my_body   = []
        self.my_health = 100
        self.enemies   = []
        self.food      = []
        self.turn      = 0

    def _random_free(self, occupied, margin=1):
        for _ in range(500):
            x = random.randint(margin, self.WIDTH  - 1 - margin)
            y = random.randint(margin, self.HEIGHT - 1 - margin)
            if (x, y) not in occupied:
                return x, y
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                if (x, y) not in occupied:
                    return x, y
        return 0, 0

    def _spawn_food(self, occupied=None, n=1):
        if occupied is None:
            occupied = set(self.my_body)
            for e in self.enemies:
                occupied.update(e["body"])
            occupied.update(self.food)
        for _ in range(n):
            for _ in range(100):
                fx = random.randint(0, self.WIDTH  - 1)
                fy = random.randint(0, self.HEIGHT - 1)
                if (fx, fy) not in occupied:
                    self.food.append((fx, fy))
                    occupied.add((fx, fy))
                    break

    def _make_state(self, pov_body, pov_health, other_bh):
        pov = {
            "id": "pov",
            "head": {"x": pov_body[0][0], "y": pov_body[0][1]},
            "body": [{"x": x, "y": y} for x, y in pov_body],
            "length": len(pov_body),
            "health": pov_health,
        }
        others = [
            {
                "id": f"other_{i}",
                "head": {"x": b[0][0], "y": b[0][1]},
                "body": [{"x": x, "y": y} for x, y in b],
                "length": len(b),
                "health": h,
            }
            for i, (b, h) in enumerate(other_bh)
        ]
        return {
            "turn": self.turn,
            "board": {
                "width":  self.WIDTH,
                "height": self.HEIGHT,
                "food":   [{"x": f[0], "y": f[1]} for f in self.food],
                "snakes": [pov] + others,
            },
            "you": pov,
        }

    def _is_dead(self, body, other_bodies, health):
        hx, hy = body[0]
        if not self._in_bounds(hx, hy):
            return True
        if body[0] in body[1:]:
            return True
        for ob in other_bodies:
            if body[0] in ob[1:]:
                return True
        return health <= 0

    def _in_bounds(self, x, y=None):
        if y is None:
            x, y = x
        return 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT

    def _observe(self):
        tensor = np.zeros((6, self.HEIGHT, self.WIDTH), dtype=np.float32)
        my_len = len(self.my_body)

        hx, hy = self.my_body[0]
        if self._in_bounds(hx, hy):
            tensor[0, hy, hx] = 1.0
        for x, y in self.my_body[1:]:
            if self._in_bounds(x, y):
                tensor[1, y, x] = 1.0

        for en in self.enemies:
            ex, ey = en["body"][0]
            if self._in_bounds(ex, ey):
                tensor[2, ey, ex] = 1.0
                if len(en["body"]) >= my_len:
                    for dx, dy in DIRS:
                        nx, ny = ex + dx, ey + dy
                        if self._in_bounds(nx, ny):
                            tensor[5, ny, nx] = 1.0
            for x, y in en["body"][1:]:
                if self._in_bounds(x, y):
                    tensor[3, y, x] = 1.0

        for fx, fy in self.food:
            tensor[4, fy, fx] = 1.0

        return tensor
