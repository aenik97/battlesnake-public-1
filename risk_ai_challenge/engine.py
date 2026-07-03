"""Deterministic Battlesnake move engine.

The engine favors cheap, reliable heuristics first and spends remaining response
time on shallow tree search. It is written without third-party dependencies so
the core strategy stays easy to profile and deploy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
import time
from typing import Iterable

MOVES = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True, order=True)
class Point:
    x: int
    y: int

    def move(self, direction: str) -> "Point":
        dx, dy = MOVES[direction]
        return Point(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class SnakeState:
    id: str
    health: int
    body: tuple[Point, ...]

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def length(self) -> int:
        return len(self.body)


@dataclass(frozen=True)
class BoardState:
    width: int
    height: int
    food: frozenset[Point]
    snakes: tuple[SnakeState, ...]
    you_id: str

    @property
    def you(self) -> SnakeState:
        return next(snake for snake in self.snakes if snake.id == self.you_id)


@dataclass(frozen=True)
class SearchResult:
    direction: str
    score: float


class Timeout(Exception):
    pass


def choose_move(game_state: dict, time_limit_ms: int = 450) -> str:
    """Return a Battlesnake move under a conservative response-time budget."""

    board = parse_state(game_state)
    deadline = time.perf_counter() + (time_limit_ms / 1000.0)
    legal = safe_moves(board, board.you)
    if not legal:
        return fallback_move(board)

    base_scores = score_root_moves(board, legal)
    best = max(base_scores, key=base_scores.get)

    try:
        for depth in range(1, 5):
            ensure_time(deadline)
            result = search_best_move(board, legal, depth, deadline)
            if result is not None:
                best = result.direction
    except Timeout:
        pass

    return best


def parse_state(game_state: dict) -> BoardState:
    board = game_state["board"]
    snakes = tuple(
        SnakeState(
            id=snake["id"],
            health=int(snake.get("health", 100)),
            body=tuple(Point(part["x"], part["y"]) for part in snake["body"]),
        )
        for snake in board.get("snakes", [])
    )
    return BoardState(
        width=int(board["width"]),
        height=int(board["height"]),
        food=frozenset(Point(food["x"], food["y"]) for food in board.get("food", [])),
        snakes=snakes,
        you_id=game_state["you"]["id"],
    )


def safe_moves(board: BoardState, snake: SnakeState) -> list[str]:
    occupied = occupied_cells(board, include_tails=False)
    dangerous_heads = {
        point
        for other in board.snakes
        if other.id != snake.id and other.length >= snake.length
        for point in adjacent_points(other.head, board)
    }

    moves = []
    for direction in MOVES:
        head = snake.head.move(direction)
        if not in_bounds(head, board):
            continue
        if head in occupied:
            continue
        if head in dangerous_heads:
            continue
        moves.append(direction)
    return moves


def score_root_moves(board: BoardState, moves: Iterable[str]) -> dict[str, float]:
    scores = {}
    for direction in moves:
        next_board = advance_board(board, {board.you_id: direction})
        you = next_board.you if has_snake(next_board, board.you_id) else None
        if you is None:
            scores[direction] = -1_000_000.0
            continue

        space = flood_fill_area(next_board, you.head, occupied_cells(next_board))
        voronoi = voronoi_control(next_board)
        food_score = nearest_food_score(next_board, you)
        head_score = head_to_head_score(next_board, you)
        length_buffer = min(space - you.length, 20)
        health_pressure = max(0, 35 - you.health)

        scores[direction] = (
            8.0 * space
            + 5.0 * voronoi.get(you.id, 0)
            + food_score
            + head_score
            + 6.0 * length_buffer
            - 9.0 * health_pressure
        )
    return scores


def search_best_move(
    board: BoardState, legal: list[str], depth: int, deadline: float
) -> SearchResult | None:
    if len(board.snakes) <= 2:
        return alpha_beta_root(board, legal, depth, deadline)
    return maxn_root(board, legal, depth, deadline)


def alpha_beta_root(
    board: BoardState, legal: list[str], depth: int, deadline: float
) -> SearchResult | None:
    best = SearchResult(legal[0], -math.inf)
    alpha = -math.inf
    beta = math.inf
    opponent = next((s for s in board.snakes if s.id != board.you_id), None)

    for direction in legal:
        ensure_time(deadline)
        opponent_moves = safe_moves(board, opponent) if opponent else ["up"]
        worst = math.inf
        for opp_direction in opponent_moves or ["up"]:
            moves = {board.you_id: direction}
            if opponent:
                moves[opponent.id] = opp_direction
            child = advance_board(board, moves)
            score = alpha_beta(child, depth - 1, alpha, beta, deadline)
            worst = min(worst, score)
            beta = min(beta, worst)
            if beta <= alpha:
                break
        if worst > best.score:
            best = SearchResult(direction, worst)
        alpha = max(alpha, best.score)
    return best


def alpha_beta(
    board: BoardState, depth: int, alpha: float, beta: float, deadline: float
) -> float:
    ensure_time(deadline)
    if depth <= 0 or not has_snake(board, board.you_id):
        return evaluate(board).get(board.you_id, -1_000_000.0)

    you = board.you
    opponent = next((s for s in board.snakes if s.id != board.you_id), None)
    legal = safe_moves(board, you) or list(MOVES)
    opponent_moves = safe_moves(board, opponent) if opponent else ["up"]

    value = -math.inf
    for direction in legal:
        worst = math.inf
        for opp_direction in opponent_moves or ["up"]:
            moves = {you.id: direction}
            if opponent:
                moves[opponent.id] = opp_direction
            child = advance_board(board, moves)
            worst = min(worst, alpha_beta(child, depth - 1, alpha, beta, deadline))
            beta = min(beta, worst)
            if beta <= alpha:
                break
        value = max(value, worst)
        alpha = max(alpha, value)
        if beta <= alpha:
            break
    return value


def maxn_root(
    board: BoardState, legal: list[str], depth: int, deadline: float
) -> SearchResult | None:
    best = SearchResult(legal[0], -math.inf)
    for direction in legal:
        ensure_time(deadline)
        child = advance_board(board, {board.you_id: direction})
        scores = maxn(child, depth - 1, deadline)
        score = scores.get(board.you_id, -1_000_000.0)
        if score > best.score:
            best = SearchResult(direction, score)
    return best


def maxn(board: BoardState, depth: int, deadline: float) -> dict[str, float]:
    ensure_time(deadline)
    scores = evaluate(board)
    if depth <= 0 or board.you_id not in scores:
        return scores

    actor = board.snakes[depth % len(board.snakes)]
    moves = safe_moves(board, actor) or list(MOVES)
    best_scores = scores
    best_actor_score = -math.inf
    for direction in moves:
        child = advance_board(board, {actor.id: direction})
        child_scores = maxn(child, depth - 1, deadline)
        actor_score = child_scores.get(actor.id, -1_000_000.0)
        if actor_score > best_actor_score:
            best_actor_score = actor_score
            best_scores = child_scores
    return best_scores


def evaluate(board: BoardState) -> dict[str, float]:
    if not has_snake(board, board.you_id):
        return {}

    control = voronoi_control(board)
    occupied = occupied_cells(board)
    scores = {}
    for snake in board.snakes:
        area = flood_fill_area(board, snake.head, occupied)
        scores[snake.id] = (
            6.0 * area
            + 4.5 * control.get(snake.id, 0)
            + 12.0 * snake.length
            + nearest_food_score(board, snake)
            + head_to_head_score(board, snake)
            + snake.health * 0.3
        )
    return scores


def nearest_food_score(board: BoardState, snake: SnakeState) -> float:
    if not board.food:
        return 0.0
    occupied = occupied_cells(board, include_tails=False)
    distances = [
        astar_distance(board, snake.head, food, occupied)
        for food in board.food
    ]
    reachable = [distance for distance in distances if distance is not None]
    if not reachable:
        return -40.0 if snake.health < 40 else 0.0
    nearest = min(reachable)
    urgency = 4.0 if snake.health < 45 else 1.4
    return max(0.0, 80.0 - nearest * 10.0) * urgency


def head_to_head_score(board: BoardState, snake: SnakeState) -> float:
    score = 0.0
    for other in board.snakes:
        if other.id == snake.id:
            continue
        distance = manhattan(snake.head, other.head)
        if distance == 1:
            if snake.length > other.length:
                score += 55.0
            elif snake.length <= other.length:
                score -= 90.0
        elif distance == 2 and snake.length <= other.length:
            score -= 20.0
    return score


def flood_fill_area(board: BoardState, start: Point, blocked: set[Point]) -> int:
    if start in blocked:
        blocked = set(blocked)
        blocked.discard(start)

    seen = {start}
    queue = deque([start])
    while queue:
        point = queue.popleft()
        for next_point in adjacent_points(point, board):
            if next_point in seen or next_point in blocked:
                continue
            seen.add(next_point)
            queue.append(next_point)
    return len(seen)


def voronoi_control(board: BoardState) -> dict[str, int]:
    blocked = occupied_cells(board)
    owner_distance: dict[Point, tuple[str | None, int]] = {}
    queue = deque()

    for snake in board.snakes:
        owner_distance[snake.head] = (snake.id, 0)
        queue.append((snake.head, snake.id, 0))

    while queue:
        point, owner, distance = queue.popleft()
        for next_point in adjacent_points(point, board):
            if next_point in blocked and next_point not in owner_distance:
                continue
            if next_point not in owner_distance:
                owner_distance[next_point] = (owner, distance + 1)
                queue.append((next_point, owner, distance + 1))
                continue

            existing_owner, existing_distance = owner_distance[next_point]
            if existing_distance == distance + 1 and existing_owner != owner:
                owner_distance[next_point] = (None, existing_distance)

    control = {snake.id: 0 for snake in board.snakes}
    for owner, _distance in owner_distance.values():
        if owner is not None:
            control[owner] += 1
    return control


def astar_distance(
    board: BoardState, start: Point, goal: Point, blocked: set[Point]
) -> int | None:
    frontier = [(manhattan(start, goal), 0, start)]
    best_cost = {start: 0}

    while frontier:
        _priority, cost, point = heapq.heappop(frontier)
        if point == goal:
            return cost
        if cost > best_cost[point]:
            continue
        for next_point in adjacent_points(point, board):
            if next_point in blocked and next_point != goal:
                continue
            new_cost = cost + 1
            if new_cost >= best_cost.get(next_point, math.inf):
                continue
            best_cost[next_point] = new_cost
            heapq.heappush(
                frontier,
                (new_cost + manhattan(next_point, goal), new_cost, next_point),
            )
    return None


def advance_board(board: BoardState, moves: dict[str, str]) -> BoardState:
    next_snakes = []
    head_counts: dict[Point, int] = {}
    moved: dict[str, SnakeState] = {}

    for snake in board.snakes:
        direction = moves.get(snake.id)
        if direction is None:
            legal = safe_moves(board, snake)
            direction = legal[0] if legal else "up"
        head = snake.head.move(direction)
        ate = head in board.food
        body = (head,) + snake.body if ate else (head,) + snake.body[:-1]
        moved_snake = SnakeState(snake.id, 100 if ate else snake.health - 1, body)
        moved[snake.id] = moved_snake
        head_counts[head] = head_counts.get(head, 0) + 1

    occupied_by_body = {
        point
        for snake in moved.values()
        for point in snake.body[1:]
    }

    for snake in moved.values():
        if snake.health <= 0:
            continue
        if not in_bounds(snake.head, board):
            continue
        if snake.head in occupied_by_body:
            continue
        if head_counts[snake.head] > 1:
            contenders = [other for other in moved.values() if other.head == snake.head]
            longest = max(other.length for other in contenders)
            longest_count = sum(1 for other in contenders if other.length == longest)
            if snake.length < longest or longest_count > 1:
                continue
        next_snakes.append(snake)

    eaten = {snake.head for snake in moved.values() if snake.head in board.food}
    return BoardState(
        width=board.width,
        height=board.height,
        food=frozenset(board.food - eaten),
        snakes=tuple(next_snakes),
        you_id=board.you_id,
    )


def occupied_cells(board: BoardState, include_tails: bool = True) -> set[Point]:
    cells = set()
    for snake in board.snakes:
        body = snake.body if include_tails else snake.body[:-1]
        cells.update(body)
    return cells


def adjacent_points(point: Point, board: BoardState) -> list[Point]:
    points = [point.move(direction) for direction in MOVES]
    return [candidate for candidate in points if in_bounds(candidate, board)]


def in_bounds(point: Point, board: BoardState) -> bool:
    return 0 <= point.x < board.width and 0 <= point.y < board.height


def has_snake(board: BoardState, snake_id: str) -> bool:
    return any(snake.id == snake_id for snake in board.snakes)


def fallback_move(board: BoardState) -> str:
    occupied = occupied_cells(board, include_tails=False)
    for direction in MOVES:
        head = board.you.head.move(direction)
        if in_bounds(head, board) and head not in occupied:
            return direction
    return "up"


def manhattan(a: Point, b: Point) -> int:
    return abs(a.x - b.x) + abs(a.y - b.y)


def ensure_time(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise Timeout
