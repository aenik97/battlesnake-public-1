"""Baseline Battlesnake policy built from five explicit strategies.

This module is intentionally simpler than the full tree-search engine. It is a
readable, deterministic baseline that scores each candidate move with five
independent strategies:

1. survival / immediate safety
2. flood-fill space
3. food pathfinding
4. head-to-head pressure and avoidance
5. multi-agent territory control
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .engine import (
    BoardState,
    MOVES,
    SnakeState,
    advance_board,
    astar_distance,
    flood_fill_area,
    has_snake,
    head_to_head_score,
    in_bounds,
    occupied_cells,
    parse_state,
    safe_moves,
    voronoi_control,
)


LOSING_SCORE = -1_000_000.0


@dataclass(frozen=True)
class StrategyScore:
    name: str
    score: float


class BaselineStrategy(Protocol):
    name: str

    def score(self, board: BoardState, snake: SnakeState, direction: str) -> float:
        ...


class SurvivalStrategy:
    name = "survival"

    def score(self, board: BoardState, snake: SnakeState, direction: str) -> float:
        head = snake.head.move(direction)
        if not in_bounds(head, board):
            return LOSING_SCORE

        blocked = occupied_cells(board, include_tails=False)
        if head in blocked:
            return LOSING_SCORE

        dangerous_heads = {
            point
            for other in board.snakes
            if other.id != snake.id and other.length >= snake.length
            for point in (other.head.move(move) for move in MOVES)
            if in_bounds(point, board)
        }
        if head in dangerous_heads:
            return -25_000.0

        escape_routes = sum(
            1
            for move in MOVES
            if in_bounds(head.move(move), board)
            and head.move(move) not in occupied_cells(board, include_tails=False)
        )
        return 1_000.0 + 80.0 * escape_routes


class FloodFillSpaceStrategy:
    name = "flood_fill_space"

    def score(self, board: BoardState, snake: SnakeState, direction: str) -> float:
        next_board = advance_board(board, {snake.id: direction})
        if not has_snake(next_board, snake.id):
            return LOSING_SCORE

        next_snake = next(s for s in next_board.snakes if s.id == snake.id)
        space = flood_fill_area(
            next_board,
            next_snake.head,
            occupied_cells(next_board),
        )
        body_buffer = space - next_snake.length
        if body_buffer < 0:
            return -10_000.0 + 250.0 * body_buffer
        return 45.0 * space + 120.0 * min(body_buffer, 12)


class FoodPathStrategy:
    name = "food_path"

    def score(self, board: BoardState, snake: SnakeState, direction: str) -> float:
        if not board.food:
            return 0.0

        next_board = advance_board(board, {snake.id: direction})
        if not has_snake(next_board, snake.id):
            return LOSING_SCORE

        next_snake = next(s for s in next_board.snakes if s.id == snake.id)
        eating_bonus = 1_250.0 if next_snake.head in board.food else 0.0
        if eating_bonus and not next_board.food:
            return eating_bonus

        blocked = occupied_cells(next_board, include_tails=False)
        distances = [
            astar_distance(next_board, next_snake.head, food, blocked)
            for food in next_board.food
        ]
        reachable = [distance for distance in distances if distance is not None]
        if not reachable:
            return -500.0 if next_snake.health < 45 else -40.0

        nearest = min(reachable)
        urgency = 4.0 if next_snake.health < 45 else 1.0
        return eating_bonus + urgency * max(0.0, 280.0 - 24.0 * nearest)


class HeadToHeadStrategy:
    name = "head_to_head"

    def score(self, board: BoardState, snake: SnakeState, direction: str) -> float:
        next_board = advance_board(board, {snake.id: direction})
        if not has_snake(next_board, snake.id):
            return LOSING_SCORE
        next_snake = next(s for s in next_board.snakes if s.id == snake.id)
        return 18.0 * head_to_head_score(next_board, next_snake)


class TerritoryControlStrategy:
    name = "territory_control"

    def score(self, board: BoardState, snake: SnakeState, direction: str) -> float:
        next_board = advance_board(board, {snake.id: direction})
        if not has_snake(next_board, snake.id):
            return LOSING_SCORE

        control = voronoi_control(next_board)
        my_control = control.get(snake.id, 0)
        enemy_best = max(
            (score for other_id, score in control.items() if other_id != snake.id),
            default=0,
        )
        return 32.0 * my_control - 12.0 * enemy_best


BASELINE_STRATEGIES: tuple[BaselineStrategy, ...] = (
    SurvivalStrategy(),
    FloodFillSpaceStrategy(),
    FoodPathStrategy(),
    HeadToHeadStrategy(),
    TerritoryControlStrategy(),
)


def choose_move(game_state: dict) -> str:
    board = parse_state(game_state)
    snake = board.you
    legal = list(MOVES)

    scores = score_moves(board, snake, legal)
    return max(scores, key=scores.get)


def score_moves(
    board: BoardState,
    snake: SnakeState,
    moves: list[str] | None = None,
) -> dict[str, float]:
    moves = moves or list(MOVES)
    safe = set(safe_moves(board, snake))
    scores: dict[str, float] = {}
    for direction in moves:
        total = 0.0
        for strategy in BASELINE_STRATEGIES:
            total += strategy.score(board, snake, direction)
        if direction not in safe:
            total -= 15_000.0
        scores[direction] = total
    return scores


def score_breakdown(board: BoardState, direction: str) -> list[StrategyScore]:
    snake = board.you
    return [
        StrategyScore(strategy.name, strategy.score(board, snake, direction))
        for strategy in BASELINE_STRATEGIES
    ]
