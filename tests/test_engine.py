import unittest

from risk_ai_challenge.engine import (
    Point,
    advance_board,
    astar_distance,
    choose_move,
    flood_fill_area,
    occupied_cells,
    parse_state,
    safe_moves,
    voronoi_control,
)


def state(you_body, snakes=None, food=None, width=7, height=7):
    snakes = snakes or []
    food = food or []
    you = {
        "id": "you",
        "health": 90,
        "body": [{"x": x, "y": y} for x, y in you_body],
    }
    all_snakes = [you] + snakes
    return {
        "game": {"id": "test"},
        "turn": 1,
        "board": {
            "height": height,
            "width": width,
            "food": [{"x": x, "y": y} for x, y in food],
            "snakes": all_snakes,
        },
        "you": you,
    }


class EngineTest(unittest.TestCase):
    def test_avoids_wall_and_body(self):
        board = parse_state(state([(0, 0), (0, 1), (1, 1)]))

        self.assertEqual(safe_moves(board, board.you), ["right"])

    def test_flood_fill_counts_reachable_space(self):
        board = parse_state(state([(1, 1)], width=3, height=3))
        blocked = {Point(1, 0), Point(1, 1), Point(1, 2)}

        self.assertEqual(flood_fill_area(board, Point(0, 1), blocked), 3)

    def test_astar_finds_food_route(self):
        board = parse_state(state([(0, 0)], food=[(3, 0)]))

        self.assertEqual(astar_distance(board, Point(0, 0), Point(3, 0), set()), 3)

    def test_voronoi_splits_control(self):
        enemy = {
            "id": "enemy",
            "health": 90,
            "body": [{"x": 4, "y": 0}],
        }
        board = parse_state(state([(0, 0)], snakes=[enemy], width=5, height=1))

        control = voronoi_control(board)

        self.assertEqual(control["you"], 2)
        self.assertEqual(control["enemy"], 2)

    def test_choose_move_runs_under_small_budget(self):
        game_state = state([(3, 3), (3, 2), (2, 2)], food=[(5, 3)])

        self.assertIn(choose_move(game_state, time_limit_ms=20), {"up", "right"})

    def test_avoids_equal_length_head_to_head_threat(self):
        enemy = {
            "id": "enemy",
            "health": 90,
            "body": [{"x": 3, "y": 2}, {"x": 3, "y": 1}, {"x": 4, "y": 1}],
        }
        board = parse_state(
            state(
                [(1, 2), (1, 1), (0, 1)],
                snakes=[enemy],
                width=5,
                height=5,
            )
        )

        self.assertNotIn("right", safe_moves(board, board.you))

    def test_longer_snake_survives_head_to_head_collision(self):
        enemy = {
            "id": "enemy",
            "health": 90,
            "body": [{"x": 3, "y": 2}, {"x": 3, "y": 1}, {"x": 4, "y": 1}],
        }
        board = parse_state(
            state(
                [(1, 2), (1, 1), (0, 1), (0, 0)],
                snakes=[enemy],
                width=5,
                height=5,
            )
        )

        next_board = advance_board(board, {"you": "right", "enemy": "left"})

        self.assertEqual([snake.id for snake in next_board.snakes], ["you"])


if __name__ == "__main__":
    unittest.main()
