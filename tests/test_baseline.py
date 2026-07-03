import unittest

from risk_ai_challenge.baseline import (
    BASELINE_STRATEGIES,
    choose_move,
    score_breakdown,
    score_moves,
)
from risk_ai_challenge.engine import parse_state


def state(you_body, snakes=None, food=None, health=90, width=7, height=7):
    snakes = snakes or []
    food = food or []
    you = {
        "id": "you",
        "health": health,
        "length": len(you_body),
        "head": {"x": you_body[0][0], "y": you_body[0][1]},
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


class BaselineTest(unittest.TestCase):
    def test_baseline_has_five_named_strategies(self):
        self.assertEqual(
            [strategy.name for strategy in BASELINE_STRATEGIES],
            [
                "survival",
                "flood_fill_space",
                "food_path",
                "head_to_head",
                "territory_control",
            ],
        )

    def test_avoids_immediate_wall_and_body_death(self):
        game_state = state([(0, 0), (0, 1), (1, 1)])

        self.assertEqual(choose_move(game_state), "right")

    def test_hungry_snake_moves_toward_reachable_food(self):
        game_state = state([(3, 3), (3, 2), (2, 2)], food=[(4, 3)], health=20)

        self.assertEqual(choose_move(game_state), "right")

    def test_score_breakdown_reports_all_strategies(self):
        board = parse_state(state([(3, 3), (3, 2), (2, 2)], food=[(4, 3)]))

        breakdown = score_breakdown(board, "right")

        self.assertEqual(len(breakdown), 5)
        self.assertTrue(all(item.name and isinstance(item.score, float) for item in breakdown))

    def test_unsafe_move_scores_below_safe_move(self):
        board = parse_state(state([(0, 0), (0, 1), (1, 1)]))
        scores = score_moves(board, board.you)

        self.assertGreater(scores["right"], scores["left"])
        self.assertGreater(scores["right"], scores["down"])


if __name__ == "__main__":
    unittest.main()
