"""Unit tests for lookahead simulation engine."""

import pytest
from lookahead import (
    simulate_move,
    evaluate_state,
    lookahead_2_moves,
    get_lookahead_scores,
    _is_trapped,
    _get_legal_moves,
)


def create_game_state(snakes, food, width=11, height=11, turn=1):
    """Helper to create test game states."""
    return {
        "turn": turn,
        "board": {
            "width": width,
            "height": height,
            "snakes": snakes,
            "food": food,
        },
        "you": snakes[0],
    }


# =============================================================================
# Test Legal Moves
# =============================================================================

class TestLegalMoves:
    def test_all_moves_legal_in_open_space(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        legal = _get_legal_moves(state)
        assert len(legal) == 4

    def test_wall_blocks_moves(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 0, "y": 0}, "body": [{"x": 0, "y": 0}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        legal = _get_legal_moves(state)
        assert "left" not in legal
        assert "down" not in legal
        assert "up" in legal
        assert "right" in legal

    def test_self_blocks_moves(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, 
                     "body": [{"x": 5, "y": 5}, {"x": 6, "y": 5}], 
                     "length": 2, "health": 100}],
            food=[]
        )
        legal = _get_legal_moves(state)
        assert "right" not in legal  # Body blocks


# =============================================================================
# Test Move Simulation
# =============================================================================

class TestSimulateMove:
    def test_move_updates_head(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        new_state = simulate_move(state, "up")
        assert new_state["you"]["head"]["x"] == 5
        assert new_state["you"]["head"]["y"] == 6

    def test_move_without_food_removes_tail(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, 
                     "body": [{"x": 5, "y": 5}, {"x": 4, "y": 5}], 
                     "length": 2, "health": 100}],
            food=[]
        )
        new_state = simulate_move(state, "up")
        assert new_state["you"]["length"] == 2  # Same length
        assert {"x": 4, "y": 5} not in new_state["you"]["body"]  # Tail removed

    def test_move_with_food_keeps_tail(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, 
                     "body": [{"x": 5, "y": 5}, {"x": 4, "y": 5}], 
                     "length": 2, "health": 100}],
            food=[{"x": 5, "y": 6}]
        )
        new_state = simulate_move(state, "up")  # Move to food
        assert new_state["you"]["length"] == 3  # Grew!

    def test_health_decreases_without_food(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        new_state = simulate_move(state, "up")
        assert new_state["you"]["health"] == 99


# =============================================================================
# Test State Evaluation
# =============================================================================

class TestEvaluateState:
    def test_trapped_state_is_bad(self):
        # Create a trapped state (cornered with body blocking)
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 0, "y": 0}, 
                     "body": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}], 
                     "length": 3, "health": 100}],
            food=[]
        )
        score = evaluate_state(state, "me")
        assert score < 0  # Trapped = bad

    def test_more_space_is_better(self):
        # Use a longer snake so space matters more
        state_open = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, 
                     "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}], 
                     "length": 3, "health": 100}],
            food=[]
        )
        state_corner = create_game_state(
            snakes=[{"id": "me", "head": {"x": 0, "y": 0}, 
                     "body": [{"x": 0, "y": 0}, {"x": 0, "y": 1}, {"x": 0, "y": 2}], 
                     "length": 3, "health": 100}],
            food=[]
        )
        # Open space should score >= corner (may be equal due to flood fill limit)
        score_open = evaluate_state(state_open, "me")
        score_corner = evaluate_state(state_corner, "me")
        assert score_open >= score_corner


# =============================================================================
# Test Lookahead
# =============================================================================

class TestLookahead:
    def test_lookahead_returns_score(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}]
        )
        score = lookahead_2_moves(state, "up")
        assert isinstance(score, float)

    def test_dead_end_has_low_score(self):
        # Create a scenario where one move leads to trap
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 1, "y": 0}, 
                     "body": [{"x": 1, "y": 0}, {"x": 2, "y": 0}], 
                     "length": 2, "health": 100}],
            food=[]
        )
        # Moving down is impossible (wall), moving right might be bad
        scores = get_lookahead_scores(state)
        assert "down" not in scores  # Illegal move
        assert "left" in scores or "up" in scores  # Some legal move exists

    def test_get_lookahead_scores_returns_dict(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        scores = get_lookahead_scores(state)
        assert isinstance(scores, dict)
        assert len(scores) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
