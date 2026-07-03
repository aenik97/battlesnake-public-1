"""Unit tests for feature extraction module."""

import pytest
from features import (
    enemy_tail_features,
    wall_corner_features,
    game_phase_features,
    food_strategy_features,
    enemy_proximity_features,
    extract_all_features,
    _get_next_cell,
    _manhattan,
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
# Test Helper Functions
# =============================================================================

class TestHelpers:
    def test_manhattan_distance(self):
        assert _manhattan((0, 0), (3, 4)) == 7
        assert _manhattan((5, 5), (5, 5)) == 0
        assert _manhattan((0, 0), (0, 5)) == 5

    def test_get_next_cell(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        assert _get_next_cell(state, "up") == (5, 6)
        assert _get_next_cell(state, "down") == (5, 4)
        assert _get_next_cell(state, "left") == (4, 5)
        assert _get_next_cell(state, "right") == (6, 5)


# =============================================================================
# Test Enemy Tail Features
# =============================================================================

class TestEnemyTailFeatures:
    def test_no_enemies(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        feats = enemy_tail_features(state, "up")
        assert feats["nearest_enemy_tail_dist"] == 999.0
        assert feats["tail_in_next_cell"] == 0.0

    def test_tail_in_next_cell(self):
        state = create_game_state(
            snakes=[
                {"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                 "length": 1, "health": 100},
                {"id": "enemy", "head": {"x": 7, "y": 5}, 
                 "body": [{"x": 7, "y": 5}, {"x": 6, "y": 5}], 
                 "length": 2, "health": 100},
            ],
            food=[]
        )
        feats = enemy_tail_features(state, "right")  # Move to (6, 5) — enemy tail
        assert feats["tail_in_next_cell"] == 1.0
        assert feats["nearest_enemy_tail_dist"] == 0.0

    def test_tails_within_range(self):
        state = create_game_state(
            snakes=[
                {"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                 "length": 1, "health": 100},
                {"id": "enemy1", "head": {"x": 8, "y": 5}, 
                 "body": [{"x": 8, "y": 5}, {"x": 7, "y": 5}, {"x": 6, "y": 5}], 
                 "length": 3, "health": 100},
                {"id": "enemy2", "head": {"x": 5, "y": 8}, 
                 "body": [{"x": 5, "y": 8}, {"x": 5, "y": 7}], 
                 "length": 2, "health": 100},
            ],
            food=[]
        )
        feats = enemy_tail_features(state, "up")  # Move to (5, 6)
        # Tail at (6,5) is distance 1, tail at (5,7) is distance 1
        assert feats["tails_within_2"] >= 2.0


# =============================================================================
# Test Wall/Corner Features
# =============================================================================

class TestWallCornerFeatures:
    def test_center_cell(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        feats = wall_corner_features(state, "up")
        assert feats["in_corner"] == 0.0
        assert feats["on_edge"] == 0.0
        assert feats["corner_escapes"] == 4.0

    def test_corner_cell(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 0, "y": 0}, "body": [{"x": 0, "y": 0}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        feats = wall_corner_features(state, "up")  # Still at edge
        assert feats["on_edge"] == 1.0
        assert feats["corner_escapes"] == 3.0  # Edge has 3 escapes

    def test_actual_corner(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 0, "y": 0}, "body": [{"x": 0, "y": 0}], 
                     "length": 1, "health": 100}],
            food=[],
            width=11,
            height=11
        )
        # Already in corner, any move gets us out
        feats = wall_corner_features(state, "right")  # Move to (1, 0) — still on edge
        assert feats["on_edge"] == 1.0
        assert feats["in_corner"] == 0.0


# =============================================================================
# Test Game Phase Features
# =============================================================================

class TestGamePhaseFeatures:
    def test_early_game(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}],
            turn=10
        )
        feats = game_phase_features(state)
        assert feats["is_early_game"] == 1.0
        assert feats["is_mid_game"] == 0.0
        assert feats["is_late_game"] == 0.0

    def test_late_game(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[],
            turn=100
        )
        feats = game_phase_features(state)
        assert feats["is_late_game"] == 1.0

    def test_food_scarcity(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[],  # No food = high scarcity
            turn=50
        )
        feats = game_phase_features(state)
        assert feats["food_scarcity"] >= 0.9


# =============================================================================
# Test Food Strategy Features
# =============================================================================

class TestFoodStrategyFeatures:
    def test_is_food(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}]
        )
        feats = food_strategy_features(state, "right")  # Move to (6, 5) — food
        assert feats["is_food"] == 1.0

    def test_food_delta_positive(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}]
        )
        feats = food_strategy_features(state, "right")  # Getting closer
        assert feats["food_delta"] > 0

    def test_no_food(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        feats = food_strategy_features(state, "up")
        assert feats["nearest_food_dist"] == 999.0
        assert feats["is_food"] == 0.0


# =============================================================================
# Test Combined Feature Extraction
# =============================================================================

class TestExtractAllFeatures:
    def test_returns_all_features(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}]
        )
        feats = extract_all_features(state, "up")
        
        # Should have 23 features
        expected_count = 23
        assert len(feats) == expected_count, f"Expected {expected_count} features, got {len(feats)}"
    
    def test_features_are_numeric(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}]
        )
        feats = extract_all_features(state, "up")
        for key, value in feats.items():
            assert isinstance(value, (int, float)), f"Feature {key} is not numeric: {type(value)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
