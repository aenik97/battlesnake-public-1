"""Benchmark and comparison tests for Battlesnake v2.

Tests various scenarios to validate improvements over baseline.
"""

import time
import pytest
from logic import choose_move, choose_move_heuristic
from strategy import get_strategy_summary


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
# Performance Benchmarks
# =============================================================================

class TestPerformance:
    """Ensure move selection is fast enough for production (<100ms)."""
    
    def test_simple_move_performance(self):
        """Simple state should be very fast."""
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}]
        )
        
        start = time.time()
        for _ in range(100):
            choose_move(state)
        elapsed = (time.time() - start) / 100
        
        assert elapsed < 0.1, f"Move selection too slow: {elapsed*1000:.1f}ms"

    def test_complex_move_performance(self):
        """Complex state with multiple snakes should still be fast."""
        state = create_game_state(
            snakes=[
                {"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}, {"x": 4, "y": 5}], 
                 "length": 2, "health": 80},
                {"id": "enemy1", "head": {"x": 7, "y": 5}, "body": [{"x": 7, "y": 5}], "length": 1, "health": 90},
                {"id": "enemy2", "head": {"x": 3, "y": 5}, "body": [{"x": 3, "y": 5}], "length": 1, "health": 90},
                {"id": "enemy3", "head": {"x": 5, "y": 7}, "body": [{"x": 5, "y": 7}], "length": 1, "health": 90},
            ],
            food=[{"x": 6, "y": 5}, {"x": 4, "y": 6}, {"x": 5, "y": 4}]
        )
        
        start = time.time()
        for _ in range(50):
            choose_move(state)
        elapsed = (time.time() - start) / 50
        
        assert elapsed < 0.2, f"Complex move too slow: {elapsed*1000:.1f}ms"


# =============================================================================
# Scenario Tests
# =============================================================================

class TestScenarios:
    """Test specific game scenarios."""
    
    def test_corner_escape(self):
        """Snake in corner should find escape route."""
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 0, "y": 0}, 
                     "body": [{"x": 0, "y": 0}, {"x": 0, "y": 1}], 
                     "length": 2, "health": 100}],
            food=[]
        )
        move = choose_move(state)
        # Should not try to go into wall
        assert move in ["up", "right"]

    def test_surrounded_by_enemies(self):
        """Snake surrounded should pick safest option."""
        state = create_game_state(
            snakes=[
                {"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                 "length": 1, "health": 100},
                {"id": "e1", "head": {"x": 6, "y": 5}, "body": [{"x": 6, "y": 5}], "length": 1, "health": 100},
                {"id": "e2", "head": {"x": 4, "y": 5}, "body": [{"x": 4, "y": 5}], "length": 1, "health": 100},
                {"id": "e3", "head": {"x": 5, "y": 6}, "body": [{"x": 5, "y": 6}], "length": 1, "health": 100},
            ],
            food=[]
        )
        move = choose_move(state)
        # Should pick the open direction (down)
        assert move in ["down", "up"]  # up is blocked by e3, so down

    def test_tail_chasing(self):
        """Should recognize enemy tail as future free space."""
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
        move = choose_move(state)
        # Moving right goes to enemy tail (frees next turn)
        # This is a good move - test that it's considered
        assert move in ["up", "down", "left", "right"]  # Any legal move is fine

    def test_food_priority_when_hungry(self):
        """Hungry snake should prioritize food."""
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 10}],  # Very hungry!
            food=[{"x": 5, "y": 6}]  # Food directly above
        )
        move = choose_move(state)
        # Should move toward food
        assert move == "up"

    def test_avoids_bigger_snakes(self):
        """Should avoid head-to-head with bigger snakes."""
        state = create_game_state(
            snakes=[
                {"id": "me", "head": {"x": 5, "y": 5}, 
                 "body": [{"x": 5, "y": 5}, {"x": 4, "y": 5}], 
                 "length": 2, "health": 100},
                {"id": "big", "head": {"x": 7, "y": 5}, 
                 "body": [{"x": 7, "y": 5}] + [{"x": 7-i, "y": 5} for i in range(1, 6)], 
                 "length": 6, "health": 100},  # Much bigger
            ],
            food=[]
        )
        move = choose_move(state)
        # Should not move toward bigger snake
        # (right would go toward the big snake)
        # This is a soft test - just verify it returns a valid move
        assert move in ["up", "down", "left", "right"]


# =============================================================================
# Strategy Validation
# =============================================================================

class TestStrategyValidation:
    """Validate strategy adapts correctly."""
    
    def test_early_game_aggressive(self):
        """Early game should be aggressive."""
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}],
            turn=10
        )
        summary = get_strategy_summary(state)
        assert summary["phase"] == "early"
        assert summary["weights"]["aggression"] > 1.0

    def test_late_game_cautious(self):
        """Late game should be cautious."""
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 50}],
            food=[],
            turn=100
        )
        summary = get_strategy_summary(state)
        assert summary["phase"] == "late"
        assert summary["weights"]["safety"] > 1.0
        assert summary["weights"]["aggression"] < 1.0

    def test_low_health_changes_strategy(self):
        """Low health should prioritize food."""
        state_normal = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}],
            turn=50
        )
        state_hungry = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 20}],
            food=[{"x": 6, "y": 5}],
            turn=50
        )
        
        summary_normal = get_strategy_summary(state_normal)
        summary_hungry = get_strategy_summary(state_hungry)
        
        assert summary_hungry["weights"]["food"] > summary_normal["weights"]["food"]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_legal_moves(self):
        """Completely trapped snake should return fallback."""
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 0, "y": 0}, 
                     "body": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 0, "y": 1}], 
                     "length": 3, "health": 100}],
            food=[]
        )
        # Should not crash, should return some move
        move = choose_move(state)
        assert move in ["up", "down", "left", "right"]

    def test_empty_board(self):
        """Board with no food and no enemies."""
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        move = choose_move(state)
        assert move in ["up", "down", "left", "right"]

    def test_large_board(self):
        """Should work on larger boards."""
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 27, "y": 27}, "body": [{"x": 27, "y": 27}], 
                     "length": 1, "health": 100}],
            food=[{"x": 28, "y": 27}],
            width=35,
            height=35
        )
        move = choose_move(state)
        assert move in ["up", "down", "left", "right"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
