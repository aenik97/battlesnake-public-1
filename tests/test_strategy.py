"""Unit tests for game phase strategy system."""

import pytest
from strategy import (
    get_game_phase,
    get_phase_weights,
    get_risk_tolerance,
    should_chase_food,
    should_avoid_enemies,
    get_strategy_summary,
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
# Test Game Phase Detection
# =============================================================================

class TestGamePhase:
    def test_early_game(self):
        assert get_game_phase(1) == "early"
        assert get_game_phase(10) == "early"
        assert get_game_phase(20) == "early"

    def test_mid_game(self):
        assert get_game_phase(21) == "mid"
        assert get_game_phase(50) == "mid"
        assert get_game_phase(80) == "mid"

    def test_late_game(self):
        assert get_game_phase(81) == "late"
        assert get_game_phase(100) == "late"
        assert get_game_phase(200) == "late"


# =============================================================================
# Test Phase Weights
# =============================================================================

class TestPhaseWeights:
    def test_early_game_aggressive(self):
        weights = get_phase_weights("early", 100, True)
        assert weights["aggression"] > 1.0
        assert weights["territory"] > 1.0

    def test_late_game_cautious(self):
        weights = get_phase_weights("late", 100, True)
        assert weights["safety"] > 1.0
        assert weights["aggression"] < 1.0

    def test_low_health_prioritizes_food(self):
        weights = get_phase_weights("mid", 20, True)
        assert weights["food"] > 1.0
        assert weights["aggression"] < 1.0

    def test_late_game_no_food(self):
        weights = get_phase_weights("late", 50, False)
        assert weights["food"] < 1.0


# =============================================================================
# Test Risk Tolerance
# =============================================================================

class TestRiskTolerance:
    def test_early_game_high_risk(self):
        risk = get_risk_tolerance("early", 100, 1, 4)
        assert risk > 0.5

    def test_late_game_low_risk(self):
        risk = get_risk_tolerance("late", 100, 1, 4)
        assert risk < 0.5

    def test_low_health_reduces_risk(self):
        risk_high = get_risk_tolerance("mid", 100, 2, 4)
        risk_low = get_risk_tolerance("mid", 20, 2, 4)
        assert risk_low < risk_high

    def test_behind_increases_risk(self):
        risk_leading = get_risk_tolerance("mid", 100, 1, 4)
        risk_behind = get_risk_tolerance("mid", 100, 4, 4)
        assert risk_behind > risk_leading

    def test_risk_bounded(self):
        risk = get_risk_tolerance("early", 100, 1, 1)
        assert 0.0 <= risk <= 1.0


# =============================================================================
# Test Food Chasing
# =============================================================================

class TestFoodChasing:
    def test_critical_health_always_chase(self):
        assert should_chase_food("early", 15, 10, True) == True

    def test_hungry_chases_close_food(self):
        assert should_chase_food("mid", 40, 3, True) == True

    def test_not_hungry_ignores_distant_food(self):
        assert should_chase_food("early", 100, 10, True) == False

    def test_no_food_no_chase(self):
        assert should_chase_food("mid", 50, 5, False) == False


# =============================================================================
# Test Enemy Avoidance
# =============================================================================

class TestEnemyAvoidance:
    def test_late_game_avoids(self):
        assert should_avoid_enemies("late", 100, 5, 5) == True

    def test_avoids_bigger_snakes(self):
        assert should_avoid_enemies("mid", 100, 5, 10) == True

    def test_low_health_avoids(self):
        assert should_avoid_enemies("mid", 30, 5, 5) == True

    def test_early_game_fights_equals(self):
        assert should_avoid_enemies("early", 100, 5, 5) == False


# =============================================================================
# Test Strategy Summary
# =============================================================================

class TestStrategySummary:
    def test_returns_complete_summary(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}],
            turn=10
        )
        summary = get_strategy_summary(state)
        
        assert "phase" in summary
        assert "turn" in summary
        assert "health" in summary
        assert "risk_tolerance" in summary
        assert "weights" in summary
        assert "strategy" in summary
        assert summary["phase"] == "early"

    def test_late_game_summary(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 50}],
            food=[],
            turn=100
        )
        summary = get_strategy_summary(state)
        assert summary["phase"] == "late"
        assert "survival" in summary["strategy"].lower() or "cautious" in summary["strategy"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
