"""Game phase strategy system for the Battlesnake.

Adapts play style based on game progression:
- Early game (turns 1-20): Aggressive expansion, fight for center
- Mid game (turns 21-80): Territory control, force collisions
- Late game (turns 81+): Risk-averse, survive, let others eliminate
"""

from typing import Dict, Tuple

# Game phase definitions
EARLY_GAME_END = 20
MID_GAME_END = 80


def get_game_phase(turn: int) -> str:
    """Determine current game phase based on turn number."""
    if turn <= EARLY_GAME_END:
        return "early"
    elif turn <= MID_GAME_END:
        return "mid"
    else:
        return "late"


def get_phase_weights(phase: str, health: int, food_available: bool) -> Dict[str, float]:
    """Get scoring weight adjustments for the current game phase.
    
    Returns multipliers for different feature categories:
    - safety: How much to prioritize avoiding danger
    - aggression: How much to fight for position/territory
    - food: How much to prioritize eating
    - territory: How much to control open space
    - lookahead: How much to plan ahead
    
    Args:
        phase: "early", "mid", or "late"
        health: Current snake health (0-100)
        food_available: Whether there's food on the board
    """
    # Base weights for each phase
    base_weights = {
        "early": {
            "safety": 0.8,      # Still safe but willing to take risks
            "aggression": 1.3,  # More aggressive
            "food": 0.7,        # Less urgent (plenty of time)
            "territory": 1.2,   # Fight for center
            "lookahead": 0.8,   # Less planning needed
        },
        "mid": {
            "safety": 1.0,      # Balanced
            "aggression": 1.0,  # Balanced
            "food": 1.0,        # Normal priority
            "territory": 1.0,   # Maintain position
            "lookahead": 1.0,   # Normal planning
        },
        "late": {
            "safety": 1.5,      # Very cautious
            "aggression": 0.5,  # Avoid fights
            "food": 1.3 if food_available else 0.5,  # Food critical if available
            "territory": 0.7,   # Less important
            "lookahead": 1.5,   # Plan carefully
        },
    }
    
    weights = base_weights.get(phase, base_weights["mid"]).copy()
    
    # Adjust for health (low health = more food-focused, less aggressive)
    if health < 30:
        weights["food"] *= 1.5
        weights["aggression"] *= 0.6
        weights["safety"] *= 1.2
    elif health < 50:
        weights["food"] *= 1.2
        weights["aggression"] *= 0.8
    
    return weights


def get_risk_tolerance(phase: str, health: int, rank: int, total_snakes: int) -> float:
    """Calculate risk tolerance (0.0 = very cautious, 1.0 = very aggressive).
    
    Args:
        phase: Current game phase
        health: Current health (0-100)
        rank: Current rank (1 = leading)
        total_snakes: Number of snakes still alive
    """
    # Base risk by phase
    base_risk = {
        "early": 0.7,   # Take more risks early
        "mid": 0.5,     # Moderate risks
        "late": 0.3,    # Play safe late
    }
    
    risk = base_risk.get(phase, 0.5)
    
    # Adjust for health
    risk *= (health / 100.0)
    
    # Adjust for rank (leading = can be safer, behind = must take risks)
    rank_ratio = rank / max(total_snakes, 1)
    if rank_ratio > 0.7:  # Behind
        risk *= 1.3
    elif rank_ratio < 0.3:  # Leading
        risk *= 0.8
    
    return max(0.0, min(1.0, risk))


def should_chase_food(phase: str, health: int, nearest_food_dist: int, food_available: bool) -> bool:
    """Decide whether to actively chase food.
    
    Args:
        phase: Current game phase
        health: Current health
        nearest_food_dist: Distance to nearest food
        food_available: Whether any food exists
    """
    if not food_available:
        return False
    
    # Always chase if critical health
    if health < 20:
        return True
    
    # Chase if hungry and food is close
    if health < 50 and nearest_food_dist <= 3:
        return True
    
    # Late game: chase food more aggressively
    if phase == "late" and health < 70:
        return True
    
    # Early game: only chase if very close
    if phase == "early":
        return nearest_food_dist <= 2
    
    # Mid game: chase if reasonably close
    return nearest_food_dist <= 4


def should_avoid_enemies(phase: str, health: int, my_length: int, nearest_enemy_length: int) -> bool:
    """Decide whether to avoid enemy confrontations.
    
    Args:
        phase: Current game phase
        health: Current health
        my_length: My snake length
        nearest_enemy_length: Length of nearest enemy
    """
    # Late game: avoid fights unless necessary
    if phase == "late":
        return True
    
    # Avoid bigger snakes always
    if nearest_enemy_length > my_length * 1.2:
        return True
    
    # Low health: avoid fights
    if health < 40:
        return True
    
    return False


def get_strategy_summary(state: Dict) -> Dict:
    """Get a complete strategy summary for the current game state.
    
    Useful for logging and debugging.
    """
    turn = state.get("turn", 1)
    you = state["you"]
    board = state["board"]
    
    phase = get_game_phase(turn)
    health = you["health"]
    food_available = len(board["food"]) > 0
    
    weights = get_phase_weights(phase, health, food_available)
    
    # Calculate rank (simplified: by length)
    snakes = board["snakes"]
    my_length = you["length"]
    rank = sum(1 for s in snakes if s["length"] > my_length) + 1
    
    risk = get_risk_tolerance(phase, health, rank, len(snakes))
    
    return {
        "phase": phase,
        "turn": turn,
        "health": health,
        "rank": rank,
        "total_snakes": len(snakes),
        "risk_tolerance": round(risk, 2),
        "weights": {k: round(v, 2) for k, v in weights.items()},
        "should_chase_food": should_chase_food(
            phase, health,
            min((abs(you["head"]["x"] - f["x"]) + abs(you["head"]["y"] - f["y"])) 
                for f in board["food"]) if board["food"] else 999,
            food_available
        ),
        "strategy": _get_strategy_description(phase, risk, weights),
    }


def _get_strategy_description(phase: str, risk: float, weights: Dict[str, float]) -> str:
    """Get a human-readable strategy description."""
    if phase == "early":
        if risk > 0.6:
            return "Aggressive expansion — fight for center control"
        return "Cautious expansion — build safe territory"
    elif phase == "mid":
        if weights["territory"] > 1.0:
            return "Territory control — dominate open space"
        return "Balanced play — maintain position, seek food"
    else:  # late
        if risk < 0.4:
            return "Survival mode — avoid all risks, let others fight"
        return "Cautious survival — minimal risks, prioritize food"
