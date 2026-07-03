"""Enhanced feature extraction for the Battlesnake ML model.

This module provides additional features beyond the baseline:
- Enemy tail awareness (predicts future free space)
- Wall/corner danger detection
- Game phase indicators
- Food scarcity metrics
"""

from typing import Dict, List, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def _manhattan(a: Point, b: Point) -> int:
    """Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _in_bounds(p: Point, width: int, height: int) -> bool:
    """Check if point is within board bounds."""
    return 0 <= p[0] < width and 0 <= p[1] < height


def _get_next_cell(state: Dict, move: str) -> Point:
    """Get the cell coordinates after making a move."""
    head = (state["you"]["head"]["x"], state["you"]["head"]["y"])
    dx, dy = DIRECTIONS[move]
    return (head[0] + dx, head[1] + dy)


def _get_enemy_tails(state: Dict) -> List[Point]:
    """Get all enemy tail positions."""
    enemies = [s for s in state["board"]["snakes"] if s["id"] != state["you"]["id"]]
    return [(s["body"][-1]["x"], s["body"][-1]["y"]) for s in enemies]


def _get_enemy_heads(state: Dict) -> List[Point]:
    """Get all enemy head positions."""
    enemies = [s for s in state["board"]["snakes"] if s["id"] != state["you"]["id"]]
    return [(s["head"]["x"], s["head"]["y"]) for s in enemies]


def _get_bigger_enemy_heads(state: Dict) -> List[Point]:
    """Get enemy heads that are >= our length (collision danger)."""
    enemies = [s for s in state["board"]["snakes"] if s["id"] != state["you"]["id"]]
    my_length = state["you"]["length"]
    return [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]


# =============================================================================
# Enemy Tail Features
# =============================================================================

def enemy_tail_features(state: Dict, move: str) -> Dict[str, float]:
    """Features about enemy tails (they free up next turn).
    
    Moving toward an enemy tail is smart because:
    - The tail cell will be free next turn
    - Can force enemies into tight spaces
    - Good for chasing and trapping
    """
    nxt = _get_next_cell(state, move)
    enemy_tails = _get_enemy_tails(state)
    
    if not enemy_tails:
        return {
            "nearest_enemy_tail_dist": 999.0,
            "tail_in_next_cell": 0.0,
            "tails_within_2": 0.0,
            "tails_within_3": 0.0,
        }
    
    distances = [_manhattan(nxt, tail) for tail in enemy_tails]
    
    return {
        "nearest_enemy_tail_dist": float(min(distances)),
        "tail_in_next_cell": 1.0 if nxt in enemy_tails else 0.0,
        "tails_within_2": float(sum(1 for d in distances if d <= 2)),
        "tails_within_3": float(sum(1 for d in distances if d <= 3)),
    }


# =============================================================================
# Wall and Corner Features
# =============================================================================

def wall_corner_features(state: Dict, move: str) -> Dict[str, float]:
    """Improved wall and corner avoidance features.
    
    Corners are dangerous because they limit escape routes.
    Edges are risky but sometimes necessary.
    """
    nxt = _get_next_cell(state, move)
    width, height = state["board"]["width"], state["board"]["height"]
    
    # Check corner/edge status
    is_corner_x = nxt[0] in [0, width - 1]
    is_corner_y = nxt[1] in [0, height - 1]
    in_corner = is_corner_x and is_corner_y
    on_edge = is_corner_x or is_corner_y
    
    # Distance to nearest wall
    wall_dist = min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])
    
    # Count escape routes from this cell
    escapes = sum(
        1
        for dx, dy in _NEIGHBORS
        if _in_bounds((nxt[0] + dx, nxt[1] + dy), width, height)
    )
    
    return {
        "in_corner": 1.0 if in_corner else 0.0,
        "on_edge": 1.0 if on_edge else 0.0,
        "wall_dist": float(wall_dist),
        "corner_escapes": float(escapes),  # 2 if corner, 3 if edge, 4 if center
    }


# =============================================================================
# Game Phase Features
# =============================================================================

def game_phase_features(state: Dict) -> Dict[str, float]:
    """Features about game progression.
    
    Different strategies work better at different game phases:
    - Early: Expand territory, be aggressive
    - Mid: Control space, force collisions
    - Late: Survive, be cautious
    """
    turn = state.get("turn", 1)
    food_count = len(state["board"]["food"])
    width, height = state["board"]["width"], state["board"]["height"]
    
    # Estimate max food based on board size (typically ~10% of cells)
    max_food = (width * height) // 10
    
    # Count alive snakes
    alive_snakes = len(state["board"]["snakes"])
    
    return {
        "turn_number": float(turn),
        "food_scarcity": 1.0 - min(food_count / max(max_food, 1), 1.0),
        "alive_snakes": float(alive_snakes),
        "is_early_game": 1.0 if turn <= 20 else 0.0,
        "is_mid_game": 1.0 if 20 < turn <= 80 else 0.0,
        "is_late_game": 1.0 if turn > 80 else 0.0,
    }


# =============================================================================
# Food Strategy Features
# =============================================================================

def food_strategy_features(state: Dict, move: str) -> Dict[str, float]:
    """Advanced food-related features.
    
    Not all food is equal:
    - Food in open space is safer to reach
    - Food near enemies is risky
    - Multiple food sources matter
    """
    nxt = _get_next_cell(state, move)
    head = (state["you"]["head"]["x"], state["you"]["head"]["y"])
    foods = [(f["x"], f["y"]) for f in state["board"]["food"]]
    enemy_heads = _get_enemy_heads(state)
    
    if not foods:
        return {
            "nearest_food_dist": 999.0,
            "food_delta": 0.0,
            "is_food": 0.0,
            "food_near_enemies": 0.0,
            "safe_food_count": 0.0,
        }
    
    nearest_now = min(_manhattan(head, f) for f in foods)
    nearest_next = min(_manhattan(nxt, f) for f in foods)
    
    # Count food near enemy heads (risky)
    food_near_enemies = sum(
        1 for f in foods
        if any(_manhattan(f, e) <= 2 for e in enemy_heads)
    )
    
    # Count food in relatively safe areas (far from enemies)
    safe_food = sum(
        1 for f in foods
        if all(_manhattan(f, e) > 3 for e in enemy_heads)
    )
    
    return {
        "nearest_food_dist": float(nearest_next),
        "food_delta": float(nearest_now - nearest_next),
        "is_food": 1.0 if nxt in foods else 0.0,
        "food_near_enemies": float(food_near_enemies),
        "safe_food_count": float(safe_food),
    }


# =============================================================================
# Enemy Proximity Features
# =============================================================================

def enemy_proximity_features(state: Dict, move: str) -> Dict[str, float]:
    """Detailed enemy proximity features.
    
    Tracks distance to various enemy elements for threat assessment.
    """
    nxt = _get_next_cell(state, move)
    enemy_heads = _get_enemy_heads(state)
    bigger_heads = _get_bigger_enemy_heads(state)
    enemy_tails = _get_enemy_tails(state)
    
    width, height = state["board"]["width"], state["board"]["height"]
    max_dist = width + height
    
    if not enemy_heads:
        return {
            "nearest_enemy_head": float(max_dist),
            "nearest_bigger_head": float(max_dist),
            "enemy_heads_within_2": 0.0,
            "enemy_heads_within_3": 0.0,
        }
    
    head_distances = [_manhattan(nxt, h) for h in enemy_heads]
    bigger_distances = [_manhattan(nxt, h) for h in bigger_heads] if bigger_heads else [max_dist]
    
    return {
        "nearest_enemy_head": float(min(head_distances)),
        "nearest_bigger_head": float(min(bigger_distances)),
        "enemy_heads_within_2": float(sum(1 for d in head_distances if d <= 2)),
        "enemy_heads_within_3": float(sum(1 for d in head_distances if d <= 3)),
    }


# =============================================================================
# Combined Feature Extractor
# =============================================================================

def extract_all_features(state: Dict, move: str) -> Dict[str, float]:
    """Extract all features for a given move.
    
    Combines baseline features with new enhanced features.
    """
    # New features
    tail_feats = enemy_tail_features(state, move)
    wall_feats = wall_corner_features(state, move)
    phase_feats = game_phase_features(state)
    food_feats = food_strategy_features(state, move)
    enemy_feats = enemy_proximity_features(state, move)
    
    return {
        **tail_feats,
        **wall_feats,
        **phase_feats,
        **food_feats,
        **enemy_feats,
    }


def get_feature_names() -> List[str]:
    """Return list of all feature names in order."""
    return [
        # Tail features
        "nearest_enemy_tail_dist",
        "tail_in_next_cell",
        "tails_within_2",
        "tails_within_3",
        # Wall/corner features
        "in_corner",
        "on_edge",
        "wall_dist",
        "corner_escapes",
        # Game phase features
        "turn_number",
        "food_scarcity",
        "alive_snakes",
        "is_early_game",
        "is_mid_game",
        "is_late_game",
        # Food features
        "nearest_food_dist",
        "food_delta",
        "is_food",
        "food_near_enemies",
        "safe_food_count",
        # Enemy features
        "nearest_enemy_head",
        "nearest_bigger_head",
        "enemy_heads_within_2",
        "enemy_heads_within_3",
    ]
