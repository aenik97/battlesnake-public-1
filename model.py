"""Model definition and scoring for the Battlesnake.

Combines the original baseline features with new enhanced features.
The model uses standardized linear scoring (z-score normalization).
"""

from typing import Dict, List, Optional

from features import extract_all_features

# =============================================================================
# Baseline Model (Original 13 features)
# =============================================================================

_BASELINE_MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}


# =============================================================================
# Enhanced Model (New features — weights to be tuned)
# =============================================================================

# Initial weights for new features (hand-tuned estimates)
# These should be replaced with trained weights after collecting game data.
_ENHANCED_FEATURES: Dict = {
    "feature_names": [
        # Tail features (high priority — predict free space)
        "nearest_enemy_tail_dist",
        "tail_in_next_cell",
        "tails_within_2",
        "tails_within_3",
        # Wall/corner features (safety)
        "in_corner",
        "on_edge",
        "wall_dist",
        "corner_escapes",
        # Game phase features (context)
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
    ],
    # Estimated means (will be updated with real data)
    "mean": [
        5.0,    # nearest_enemy_tail_dist
        0.1,    # tail_in_next_cell
        0.5,    # tails_within_2
        1.0,    # tails_within_3
        0.1,    # in_corner
        0.3,    # on_edge
        2.0,    # wall_dist
        3.5,    # corner_escapes
        50.0,   # turn_number
        0.5,    # food_scarcity
        3.0,    # alive_snakes
        0.3,    # is_early_game
        0.5,    # is_mid_game
        0.2,    # is_late_game
        5.0,    # nearest_food_dist
        0.0,    # food_delta
        0.1,    # is_food
        1.0,    # food_near_enemies
        3.0,    # safe_food_count
        5.0,    # nearest_enemy_head
        7.0,    # nearest_bigger_head
        0.3,    # enemy_heads_within_2
        0.8,    # enemy_heads_within_3
    ],
    # Estimated stds
    "std": [
        3.0,
        0.3,
        0.8,
        1.2,
        0.3,
        0.5,
        1.5,
        0.8,
        30.0,
        0.3,
        1.5,
        0.5,
        0.5,
        0.4,
        3.0,
        1.0,
        0.3,
        1.0,
        2.0,
        3.0,
        4.0,
        0.5,
        1.0,
    ],
    # Hand-tuned coefficients (positive = good, negative = bad)
    "coef": [
        -0.5,   # nearest_enemy_tail_dist (closer is better, so negative)
        15.0,   # tail_in_next_cell (FREE SPACE! very good)
        2.0,    # tails_within_2
        1.0,    # tails_within_3
        -20.0,  # in_corner (bad — limited escapes)
        -5.0,   # on_edge (somewhat risky)
        1.5,    # wall_dist (farther from walls is safer)
        3.0,    # corner_escapes (more escapes is good)
        0.0,    # turn_number (neutral — handled by phase flags)
        3.0,    # food_scarcity (scarcity → prioritize food)
        0.0,    # alive_snakes (neutral)
        2.0,    # is_early_game (aggressive phase)
        0.0,    # is_mid_game
        5.0,    # is_late_game (cautious phase)
        -1.0,   # nearest_food_dist (closer is better)
        2.0,    # food_delta (getting closer is good)
        8.0,    # is_food (eating is great)
        -3.0,   # food_near_enemies (risky food)
        1.0,    # safe_food_count
        0.5,    # nearest_enemy_head (neutral-slight positive)
        -1.5,   # nearest_bigger_head (avoid bigger snakes)
        -5.0,   # enemy_heads_within_2 (danger!)
        -2.0,   # enemy_heads_within_3
    ],
    "intercept": 0.0,
}


# =============================================================================
# Model Scoring
# =============================================================================

def score_move_baseline(baseline_feats: Dict[str, float]) -> float:
    """Score a move using the original baseline model."""
    names = _BASELINE_MODEL["feature_names"]
    mean = _BASELINE_MODEL["mean"]
    std = _BASELINE_MODEL["std"]
    coef = _BASELINE_MODEL["coef"]
    intercept = _BASELINE_MODEL["intercept"]
    
    score = intercept
    for i, name in enumerate(names):
        value = baseline_feats.get(name, 0.0)
        z = (value - mean[i]) / std[i] if std[i] else 0.0
        score += coef[i] * z
    return score


def score_move_enhanced(enanced_feats: Dict[str, float]) -> float:
    """Score a move using the new enhanced features."""
    names = _ENHANCED_FEATURES["feature_names"]
    mean = _ENHANCED_FEATURES["mean"]
    std = _ENHANCED_FEATURES["std"]
    coef = _ENHANCED_FEATURES["coef"]
    intercept = _ENHANCED_FEATURES["intercept"]
    
    score = intercept
    for i, name in enumerate(names):
        value = enanced_feats.get(name, 0.0)
        z = (value - mean[i]) / std[i] if std[i] else 0.0
        score += coef[i] * z
    return score


def combined_score(baseline_feats: Dict[str, float], enhanced_feats: Dict[str, float],
                   baseline_weight: float = 0.7, enhanced_weight: float = 0.3) -> float:
    """Combine baseline and enhanced model scores.
    
    Args:
        baseline_feats: Original 13 features
        enhanced_feats: New 24 features
        baseline_weight: Weight for baseline model (default 0.7)
        enhanced_weight: Weight for enhanced model (default 0.3)
    
    Returns:
        Combined score
    """
    base_score = score_move_baseline(baseline_feats)
    enh_score = score_move_enhanced(enhanced_feats)
    
    return baseline_weight * base_score + enhanced_weight * enh_score


# =============================================================================
# Feature Importance Reference
# =============================================================================

def get_feature_importance() -> Dict[str, float]:
    """Return feature importance (absolute coefficient values)."""
    importance = {}
    for model, name in [(_BASELINE_MODEL, "baseline"), (_ENHANCED_FEATURES, "enhanced")]:
        for i, feat_name in enumerate(model["feature_names"]):
            importance[f"{name}_{feat_name}"] = abs(model["coef"][i])
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
