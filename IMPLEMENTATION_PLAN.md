# Battlesnake v2 — Implementation Plan

## 🎯 Goal
Transform the baseline bot into a **significantly stronger player** by adding:
1. **Enemy tail awareness** (predict future free space)
2. **2-move lookahead** (simulate next turn)
3. **Game phase strategy** (adapt to early/mid/late game)
4. **Improved wall/corner avoidance**

---

## 📁 Proposed File Structure

```
battlesnake-public-1/
├── backend.py              # HTTP server (unchanged)
├── logic.py                # Core decision logic (enhanced)
├── features.py             # NEW: Feature extraction functions
├── lookahead.py            # NEW: 2-move simulation engine
├── strategy.py             # NEW: Game phase strategy system
├── model.py                # NEW: Model definition and scoring
├── tests/
│   ├── __init__.py
│   ├── test_logic.py       # Unit tests for move selection
│   ├── test_features.py    # Feature extraction tests
│   └── test_lookahead.py   # Lookahead simulation tests
├── requirements.txt        # Updated dependencies
├── render.yaml             # Deployment config (unchanged)
└── README.md               # Updated documentation
```

---

## 🚀 Phase 1: Feature Engineering (Quick Wins)

### 1.1 New Features to Add

```python
# features.py — New feature extraction

def enemy_tail_features(state: Dict, move: str) -> Dict[str, float]:
    """Features about enemy tails (they free up next turn)."""
    enemies = [s for s in state["board"]["snakes"] if s["id"] != state["you"]["id"]]
    enemy_tails = [(s["body"][-1]["x"], s["body"][-1]["y"]) for s in enemies]
    nxt = _get_next_cell(state, move)
    
    return {
        "nearest_enemy_tail_dist": min(manhattan(nxt, t) for t in enemy_tails) if enemy_tails else 999,
        "tail_in_next_cell": 1.0 if nxt in enemy_tails else 0.0,  # FREE SPACE NEXT TURN!
        "tails_within_2": sum(1 for t in enemy_tails if manhattan(nxt, t) <= 2),
    }

def wall_corner_features(state: Dict, move: str) -> Dict[str, float]:
    """Improved wall and corner avoidance."""
    nxt = _get_next_cell(state, move)
    width, height = state["board"]["width"], state["board"]["height"]
    
    in_corner = (nxt[0] in [0, width-1]) and (nxt[1] in [0, height-1])
    on_edge = (nxt[0] in [0, width-1]) or (nxt[1] in [0, height-1])
    
    return {
        "in_corner": 1.0 if in_corner else 0.0,
        "on_edge": 1.0 if on_edge else 0.0,
        "corner_escape": _count_corner_escapes(nxt, state) if in_corner else 4.0,
    }

def game_phase_features(state: Dict) -> Dict[str, float]:
    """Features about game progression."""
    turn = state.get("turn", 1)
    food_count = len(state["board"]["food"])
    max_food = state["board"]["width"] * state["board"]["height"] // 10  # estimate
    
    return {
        "turn_number": float(turn),
        "food_scarcity": 1.0 - (food_count / max(max_food, 1)),
        "is_late_game": 1.0 if turn > 100 else 0.0,
        "is_early_game": 1.0 if turn < 20 else 0.0,
    }
```

### 1.2 Updated Feature List (13 → 20+)

| Category | Features |
|----------|----------|
| **Space** | `space_capped`, `open_space`, `voronoi` |
| **Safety** | `reaches_tail`, `escape`, `h2h_danger`, `in_corner`, `corner_escape` |
| **Enemies** | `near_bigger_head`, `near_enemy_head`, `nearest_enemy_tail_dist`, `tail_in_next_cell`, `tails_within_2` |
| **Walls** | `wall_dist`, `on_edge` |
| **Food** | `food_score`, `food_delta`, `is_food`, `food_scarcity` |
| **Position** | `dist_to_center` |
| **Game Phase** | `turn_number`, `is_late_game`, `is_early_game` |

---

## 🔮 Phase 2: 2-Move Lookahead

### 2.1 Lookahead Engine

```python
# lookahead.py — Simulate 2 moves ahead

def simulate_2_moves(state: Dict, move1: str, move2: str) -> Dict:
    """Simulate playing move1 then move2, return resulting state features."""
    simulated = _clone_state(state)
    
    # Apply move1
    _apply_move(simulated, move1)
    # Apply move2
    _apply_move(simulated, move2)
    
    return {
        "space_after_2": _flood_fill(simulated["you"]["head"], simulated["occupied"], ...),
        "health_after_2": simulated["you"]["health"],
        "is_trapped_after_2": _is_trapped(simulated),
        "food_reached": _count_food_reached(state, simulated),
    }

def lookahead_score(state: Dict, move: str) -> float:
    """Score a move by simulating all possible 2nd moves."""
    legal_next = _get_legal_moves_after(state, move)
    if not legal_next:
        return -1000  # Dead end
    
    # Score best possible continuation
    best_continuation = -float('inf')
    for move2 in legal_next:
        sim = simulate_2_moves(state, move, move2)
        score = (sim["space_after_2"] * 0.4 + 
                 sim["health_after_2"] * 0.3 +
                 (0 if sim["is_trapped_after_2"] else 100) * 0.3)
        best_continuation = max(best_continuation, score)
    
    return best_continuation
```

### 2.2 Integration into Main Logic

```python
# logic.py — Enhanced choose_move

def choose_move(state: Dict) -> str:
    legal = _legal_moves(state)
    if not legal:
        return "up"  # Desperation
    
    best_move = None
    best_score = -float('inf')
    
    for move in legal:
        # Base model score
        model_score = _model_score(state, move)
        
        # Lookahead bonus (weighted by game phase)
        phase = _get_game_phase(state)
        lookahead_weight = {
            "early": 0.3,
            "mid": 0.5,
            "late": 0.7  # More careful in late game
        }[phase]
        
        lookahead_bonus = lookahead_score(state, move) * lookahead_weight
        
        # Combined score
        total_score = model_score + lookahead_bonus
        
        if total_score > best_score:
            best_score = total_score
            best_move = move
    
    return best_move
```

---

## 🎮 Phase 3: Game Phase Strategy

### 3.1 Strategy System

```python
# strategy.py — Adaptive play style

class GamePhase:
    EARLY = "early"    # Turns 1-20
    MID = "mid"        # Turns 21-80
    LATE = "late"      # Turns 81+

def get_phase(state: Dict) -> str:
    turn = state.get("turn", 1)
    if turn <= 20:
        return GamePhase.EARLY
    elif turn <= 80:
        return GamePhase.MID
    else:
        return GamePhase.LATE

def get_strategy_weights(phase: str, health: int, rank: int) -> Dict[str, float]:
    """Return feature weights based on game phase and state."""
    base = {
        "safety": 0.4,
        "aggression": 0.3,
        "food": 0.2,
        "territory": 0.1,
    }
    
    if phase == GamePhase.EARLY:
        base["aggression"] += 0.2
        base["territory"] += 0.1
        base["safety"] -= 0.1
    elif phase == GamePhase.LATE:
        base["safety"] += 0.3
        base["aggression"] -= 0.2
        base["food"] -= 0.1
    
    # Adjust for health
    if health < 30:
        base["food"] += 0.3
        base["safety"] -= 0.1
    
    return base
```

---

## 🧪 Phase 4: Testing Framework

### 4.1 Test Suite Structure

```python
# tests/test_logic.py

import pytest
from logic import choose_move, choose_move_model

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

class TestBasicMoves:
    def test_returns_valid_direction(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, "body": [{"x": 5, "y": 5}], 
                     "length": 1, "health": 100}],
            food=[{"x": 6, "y": 5}]
        )
        move = choose_move(state)
        assert move in ["up", "down", "left", "right"]
    
    def test_avoids_walls(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 0, "y": 0}, "body": [{"x": 0, "y": 0}], 
                     "length": 1, "health": 100}],
            food=[]
        )
        move = choose_move(state)
        assert move in ["up", "right"]  # Can't go down or left
    
    def test_avoids_self_collision(self):
        state = create_game_state(
            snakes=[{"id": "me", "head": {"x": 5, "y": 5}, 
                     "body": [{"x": 5, "y": 5}, {"x": 4, "y": 5}], 
                     "length": 2, "health": 100}],
            food=[]
        )
        move = choose_move(state)
        assert move != "left"  # Would hit own body

class TestTailAwareness:
    def test_prefers_enemy_tail_cells(self):
        """Should prefer moving to enemy tail (frees next turn)."""
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
        # (6, 5) is enemy tail — should be preferred
        move = choose_move(state)
        assert move == "right"

class TestLookahead:
    def test_avoids_dead_ends(self):
        """Should avoid moves that lead to trapped positions."""
        # Create a scenario where one move leads to a corner trap
        pass  # Implementation depends on board setup

class TestGamePhase:
    def test_early_game_aggressive(self):
        """Early game should be more aggressive."""
        pass
    
    def test_late_game_cautious(self):
        """Late game should prioritize safety."""
        pass
```

### 4.2 Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=logic --cov=features --cov=lookahead
```

---

## 📋 Implementation Checklist

### Week 1: Foundation
- [ ] Create `features.py` with new feature extraction
- [ ] Create `model.py` with updated model definition
- [ ] Add new features to `_MODEL` dictionary
- [ ] Write unit tests for new features
- [ ] Verify baseline still works

### Week 2: Lookahead
- [ ] Create `lookahead.py` with simulation engine
- [ ] Implement state cloning and move application
- [ ] Integrate lookahead scoring into `choose_move`
- [ ] Write lookahead tests
- [ ] Benchmark performance impact

### Week 3: Strategy
- [ ] Create `strategy.py` with game phase system
- [ ] Implement adaptive weight adjustment
- [ ] Add strategy logs for debugging
- [ ] Test phase transitions

### Week 4: Testing & Tuning
- [ ] Complete test suite (80%+ coverage)
- [ ] Run headless simulations
- [ ] Tune feature weights
- [ ] Performance optimization
- [ ] Update README and documentation

---

## 📊 Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Win rate (solo) | ~50% | >70% |
| Average survival | ~40 turns | >60 turns |
| Food collected | ~15 | >25 |
| Illegal moves | 0% | 0% |
| Response time | <100ms | <150ms |

---

## 🔧 Deployment Notes

1. **Keep it lightweight**: No heavy ML libraries (numpy, sklearn)
2. **Pure Python**: Ensure compatibility with Render free tier
3. **Graceful degradation**: If lookahead fails, fall back to model
4. **Logging**: Add detailed logs for post-game analysis

---

## 🎯 First Step

**Start with Phase 1.1**: Add enemy tail features to `features.py`

This is the highest ROI improvement:
- Low effort (20 lines of code)
- High impact (predicts future free space)
- Easy to test
- No performance penalty

Want me to start implementing Phase 1?
