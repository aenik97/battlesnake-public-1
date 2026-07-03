# Battlesnake ML Inference Bot v2

A [Battlesnake](https://play.battlesnake.com) written in Python and Flask. This
version uses a **combined scoring system** with enhanced features and lookahead.

## What It Does

Each turn, `logic.py`:

- Gets legal moves for the current board.
- Calculates **baseline features** (13 original features).
- Calculates **enhanced features** (23 new features: tail awareness, wall safety, game phase).
- Runs **2-move lookahead simulation** to avoid traps.
- Combines all scores and returns the highest-scoring move.

## Architecture

### Scoring System (Weighted Combination)

| Component | Weight | Description |
|-----------|--------|-------------|
| Baseline Model | 60% | Original linear model (99.3% accuracy) |
| Enhanced Features | 30% | 23 new features for better decision-making |
| Lookahead | 10% | 2-move simulation to avoid dead ends |

### Enhanced Features (23 total)

**Enemy Tail Awareness (4):**
- `nearest_enemy_tail_dist` — distance to closest enemy tail
- `tail_in_next_cell` — is next cell an enemy tail? (frees next turn!)
- `tails_within_2`, `tails_within_3` — nearby tail count

**Wall/Corner Safety (4):**
- `in_corner`, `on_edge` — position danger flags
- `wall_dist` — distance to nearest wall
- `corner_escapes` — available escape routes

**Game Phase (6):**
- `turn_number`, `food_scarcity`, `alive_snakes`
- `is_early_game`, `is_mid_game`, `is_late_game`

**Food Strategy (5):**
- `nearest_food_dist`, `food_delta`, `is_food`
- `food_near_enemies`, `safe_food_count`

**Enemy Proximity (4):**
- `nearest_enemy_head`, `nearest_bigger_head`
- `enemy_heads_within_2`, `enemy_heads_within_3`

## Files

- `backend.py` — Battlesnake HTTP server with `/`, `/start`, `/move`, and `/end`.
- `logic.py` — Combined scoring: baseline + enhanced + lookahead + fallback.
- `features.py` — Enhanced feature extraction (23 new features).
- `model.py` — Model definitions and scoring functions.
- `lookahead.py` — 2-move simulation engine.
- `tests/` — Unit tests for features and lookahead.
- `requirements.txt` — Runtime dependencies.
- `render.yaml` — Render deployment config.

## Run Locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backend.py
```

### Run Tests

```bash
# Run all tests
PYTHONPATH=. python -m pytest tests/ -v

# Run specific test file
PYTHONPATH=. python -m pytest tests/test_features.py -v
```

### Test Your Battlesnake

With the Battlesnake CLI:

```bash
battlesnake play -W 11 -H 11 \
  -n ml -u http://localhost:8000 \
  -g solo \
  -v -c -d 300
```

## Deploy to Render

1. Push this repo to GitHub.
2. In the [Render dashboard](https://dashboard.render.com): **New -> Blueprint**,
   connect the repo. Render reads `render.yaml` and provisions a free web
   service running `gunicorn backend:app`.
   - Or **New -> Web Service** manually with build command
     `pip install -r requirements.txt` and start command
     `gunicorn backend:app --bind 0.0.0.0:$PORT`.
3. Wait for the deploy to go live. Note the public URL, e.g.
   `https://battlesnake-xxxx.onrender.com`.
4. Visit that URL in a browser — you should see the appearance JSON.

## Register on Battlesnake

1. Create an account at [play.battlesnake.com](https://play.battlesnake.com).
2. **Create Battlesnake** -> paste your Render URL as the server URL.
3. Now you can use it in a game!

## Tuning

Adjust scoring weights in `logic.py`:

```python
BASELINE_WEIGHT = 0.6    # Original model
ENHANCED_WEIGHT = 0.3    # New features
LOOKAHEAD_WEIGHT = 0.1   # Simulation
```

Adjust feature coefficients in `model.py` `_ENHANCED_FEATURES["coef"]`.
