# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Set up environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run locally (dev)
.venv/bin/python backend.py

# Run with gunicorn (production-like)
gunicorn backend:app --bind 0.0.0.0:8000

# Test with Battlesnake CLI
battlesnake play -W 11 -H 11 -n ml -u http://localhost:8000 -g solo -v -c -d 300
```

## Architecture

Two-file project:

- **`backend.py`** — Flask HTTP server. Implements the four Battlesnake engine endpoints (`GET /`, `POST /start`, `POST /move`, `POST /end`). Delegates all logic to `logic.py`.

- **`logic.py`** — All move selection logic. Has two layers:
  1. **Model path** (`choose_move_model`): scores each legal move using a standardized linear model embedded as `_MODEL` dict (feature names, per-feature mean/std for z-scoring, coefficients, intercept). Features are computed by `_candidate_features` per candidate move.
  2. **Heuristic fallback** (`choose_move_heuristic`): flood-fill for open space, head-to-head collision penalty, food steering when `health < 50`. Used if the model raises any exception.

### Board coordinate system

`(0, 0)` is **bottom-left**. Directions: `up` → y+1, `down` → y-1, `left` → x-1, `right` → x+1.

### Key model features

`voronoi` (Voronoi territory control) carries the highest coefficient weight. `h2h_danger` and `near_bigger_head` apply penalties. `reaches_tail` (anti-self-trap signal via BFS to own tail) also has significant weight.

### Deployment

Render reads `render.yaml` and runs `gunicorn backend:app --bind 0.0.0.0:$PORT`. The `/` endpoint must return valid appearance JSON (health check path).
