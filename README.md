# Battlesnake Deterministic Search Bot

A [Battlesnake](https://play.battlesnake.com) written in Python and Flask. This
version uses a time-bounded deterministic engine as the primary policy, with the
older embedded linear model kept as a fallback.

## What It Does

Each turn, `logic.py`:

- Calls `risk_ai_challenge.engine.choose_move`.
- Filters unsafe moves.
- Scores flood-fill territory, Voronoi/area control, food paths, length, health,
  and head-to-head threats.
- Uses A* for reachable food distance.
- Spends remaining budget on shallow iterative search: alpha-beta for two-snake
  boards and MaxN for multi-snake boards.
- Falls back to the embedded pure-Python linear model and then a simple
  heuristic if anything unexpected happens.


## Files

- `backend.py` — Battlesnake HTTP server with `/`, `/start`, `/move`, and `/end`.
- `logic.py` — compatibility entrypoint used by `backend.py`, plus model fallback.
- `risk_ai_challenge/engine.py` — deterministic search, flood fill, A*, Voronoi,
  alpha-beta, MaxN, and board simulation.
- `tests/` — unit tests for safety, pathfinding, territory, and collision rules.
- `requirements.txt` — runtime dependencies.
- `render.yaml` — Render deployment config.

## Run Locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backend.py
```

Test your battlesnake with the Battlesnake CLI:

```bash
battlesnake play -W 11 -H 11 \
  -n ml -u http://localhost:8000 \
  -g solo \
  -v -c -d 300
```

## Fetch Public Game Recordings

Recent leaderboard games can be downloaded without extra dependencies:

```bash
python3 scripts/fetch_battlesnake_games.py \
  --leaderboard standard \
  --player andreammm \
  --limit 3 \
  --out data/games
```

To download recent games for the top Standard leaderboard players:

```bash
python3 scripts/fetch_battlesnake_games.py \
  --leaderboard standard \
  --top-players 25 \
  --limit 20 \
  --out data/games
```

Or fetch a known game directly:

```bash
python3 scripts/fetch_battlesnake_games.py \
  --game-id b689f661-6a6d-4ec9-9588-cbb7c09a7517 \
  --out data/games
```

The script discovers `/game/<uuid>` links from
`/leaderboard/<leaderboard>/<player>/stats`, loads game metadata from the
engine REST endpoint, then records all websocket `frame` events into JSON.
Existing JSON files are skipped by default, so repeated runs resume cleanly.

## Visual Replay Tool

Render a downloaded game recording as a self-contained HTML replay:

```bash
python3 scripts/replay_game.py \
  --game-json data/games/b689f661-6a6d-4ec9-9588-cbb7c09a7517.json
```

The generated file is written to `outputs/<game-id>-replay.html` by default.
It includes play/pause, a turn scrubber, speed control, food, hazards, snake
colors, health, length, latency, inferred moves, and death details.

You can also generate a local four-snake scrimmage replay without Battlesnake
CLI or Flask:

```bash
python3 scripts/replay_game.py \
  --simulate-four \
  --max-turns 80 \
  --move-time-limit-ms 80 \
  --out outputs/local-four-snake-replay.html
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
