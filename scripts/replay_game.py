#!/usr/bin/env python3
"""Render Battlesnake recordings or local scrimmages as an interactive HTML replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    args = parse_args()
    if args.simulate_four:
        replay = build_local_scrimmage_replay(args)
        default_name = "local-four-snake-replay.html"
    else:
        game_json = args.game_json or find_default_game_json()
        replay = build_recording_replay(game_json)
        default_name = f"{game_json.stem}-replay.html"

    out = args.out or (ROOT / "outputs" / default_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(replay), encoding="utf-8")
    print(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive Battlesnake replay HTML file."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--game-json",
        type=Path,
        help="Recording JSON from scripts/fetch_battlesnake_games.py.",
    )
    source.add_argument(
        "--simulate-four",
        action="store_true",
        help="Generate a local four-snake scrimmage replay using this bot engine.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output HTML path. Defaults to outputs/<game-id>-replay.html.",
    )
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--move-time-limit-ms",
        type=int,
        default=80,
        help="Local scrimmage per-snake engine budget. Real games use recorded frames.",
    )
    return parser.parse_args()


def find_default_game_json() -> Path:
    games = sorted((ROOT / "data" / "games").glob("*.json"))
    if not games:
        raise SystemExit("No data/games/*.json found. Pass --simulate-four or --game-json.")
    return games[0]


def build_recording_replay(path: Path) -> dict[str, Any]:
    recording = json.loads(path.read_text(encoding="utf-8"))
    raw_frames = recording.get("frames")
    if not raw_frames:
        raw_frames = [
            event["Data"]
            for event in recording.get("events", [])
            if event.get("Type") == "frame" and event.get("Data")
        ]
    if not raw_frames:
        raise SystemExit(f"No frames found in {path}")

    game = recording.get("metadata", {}).get("Game", {})
    width = int(game.get("Width") or infer_dimension(raw_frames, "X"))
    height = int(game.get("Height") or infer_dimension(raw_frames, "Y"))
    frames = [
        normalize_engine_frame(raw_frame, previous=raw_frames[index - 1] if index else None)
        for index, raw_frame in enumerate(raw_frames)
    ]

    return {
        "title": recording.get("game_id") or path.stem,
        "source": str(path),
        "gameUrl": recording.get("game_url", ""),
        "width": width,
        "height": height,
        "frames": frames,
    }


def build_local_scrimmage_replay(args: argparse.Namespace) -> dict[str, Any]:
    from risk_ai_challenge.engine import choose_move
    from scripts import simulate_four_snakes as sim

    config = sim.ScrimmageConfig(
        max_turns=args.max_turns,
        seed=args.seed,
        move_time_limit_ms=args.move_time_limit_ms,
    )
    sim.random.seed(config.seed)
    state = sim.initial_state(config)
    policies = {
        snake["id"]: lambda game_state, limit=config.move_time_limit_ms: choose_move(
            game_state, time_limit_ms=limit
        )
        for snake in state["board"]["snakes"]
    }

    frames = [normalize_local_frame(state, moves={})]
    for _turn in range(config.max_turns):
        if len(state["board"]["snakes"]) <= 1:
            break
        moves = sim.choose_all_moves(state, policies)
        state = sim.advance(state, moves, config)
        frames.append(normalize_local_frame(state, moves=moves))

    return {
        "title": "local four-snake scrimmage",
        "source": "simulate-four",
        "gameUrl": "",
        "width": config.width,
        "height": config.height,
        "frames": frames,
    }


def normalize_engine_frame(raw_frame: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    previous_heads = {}
    if previous:
        previous_heads = {
            snake["ID"]: point_from_engine(snake["Body"][0])
            for snake in previous.get("Snakes", [])
            if snake.get("Body")
        }

    moves = {}
    snakes = []
    for snake in raw_frame.get("Snakes", []):
        body = [point_from_engine(part) for part in snake.get("Body", [])]
        snake_id = snake.get("ID", "")
        if body and snake_id in previous_heads:
            moves[snake_id] = infer_move(previous_heads[snake_id], body[0])
        snakes.append(
            {
                "id": snake_id,
                "name": snake.get("Name") or snake_id,
                "author": snake.get("Author", ""),
                "color": normalize_color(snake.get("Color"), snake_id),
                "health": snake.get("Health"),
                "length": len(body),
                "latency": snake.get("Latency", ""),
                "death": snake.get("Death"),
                "body": body,
            }
        )

    return {
        "turn": raw_frame.get("Turn", 0),
        "food": [point_from_engine(part) for part in raw_frame.get("Food", [])],
        "hazards": [point_from_engine(part) for part in raw_frame.get("Hazards", [])],
        "snakes": snakes,
        "moves": moves,
    }


def normalize_local_frame(state: dict[str, Any], moves: dict[str, str]) -> dict[str, Any]:
    return {
        "turn": state["turn"],
        "food": [dict(part) for part in state["board"]["food"]],
        "hazards": [],
        "moves": dict(moves),
        "snakes": [
            {
                "id": snake["id"],
                "name": snake.get("name", snake["id"]),
                "author": "",
                "color": normalize_color(None, snake["id"]),
                "health": snake["health"],
                "length": snake["length"],
                "latency": "",
                "death": None,
                "body": [dict(part) for part in snake["body"]],
            }
            for snake in state["board"]["snakes"]
        ],
    }


def point_from_engine(part: dict[str, Any]) -> dict[str, int]:
    return {"x": int(part["X"]), "y": int(part["Y"])}


def infer_dimension(frames: list[dict[str, Any]], axis: str) -> int:
    max_value = 0
    for frame in frames:
        for part in frame.get("Food", []) + frame.get("Hazards", []):
            max_value = max(max_value, int(part.get(axis, 0)))
        for snake in frame.get("Snakes", []):
            for part in snake.get("Body", []):
                max_value = max(max_value, int(part.get(axis, 0)))
    return max_value + 1


def infer_move(previous: dict[str, int], current: dict[str, int]) -> str:
    dx = current["x"] - previous["x"]
    dy = current["y"] - previous["y"]
    if dx == 1:
        return "right"
    if dx == -1:
        return "left"
    if dy == 1:
        return "up"
    if dy == -1:
        return "down"
    return "-"


def normalize_color(color: str | None, seed: str) -> str:
    if isinstance(color, str) and color.startswith("#") and len(color) in (4, 7):
        return color
    palette = ["#29b6f6", "#ff6b6b", "#65d36e", "#c084fc", "#f2c94c", "#f97316"]
    return palette[sum(ord(char) for char in seed) % len(palette)]


def render_html(replay: dict[str, Any]) -> str:
    data = json.dumps(replay, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Battlesnake Replay</title>
  <style>
    :root {{
      --bg: #101114;
      --panel: #1a1d22;
      --panel-2: #22262d;
      --line: #333945;
      --text: #f4f6fa;
      --muted: #aab2c0;
      --food: #f2c94c;
      --hazard: #7f1d1d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1280px, calc(100vw - 28px));
      margin: 0 auto;
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 780px) minmax(300px, 1fr);
      gap: 18px;
      align-items: center;
      padding: 18px 0;
    }}
    .board {{
      display: grid;
      grid-template-columns: repeat(var(--board-w), 1fr);
      aspect-ratio: var(--board-w) / var(--board-h);
      width: 100%;
      border: 1px solid var(--line);
      background: #15171b;
    }}
    .cell {{
      position: relative;
      border-right: 1px solid rgba(255,255,255,.055);
      border-bottom: 1px solid rgba(255,255,255,.055);
      min-width: 0;
    }}
    .cell::after {{
      content: "";
      position: absolute;
      inset: 18%;
      border-radius: 4px;
      background: transparent;
    }}
    .food::after {{
      inset: 26%;
      border-radius: 50%;
      background: var(--food);
      box-shadow: 0 0 14px rgba(242,201,76,.48);
    }}
    .hazard {{ background: rgba(127,29,29,.48); }}
    .body::after {{ background: var(--snake-color); opacity: .62; }}
    .head::after {{
      background: var(--snake-color);
      inset: 9%;
      opacity: 1;
      border: 2px solid rgba(255,255,255,.82);
      box-shadow: 0 0 18px rgba(255,255,255,.18);
    }}
    aside {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 16px;
      max-height: calc(100vh - 36px);
      overflow: auto;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 14px;
      overflow-wrap: anywhere;
    }}
    .controls {{
      display: grid;
      gap: 12px;
      margin-bottom: 18px;
    }}
    button {{
      min-height: 40px;
      border: 1px solid #414956;
      background: var(--panel-2);
      color: var(--text);
      font-weight: 700;
      cursor: pointer;
    }}
    input[type="range"] {{ width: 100%; }}
    .row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .snakes {{
      display: grid;
      gap: 6px;
    }}
    .snake {{
      display: grid;
      grid-template-columns: 12px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 9px 0;
      border-top: 1px solid var(--line);
      font-size: 14px;
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--snake-color);
    }}
    .name {{
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .detail {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .move {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
    }}
    @media (max-width: 860px) {{
      main {{ grid-template-columns: 1fr; align-items: start; }}
      aside {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <main>
    <section id="board" class="board" aria-label="Battlesnake replay board"></section>
    <aside>
      <h1 id="title">Battlesnake Replay</h1>
      <div id="meta" class="meta"></div>
      <div class="controls">
        <button id="play">Play</button>
        <input id="turn" type="range" min="0" max="0" value="0">
        <div class="row"><span id="turnLabel">turn 0</span><span id="aliveLabel">alive 0</span></div>
        <label class="row">Speed <input id="speed" type="range" min="60" max="900" step="20" value="180"></label>
      </div>
      <div id="snakes" class="snakes"></div>
    </aside>
  </main>
  <script>
    const replay = {data};
    const board = document.getElementById("board");
    const slider = document.getElementById("turn");
    const play = document.getElementById("play");
    const speed = document.getElementById("speed");
    const title = document.getElementById("title");
    const meta = document.getElementById("meta");
    const turnLabel = document.getElementById("turnLabel");
    const aliveLabel = document.getElementById("aliveLabel");
    const snakes = document.getElementById("snakes");
    let timer = null;

    document.documentElement.style.setProperty("--board-w", replay.width);
    document.documentElement.style.setProperty("--board-h", replay.height);
    slider.max = String(replay.frames.length - 1);
    title.textContent = replay.title || "Battlesnake Replay";
    meta.textContent = `${{replay.width}}x${{replay.height}} · ${{replay.frames.length}} frames · ${{replay.source || ""}}`;

    function key(p) {{ return `${{p.x}},${{p.y}}`; }}

    function render(index) {{
      const frame = replay.frames[index];
      const food = new Set(frame.food.map(key));
      const hazards = new Set((frame.hazards || []).map(key));
      const bodies = new Map();
      const heads = new Map();
      const colors = new Map();
      for (const snake of frame.snakes) {{
        colors.set(snake.id, snake.color);
        snake.body.forEach((part, i) => {{
          const bucket = i === 0 ? heads : bodies;
          bucket.set(key(part), snake.id);
        }});
      }}

      board.innerHTML = "";
      for (let y = replay.height - 1; y >= 0; y--) {{
        for (let x = 0; x < replay.width; x++) {{
          const cell = document.createElement("div");
          const k = `${{x}},${{y}}`;
          cell.className = "cell";
          if (hazards.has(k)) cell.classList.add("hazard");
          if (food.has(k)) cell.classList.add("food");
          const bodyId = bodies.get(k);
          const headId = heads.get(k);
          if (bodyId) {{
            cell.classList.add("body");
            cell.style.setProperty("--snake-color", colors.get(bodyId));
          }}
          if (headId) {{
            cell.classList.add("head");
            cell.style.setProperty("--snake-color", colors.get(headId));
          }}
          board.appendChild(cell);
        }}
      }}

      const lastTurn = replay.frames[replay.frames.length - 1].turn;
      turnLabel.textContent = `turn ${{frame.turn}} / ${{lastTurn}}`;
      aliveLabel.textContent = `alive ${{frame.snakes.filter(s => !s.death).length}}`;
      snakes.innerHTML = "";
      for (const snake of frame.snakes) {{
        const row = document.createElement("div");
        row.className = "snake";
        row.style.setProperty("--snake-color", snake.color);
        const death = snake.death ? ` · out: ${{snake.death.Cause || snake.death.cause || "dead"}}` : "";
        const author = snake.author ? ` · ${{snake.author}}` : "";
        row.innerHTML = `
          <span class="dot"></span>
          <span class="name">
            ${{escapeHtml(snake.name || snake.id)}}
            <div class="detail">len ${{snake.length}} · hp ${{snake.health ?? "-"}} · lat ${{snake.latency || "-"}}${{author}}${{death}}</div>
          </span>
          <span class="move">${{frame.moves[snake.id] || "-"}}</span>
        `;
        snakes.appendChild(row);
      }}
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, ch => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[ch]));
    }}

    function stop() {{
      clearInterval(timer);
      timer = null;
      play.textContent = "Play";
    }}

    function start() {{
      timer = setInterval(() => {{
        let next = Number(slider.value) + 1;
        if (next >= replay.frames.length) {{
          stop();
          return;
        }}
        slider.value = String(next);
        render(next);
      }}, Number(speed.value));
      play.textContent = "Pause";
    }}

    play.addEventListener("click", () => timer ? stop() : start());
    slider.addEventListener("input", () => render(Number(slider.value)));
    speed.addEventListener("input", () => {{
      if (timer) {{
        stop();
        start();
      }}
    }});
    window.addEventListener("keydown", event => {{
      if (event.key === " ") {{
        event.preventDefault();
        timer ? stop() : start();
      }}
      if (event.key === "ArrowRight") {{
        slider.value = String(Math.min(Number(slider.value) + 1, replay.frames.length - 1));
        render(Number(slider.value));
      }}
      if (event.key === "ArrowLeft") {{
        slider.value = String(Math.max(Number(slider.value) - 1, 0));
        render(Number(slider.value));
      }}
    }});
    render(0);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
