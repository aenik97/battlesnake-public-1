#!/usr/bin/env python3
"""Fetch public Battlesnake leaderboard game recordings.

The public game page embeds board.battlesnake.com, which loads:

  GET  https://engine.battlesnake.com/games/<game_id>
  WSS  wss://engine.battlesnake.com/games/<game_id>/events

This script uses only Python stdlib so it can run next to the baseline bot
without adding runtime dependencies.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import socket
import ssl
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


PLAY_BASE = "https://play.battlesnake.com"
ENGINE_BASE = "https://engine.battlesnake.com"
GAME_ID_RE = re.compile(
    r"/game/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
LEADERBOARD_STATS_RE = re.compile(r"/leaderboard/([^/]+)/([^/]+)/stats")
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def fetch_text(url: str, timeout: float = 20.0) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "risk-ai-challenge/0.1",
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    return json.loads(fetch_text(url, timeout=timeout))


def discover_game_ids_from_stats(
    leaderboard: str,
    player: str,
    limit: int,
) -> list[str]:
    stats_url = f"{PLAY_BASE}/leaderboard/{leaderboard}/{player}/stats"
    html = fetch_text(stats_url)
    seen: set[str] = set()
    game_ids: list[str] = []
    for match in GAME_ID_RE.finditer(html):
        game_id = match.group(1).lower()
        if game_id not in seen:
            seen.add(game_id)
            game_ids.append(game_id)
        if len(game_ids) >= limit:
            break
    return game_ids


def discover_top_players(leaderboard: str, limit: int) -> list[str]:
    leaderboard_url = f"{PLAY_BASE}/leaderboard/{leaderboard}"
    html = fetch_text(leaderboard_url)
    players: list[str] = []
    seen: set[str] = set()
    for board, player in LEADERBOARD_STATS_RE.findall(html):
        player = urllib.parse.unquote(player)
        if board == leaderboard and player not in seen:
            seen.add(player)
            players.append(player)
        if len(players) >= limit:
            break
    return players


def _read_until(sock: ssl.SSLSocket, marker: bytes, limit: int = 65536) -> bytes:
    data = b""
    while marker not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if len(data) > limit:
            raise RuntimeError("websocket handshake response is too large")
    return data


def _recv_exact(sock: ssl.SSLSocket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("websocket closed while reading frame")
        data += chunk
    return data


class StdlibWebSocket:
    def __init__(self, url: str, timeout: float = 30.0) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "wss":
            raise ValueError(f"only wss:// URLs are supported: {url}")
        self.host = parsed.hostname or ""
        self.port = parsed.port or 443
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += "?" + parsed.query
        self.timeout = timeout
        self.sock: ssl.SSLSocket | None = None

    def __enter__(self) -> "StdlibWebSocket":
        raw = socket.create_connection((self.host, self.port), timeout=self.timeout)
        context = ssl.create_default_context()
        self.sock = context.wrap_socket(raw, server_hostname=self.host)
        self.sock.settimeout(self.timeout)

        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Origin: https://board.battlesnake.com\r\n"
            "\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = _read_until(self.sock, b"\r\n\r\n")
        header_text = response.decode("iso-8859-1", errors="replace")
        if " 101 " not in header_text.splitlines()[0]:
            raise RuntimeError(f"websocket handshake failed: {header_text[:200]}")

        expected = base64.b64encode(
            hashlib.sha1((key + WS_GUID).encode("ascii")).digest()
        ).decode("ascii")
        if expected.lower() not in header_text.lower():
            raise RuntimeError("websocket handshake accept key mismatch")
        return self

    def __exit__(self, *exc: object) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def recv_text(self) -> str | None:
        if self.sock is None:
            raise RuntimeError("websocket is not connected")

        while True:
            first, second = _recv_exact(self.sock, 2)
            opcode = first & 0x0F
            masked = bool(second & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack("!H", _recv_exact(self.sock, 2))[0]
            elif length == 127:
                length = struct.unpack("!Q", _recv_exact(self.sock, 8))[0]

            mask = _recv_exact(self.sock, 4) if masked else b""
            payload = _recv_exact(self.sock, length) if length else b""
            if masked:
                payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))

            if opcode == 0x1:
                return payload.decode("utf-8")
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._send_pong(payload)
                continue
            if opcode in (0xA, 0x0):
                continue
            raise RuntimeError(f"unsupported websocket opcode: {opcode}")

    def _send_pong(self, payload: bytes) -> None:
        if self.sock is None:
            return
        header = bytes([0x8A, 0x80 | len(payload)])
        mask = os.urandom(4)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked)


def fetch_game_events(
    game_id: str,
    engine_base: str = ENGINE_BASE,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    ws_base = engine_base.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_base}/games/{game_id}/events"
    events: list[dict[str, Any]] = []
    with StdlibWebSocket(ws_url, timeout=timeout) as ws:
        while True:
            message = ws.recv_text()
            if message is None:
                break
            event = json.loads(message)
            events.append(event)
            if event.get("Type") == "game_end":
                break
    return events


def fetch_game_recording(
    game_id: str,
    engine_base: str = ENGINE_BASE,
    timeout: float = 30.0,
) -> dict[str, Any]:
    metadata = fetch_json(f"{engine_base}/games/{game_id}", timeout=timeout)
    events = fetch_game_events(game_id, engine_base=engine_base, timeout=timeout)
    frames = [event["Data"] for event in events if event.get("Type") == "frame"]
    return {
        "game_id": game_id,
        "game_url": f"{PLAY_BASE}/game/{game_id}",
        "engine_base": engine_base,
        "fetched_at_unix": int(time.time()),
        "metadata": metadata,
        "events": events,
        "frames": frames,
    }


def recording_path(out_dir: Path, game_id: str) -> Path:
    return out_dir / f"{game_id}.json"


def write_recording(recording: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = recording_path(out_dir, recording["game_id"])
    path.write_text(json.dumps(recording, indent=2, sort_keys=True), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch public Battlesnake game recordings from leaderboards."
    )
    parser.add_argument("--leaderboard", default="standard")
    parser.add_argument("--player", help="Leaderboard player slug, e.g. andreammm")
    parser.add_argument(
        "--top-players",
        type=int,
        default=0,
        help="Discover this many top players from the leaderboard.",
    )
    parser.add_argument("--game-id", action="append", default=[])
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of recent games to collect per player.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/games"))
    parser.add_argument("--engine", default=ENGINE_BASE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download games even when the output JSON already exists.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/games_manifest.json"),
        help="Write a small manifest with discovered players and game sources.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    game_ids = [game_id.lower() for game_id in args.game_id]
    players: list[str] = []
    game_sources: dict[str, list[str]] = {}

    if args.player:
        players.append(args.player)
    if args.top_players:
        discovered_players = discover_top_players(args.leaderboard, args.top_players)
        players.extend(discovered_players)

    players = list(dict.fromkeys(players))
    for player in players:
        player_game_ids = discover_game_ids_from_stats(
            args.leaderboard,
            player,
            args.limit,
        )
        print(
            f"discovered {len(player_game_ids)} games for {player} "
            f"on {args.leaderboard}",
            flush=True,
        )
        for game_id in player_game_ids:
            game_ids.append(game_id)
            game_sources.setdefault(game_id, []).append(player)
    for game_id in args.game_id:
        game_sources.setdefault(game_id.lower(), []).append("direct")

    deduped = list(dict.fromkeys(game_ids))
    if not deduped:
        raise SystemExit("Provide --game-id, --player, or --top-players.")

    manifest: dict[str, Any] = {
        "leaderboard": args.leaderboard,
        "players": players,
        "requested_games_per_player": args.limit,
        "discovered_game_count": len(deduped),
        "downloaded": [],
        "skipped_existing": [],
        "failed": [],
        "sources": game_sources,
    }

    for game_id in deduped:
        path = recording_path(args.out, game_id)
        if path.exists() and not args.force:
            print(f"skipping existing {path}", flush=True)
            manifest["skipped_existing"].append(str(path))
            continue

        print(f"fetching {game_id} ...", flush=True)
        try:
            recording = fetch_game_recording(
                game_id,
                engine_base=args.engine,
                timeout=args.timeout,
            )
        except (OSError, EOFError, urllib.error.URLError, RuntimeError) as exc:
            print(f"failed {game_id}: {exc}", flush=True)
            manifest["failed"].append({"game_id": game_id, "error": str(exc)})
            continue
        recording["leaderboard_sources"] = game_sources.get(game_id, [])
        path = write_recording(recording, args.out)
        print(f"saved {path} ({len(recording['frames'])} frames)", flush=True)
        manifest["downloaded"].append(
            {
                "game_id": game_id,
                "path": str(path),
                "frames": len(recording["frames"]),
                "sources": game_sources.get(game_id, []),
            }
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote manifest {args.manifest}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
