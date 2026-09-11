"""Compare the current and submitted agents against the same local Stockfish setup.

Stockfish is test equipment only: this script neither copies it into the repository nor calls it
from the competition agent.  Each build gets both colours from exactly the same ordered opening
positions so the W/D/L delta is more useful than two unrelated gauntlets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import chess
import chess.engine

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).with_name("engine_worker.py").resolve()
DEFAULT_STARTS = REPO_ROOT / "games" / "public-starts.json"
SUPPORTED_ELOS = (2400, 2600, 2800)


@dataclass(frozen=True)
class Start:
    name: str
    fen: str


@dataclass
class Score:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.wins + self.draws / 2

    @property
    def percentage(self) -> float:
        return self.points / self.games if self.games else 0.0

    def record(self, result: str) -> None:
        if result == "win":
            self.wins += 1
        elif result == "draw":
            self.draws += 1
        elif result == "loss":
            self.losses += 1
        else:
            raise ValueError(f"unknown result: {result}")

    def wdl(self) -> str:
        return f"+{self.wins} ={self.draws} -{self.losses}"


class AgentWorker:
    """A persistent, isolated import of one agent root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.process = subprocess.Popen(
            [sys.executable, str(WORKER), str(self.root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def start(self) -> None:
        reply = self._read()
        if reply.get("ready") is not True:
            raise RuntimeError(f"agent at {self.root} failed to start: {reply}")

    def reset(self) -> None:
        self._write({"reset": True})
        if self._read().get("reset") is not True:
            raise RuntimeError(f"agent at {self.root} failed to reset")

    def move(self, fen: str, time_left_ms: int, fixed_nodes: int) -> str:
        self._write(
            {"fen": fen, "time_left_ms": time_left_ms, "fixed_nodes": fixed_nodes}
        )
        move = self._read().get("move")
        if not isinstance(move, str):
            raise RuntimeError(f"invalid agent response from {self.root}: {move!r}")
        return move

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def _write(self, payload: dict[str, object]) -> None:
        stream = _pipe(self.process.stdin)
        stream.write(json.dumps(payload) + "\n")
        stream.flush()

    def _read(self) -> dict[str, object]:
        line = _pipe(self.process.stdout).readline()
        if not line:
            raise RuntimeError(
                f"agent at {self.root} exited with code {self.process.poll()}"
            )
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid agent response from {self.root}: {payload!r}")
        return payload


def _pipe(stream: IO[str] | None) -> IO[str]:
    if stream is None:
        raise RuntimeError("agent worker pipe is unavailable")
    return stream


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _load_starts(path: Path) -> list[Start]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"opening file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"opening file is not valid JSON: {path}: {error}") from error

    if not isinstance(payload, dict) or not isinstance(payload.get("starts"), list):
        raise ValueError(f"opening file must contain a 'starts' list: {path}")

    sortable: list[tuple[int, str, Start]] = []
    for index, item in enumerate(payload["starts"]):
        if not isinstance(item, dict):
            raise ValueError(f"start {index + 1} is not an object")
        fen = item.get("fen")
        if not isinstance(fen, str):
            raise ValueError(f"start {index + 1} has no FEN string")
        try:
            chess.Board(fen)
        except ValueError as error:
            raise ValueError(f"start {index + 1} has an invalid FEN: {fen}") from error

        round_value = item.get("round", 0)
        round_number = round_value if isinstance(round_value, int) else 0
        game_value = item.get("game_id", "")
        game_id = game_value if isinstance(game_value, str) else ""
        suffix = f"-{game_id[:8]}" if game_id else f"-{index + 1}"
        sortable.append(
            (round_number, game_id, Start(name=f"round-{round_number}{suffix}", fen=fen))
        )

    # Newest starts first, with game id as an explicit stable tie breaker.
    starts = [item[2] for item in sorted(sortable, key=lambda item: (-item[0], item[1]))]
    if not starts:
        raise ValueError(f"opening file contains no starts: {path}")
    return starts


def _agent_result(winner: chess.Color | None, agent_color: chess.Color) -> str:
    if winner is None:
        return "draw"
    return "win" if winner == agent_color else "loss"


def _play(
    worker: AgentWorker,
    stockfish: chess.engine.SimpleEngine,
    start: Start,
    agent_color: chess.Color,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
    agent_nodes: int,
    stockfish_nodes: int,
) -> tuple[str, str]:
    worker.reset()
    board = chess.Board(start.fen)
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}
    stockfish_game = object()

    while len(board.move_stack) < ply_cap:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return _agent_result(outcome.winner, agent_color), outcome.termination.name.lower()

        mover = board.turn
        started = time.perf_counter()
        if mover == agent_color:
            uci = worker.move(board.fen(), max(0, int(clocks[mover])), agent_nodes)
            move: chess.Move | None
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                move = None
        else:
            limit = (
                chess.engine.Limit(nodes=stockfish_nodes)
                if stockfish_nodes
                else chess.engine.Limit(
                    white_clock=clocks[chess.WHITE] / 1_000,
                    black_clock=clocks[chess.BLACK] / 1_000,
                    white_inc=increment_ms / 1_000,
                    black_inc=increment_ms / 1_000,
                )
            )
            reply = stockfish.play(board, limit, game=stockfish_game)
            move = reply.move
        fixed_nodes = agent_nodes if mover == agent_color else stockfish_nodes
        if not fixed_nodes:
            clocks[mover] -= (time.perf_counter() - started) * 1_000

        if clocks[mover] < 0:
            return ("loss", "agent-flag") if mover == agent_color else ("win", "stockfish-flag")
        if move is None or move not in board.legal_moves:
            return (
                ("loss", "agent-illegal")
                if mover == agent_color
                else ("win", "stockfish-illegal")
            )
        board.push(move)
        clocks[mover] += increment_ms

    return "draw", "ply-cap"


def _configure_stockfish(engine: chess.engine.SimpleEngine, elo: int) -> None:
    required = {"Threads", "Hash", "UCI_LimitStrength", "UCI_Elo"}
    missing = sorted(required.difference(engine.options))
    if missing:
        raise RuntimeError(f"Stockfish is missing required UCI options: {', '.join(missing)}")
    engine.configure({"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": elo})


def _signed(value: int) -> str:
    return f"{value:+d}"


def _print_summary(label: str, score: Score) -> None:
    print(f"{label}: {score.wdl()}, score {score.percentage:.1%}")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current repo and an exact old agent root against identical paired "
            "Stockfish games. Stockfish remains local test equipment only."
        )
    )
    parser.add_argument("--old", type=Path, required=True, help="root of the exact old agent")
    parser.add_argument(
        "--stockfish",
        type=Path,
        default=os.environ.get("STOCKFISH_PATH"),
        help="local Stockfish binary (or set STOCKFISH_PATH)",
    )
    parser.add_argument("--elo", type=int, choices=SUPPORTED_ELOS, default=2400)
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--pairs", type=_positive_int, help="paired openings per build")
    count.add_argument(
        "--games",
        type=_positive_int,
        help="games per build; must be even because every opening uses both colours",
    )
    parser.add_argument("--base-ms", type=_positive_int, default=5_000)
    parser.add_argument("--increment-ms", type=_nonnegative_int, default=100)
    parser.add_argument("--ply-cap", type=_positive_int, default=600)
    parser.add_argument("--start-index", type=_nonnegative_int, default=0)
    parser.add_argument("--starts", type=Path, default=DEFAULT_STARTS)
    parser.add_argument(
        "--agent-nodes",
        type=_nonnegative_int,
        default=0,
        help="deterministic nodes per agent move; use with --stockfish-nodes",
    )
    parser.add_argument(
        "--stockfish-nodes",
        type=_nonnegative_int,
        default=0,
        help="deterministic nodes per Stockfish move; use with --agent-nodes",
    )
    arguments = parser.parse_args()
    if arguments.games is not None and arguments.games % 2:
        parser.error("--games must be even so each opening is played with both colours")
    if arguments.stockfish is None:
        parser.error("pass --stockfish or set STOCKFISH_PATH (the binary is test-only)")
    if bool(arguments.agent_nodes) != bool(arguments.stockfish_nodes):
        parser.error("set both --agent-nodes and --stockfish-nodes, or neither")
    return arguments


def main() -> None:
    arguments = _parse_arguments()
    old_root = arguments.old.resolve()
    stockfish_path = arguments.stockfish.resolve()
    for label, path in (
        ("current agent", REPO_ROOT / "agent.py"),
        ("old agent", old_root / "agent.py"),
        ("worker", WORKER),
        ("Stockfish binary", stockfish_path),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")

    try:
        starts = _load_starts(arguments.starts.resolve())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    pairs = arguments.pairs
    if pairs is None:
        pairs = arguments.games // 2 if arguments.games is not None else 2
    games_per_build = pairs * 2

    selected = [starts[(arguments.start_index + index) % len(starts)] for index in range(pairs)]
    compute = (
        f"nodes agent={arguments.agent_nodes:,}, Stockfish={arguments.stockfish_nodes:,}"
        if arguments.agent_nodes
        else f"clock {arguments.base_ms / 1_000:g}+{arguments.increment_ms / 1_000:g}"
    )
    print(
        f"Stockfish UCI_Elo {arguments.elo}; {games_per_build} games/build "
        f"({pairs} paired starts); {compute}; ply cap {arguments.ply_cap}",
        flush=True,
    )
    print("Starting both isolated agents; their one-time compilation may take a while.", flush=True)

    workers = {
        "current": AgentWorker(REPO_ROOT),
        "old": AgentWorker(old_root),
    }
    stockfish: chess.engine.SimpleEngine | None = None
    scores = {"current": Score(), "old": Score()}
    played = 0
    total_physical_games = games_per_build * 2
    try:
        # The processes are created before either readiness read, allowing their imports to overlap.
        workers["current"].start()
        workers["old"].start()
        stockfish = chess.engine.SimpleEngine.popen_uci(str(stockfish_path))
        _configure_stockfish(stockfish, arguments.elo)
        print("Both agents and Stockfish are ready.", flush=True)

        for pair_index, start in enumerate(selected):
            for color_index, agent_color in enumerate((chess.WHITE, chess.BLACK)):
                # Alternate which build goes first to avoid systematically favouring run order.
                build_order = (
                    ("current", "old")
                    if (pair_index + color_index) % 2 == 0
                    else ("old", "current")
                )
                for build in build_order:
                    result, termination = _play(
                        workers[build],
                        stockfish,
                        start,
                        agent_color,
                        arguments.base_ms,
                        arguments.increment_ms,
                        arguments.ply_cap,
                        arguments.agent_nodes,
                        arguments.stockfish_nodes,
                    )
                    scores[build].record(result)
                    played += 1
                    color_name = "White" if agent_color == chess.WHITE else "Black"
                    print(
                        f"game {played}/{total_physical_games}: {start.name}, {build} "
                        f"as {color_name}: {result} by {termination}",
                        flush=True,
                    )
    finally:
        if stockfish is not None:
            stockfish.quit()
        for worker in workers.values():
            worker.close()

    current = scores["current"]
    old = scores["old"]
    print()
    _print_summary("current vs Stockfish", current)
    _print_summary("old vs Stockfish", old)
    percentage_delta = (current.percentage - old.percentage) * 100
    print(
        "delta current-old: "
        f"wins {_signed(current.wins - old.wins)}, "
        f"draws {_signed(current.draws - old.draws)}, "
        f"losses {_signed(current.losses - old.losses)}, "
        f"score {percentage_delta:+.1f} percentage points"
    )


if __name__ == "__main__":
    main()
