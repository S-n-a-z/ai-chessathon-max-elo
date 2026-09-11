"""Play a color-balanced match between this engine and a previous checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

import chess
import chess.pgn

WORKER = Path(__file__).with_name("engine_worker.py")
COMPETITION_PGNS = (
    Path("games/round-24-draw.pgn"),
    Path("games/round-25-loss.pgn"),
    Path("games/round-28-loss.pgn"),
    Path("games/round-29-loss.pgn"),
    Path("games/round-30-loss.pgn"),
    Path("games/round-33-loss.pgn"),
    Path("games/round-34-loss.pgn"),
    Path("games/round-37-loss.pgn"),
    Path("games/round-38-loss.pgn"),
    Path("games/round-41-loss.pgn"),
    Path("games/round-43-loss.pgn"),
    Path("games/round-44-draw.pgn"),
    Path("games/round-45-draw.pgn"),
)
OPENING_LINES = (
    (),
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"),
)
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


class Engine:
    def __init__(self, root: Path) -> None:
        self.process = subprocess.Popen(
            [sys.executable, str(WORKER), str(root.resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def start(self) -> None:
        reply = self._read()
        if reply.get("ready") is not True:
            raise RuntimeError(f"engine failed to start: {reply}")

    def reset(self) -> None:
        self._write({"reset": True})
        if self._read().get("reset") is not True:
            raise RuntimeError("engine failed to reset")

    def move(
        self, fen: str, time_left_ms: int, fixed_depth: int, fixed_nodes: int
    ) -> str:
        self._write(
            {
                "fen": fen,
                "time_left_ms": time_left_ms,
                "fixed_depth": fixed_depth,
                "fixed_nodes": fixed_nodes,
            }
        )
        move = self._read().get("move")
        if not isinstance(move, str):
            raise RuntimeError(f"invalid engine response: {move!r}")
        return move

    def close(self) -> None:
        self.process.kill()
        self.process.wait()

    def _write(self, payload: dict[str, object]) -> None:
        stream = _pipe(self.process.stdin)
        stream.write(json.dumps(payload) + "\n")
        stream.flush()

    def _read(self) -> dict[str, object]:
        line = _pipe(self.process.stdout).readline()
        if not line:
            raise RuntimeError(f"engine exited with code {self.process.poll()}")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid engine response: {payload!r}")
        return payload


def _pipe(stream: IO[str] | None) -> IO[str]:
    if stream is None:
        raise RuntimeError("worker pipe is unavailable")
    return stream


def _starts() -> list[tuple[str, str]]:
    starts: list[tuple[str, str]] = []
    public_starts = Path("games/public-starts.json")
    if public_starts.exists():
        payload = json.loads(public_starts.read_text(encoding="utf-8"))
        for item in payload["starts"]:
            starts.append((f"round-{item['round']}", item["fen"]))
    for path in COMPETITION_PGNS:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            game = chess.pgn.read_game(stream)
        if game is not None:
            fen = game.headers.get("FEN", chess.STARTING_FEN)
            if not any(existing_fen == fen for _, existing_fen in starts):
                starts.append((path.stem, fen))
    for index, line in enumerate(OPENING_LINES):
        board = chess.Board()
        for uci in line:
            board.push_uci(uci)
        starts.append((f"neutral-{index + 1}", board.fen()))
    return starts


def _adjudicate(board: chess.Board) -> chess.Color | None:
    balance = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )
    return chess.WHITE if balance > 0 else chess.BLACK if balance < 0 else None


def _play(
    white: Engine,
    black: Engine,
    start_fen: str,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
    fixed_depth: int,
    fixed_nodes: int,
) -> tuple[chess.Color | None, str]:
    white.reset()
    black.reset()
    board = chess.Board(start_fen)
    players = {chess.WHITE: white, chess.BLACK: black}
    clocks = {chess.WHITE: float(base_ms), chess.BLACK: float(base_ms)}

    while len(board.move_stack) < ply_cap:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return outcome.winner, outcome.termination.name.lower()
        mover = board.turn
        started = time.perf_counter()
        uci = players[mover].move(
            board.fen(), int(clocks[mover]), fixed_depth, fixed_nodes
        )
        if not fixed_depth and not fixed_nodes:
            clocks[mover] -= (time.perf_counter() - started) * 1_000
            if clocks[mover] < 0:
                return not mover, "flag"
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return not mover, "illegal"
        if move not in board.legal_moves:
            return not mover, "illegal"
        board.push(move)
        clocks[mover] += increment_ms
    return _adjudicate(board), "adjudication"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--base-ms", type=int, default=5_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--ply-cap", type=int, default=180)
    parser.add_argument(
        "--fixed-depth",
        type=int,
        default=0,
        help="use deterministic full-width iterative search instead of the clock",
    )
    parser.add_argument(
        "--fixed-nodes",
        type=int,
        default=0,
        help="use a deterministic total node budget per move instead of the clock",
    )
    arguments = parser.parse_args()

    new = Engine(arguments.candidate)
    old = Engine(arguments.old)
    print("Compiling both engines once; this is the slow part of the local match.", flush=True)
    new.start()
    old.start()
    print("Both engines ready.", flush=True)
    wins = draws = losses = 0
    starts = _starts()
    try:
        for pair in range(arguments.pairs):
            start_name, fen = starts[(arguments.start_index + pair) % len(starts)]
            for new_is_white in (True, False):
                white, black = (new, old) if new_is_white else (old, new)
                winner, termination = _play(
                    white,
                    black,
                    fen,
                    arguments.base_ms,
                    arguments.increment_ms,
                    arguments.ply_cap,
                    arguments.fixed_depth,
                    arguments.fixed_nodes,
                )
                if winner is None:
                    draws += 1
                    result = "draw"
                elif winner == new_is_white:
                    wins += 1
                    result = "win"
                else:
                    losses += 1
                    result = "loss"
                number = wins + draws + losses
                print(
                    f"game {number}/{arguments.pairs * 2}: {start_name}, "
                    f"new {'White' if new_is_white else 'Black'} {result} by {termination}",
                    flush=True,
                )
    finally:
        new.close()
        old.close()

    games = wins + draws + losses
    score = (wins + draws / 2) / games
    print(f"new vs submitted: +{wins} ={draws} -{losses}, score {score:.1%}")


if __name__ == "__main__":
    main()
