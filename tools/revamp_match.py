"""Colour-paired regression games with genuine draws, PGNs and immutable build hashes.

This is local test equipment, excluded from submission archives. Persistent workers reset
between games to amortize JIT; use --fresh for independent platform-runner processes.
Fixed-node games compare search choices; wall-clock games also measure inference cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import chess
import chess.pgn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.sandbox import AgentFailure, local  # noqa: E402


class Worker:
    """One isolated, resettable engine with a bounded response wait."""

    def __init__(self, root: Path, log: Path) -> None:
        self.log = log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, str(REPO / "tools/engine_worker.py"), str(root.resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put("")

    def read(self, timeout: float) -> dict[str, Any]:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError("worker timeout") from None
        if not line:
            raise RuntimeError(f"worker exited: {self.process.poll()}")
        payload: dict[str, Any] = json.loads(line)
        return payload

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return self.read(timeout)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.log.close()


def fingerprint(root: Path) -> dict[str, str]:
    paths = sorted(root.glob("*.py")) + sorted((root / "weights").rglob("*"))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file() and "__pycache__" not in path.parts
    }


def play(
    roots: list[Path],
    workers: list[Worker],
    fen: str,
    candidate_white: bool,
    args: argparse.Namespace,
    game_number: int,
) -> dict[str, Any]:
    board = chess.Board(fen)
    game = chess.pgn.Game.from_board(board)
    node: chess.pgn.GameNode = game
    colors = {chess.WHITE: 0 if candidate_white else 1, chess.BLACK: 1 if candidate_white else 0}
    game.headers["White"] = "candidate" if candidate_white else "baseline"
    game.headers["Black"] = "baseline" if candidate_white else "candidate"
    game.headers["TimeControl"] = f"{args.base_ms / 1000:g}+{args.increment_ms / 1000:g}"
    clock = {chess.WHITE: float(args.base_ms), chess.BLACK: float(args.base_ms)}
    elapsed_by_build = [0.0, 0.0]
    moves_by_build = [0, 0]
    fresh = []
    init_seconds = []
    result = "*"
    termination = "unfinished"
    try:
        for index in range(2):
            started = time.perf_counter()
            if args.fresh:
                runner = local(roots[index])
                fresh.append(runner)
                runner.start(90)
            else:
                if not workers[index].request({"reset": True}, 10).get("reset"):
                    raise RuntimeError("worker reset failed")
            init_seconds.append(time.perf_counter() - started)
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                result, termination = outcome.result(), outcome.termination.name.lower()
                break
            if board.ply() >= 600:
                result, termination = "1/2-1/2", "600-ply-draw"
                break
            mover = board.turn
            index = colors[mover]
            started = time.perf_counter()
            try:
                if args.fresh:
                    uci = fresh[index].move(board.fen(), int(clock[mover]))
                else:
                    reply = workers[index].request(
                        {
                            "fen": board.fen(),
                            "time_left_ms": int(clock[mover]),
                            "fixed_nodes": args.nodes,
                        },
                        120 if args.nodes else max(1, clock[mover] / 1000 + 1),
                    )
                    uci = str(reply.get("move"))
                elapsed = (time.perf_counter() - started) * 1000
                elapsed_by_build[index] += elapsed
                moves_by_build[index] += 1
                if not args.nodes:
                    clock[mover] -= elapsed
                    if clock[mover] < 0:
                        raise AgentFailure("flag")
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise AgentFailure("illegal")
            except (AgentFailure, RuntimeError, ValueError, BrokenPipeError) as error:
                termination = error.reason if isinstance(error, AgentFailure) else str(error)
                result = "0-1" if mover else "1-0"
                if termination == "flag" and board.has_insufficient_material(not mover):
                    result = "1/2-1/2"
                break
            board.push(move)
            clock[mover] += args.increment_ms
            node = node.add_variation(move)
            node.comment = f"elapsed_ms={elapsed:.2f} remaining_ms={clock[mover]:.1f}"
    finally:
        for index, runner in enumerate(fresh):
            runner.stop()
            (args.out / f"game-{game_number:03}-engine-{index}.log").write_text(
                runner.stderr_tail, encoding="utf-8"
            )
    if any("Compiled search initialization failed:" in runner.stderr_tail for runner in fresh):
        raise RuntimeError("compiled search failed in a fresh process; refusing fallback result")
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    (args.out / f"game-{game_number:03}.pgn").write_text(str(game) + "\n", encoding="utf-8")
    points = 0.5 if result == "1/2-1/2" else float((result == "1-0") == candidate_white)
    return {
        "game": game_number,
        "fen": fen,
        "candidate_white": candidate_white,
        "result": result,
        "termination": termination,
        "candidate_points": points,
        "plies": len(board.move_stack),
        "elapsed_ms": elapsed_by_build,
        "move_counts": moves_by_build,
        "init_seconds": init_seconds,
        "remaining_ms": {"white": clock[chess.WHITE], "black": clock[chess.BLACK]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=REPO)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--starts", type=Path, default=REPO / "games/public-starts.json")
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--init-seconds",
        type=float,
        default=90,
        help="local exploratory import timeout; competition limit remains 90s",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.fresh and args.nodes:
        parser.error("--fresh uses the platform runner and requires wall-clock mode")
    if args.pairs < 1 or args.base_ms < 1 or args.increment_ms < 0 or args.nodes < 0:
        parser.error("invalid match limits")
    args.out.mkdir(parents=True, exist_ok=True)
    roots = [args.candidate.resolve(), args.baseline.resolve()]
    starts = json.loads(args.starts.read_text(encoding="utf-8"))["starts"]
    report: dict[str, Any] = {
        "candidate": str(roots[0]),
        "baseline": str(roots[1]),
        "fingerprints": [fingerprint(root) for root in roots],
        "nodes": args.nodes,
        "base_ms": args.base_ms,
        "increment_ms": args.increment_ms,
        "fresh_process_per_game": args.fresh,
        "local_init_budget_seconds": args.init_seconds,
        "games": [],
    }
    workers: list[Worker] = []
    try:
        if not args.fresh:
            # Sequential imports make the measured init time meaningful on a shared laptop.
            for index, root in enumerate(roots):
                started = time.perf_counter()
                worker = Worker(root, args.out / f"engine-{index}.log")
                workers.append(worker)
                ready = worker.read(args.init_seconds)
                if not ready.get("ready"):
                    raise RuntimeError("engine did not initialize")
                if ready.get("compiled_search") is not True:
                    raise RuntimeError("compiled search failed; refusing to benchmark the fallback")
                print(
                    f"engine {index} initialized in {time.perf_counter() - started:.2f}s",
                    flush=True,
                )
        for pair in range(args.pairs):
            start = starts[(args.start_index + pair) % len(starts)]
            for candidate_white in (True, False):
                result = play(
                    roots, workers, start["fen"], candidate_white, args, len(report["games"]) + 1
                )
                report["games"].append(result)
                points = [game["candidate_points"] for game in report["games"]]
                wins, losses = points.count(1.0), points.count(0.0)
                draws = points.count(0.5)
                report["summary"] = {
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "score": sum(points) / len(points),
                }
                # Pair-level standard error preserves correlation between swapped colours.
                pairs = [(points[i] + points[i + 1]) / 2 for i in range(0, len(points) - 1, 2)]
                if len(pairs) >= 2:
                    mean = sum(pairs) / len(pairs)
                    sem = math.sqrt(
                        sum((x - mean) ** 2 for x in pairs) / ((len(pairs) - 1) * len(pairs))
                    )
                    report["summary"]["pair_score_standard_error"] = sem
                (args.out / "results.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
                print(
                    f"game {result['game']}/{args.pairs * 2}: {result['result']} "
                    f"({result['termination']}), candidate +{wins} ={draws} -{losses}",
                    flush=True,
                )
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
