"""Offline game-domain checks of original residual nets against fresh teacher labels.

The local teacher executable is used only by this development tool. Neither it nor the
labelled positions belong in a submission. Every competition game stays an external group.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import numpy as np
from benchmark_king_nnue import load_network, numpy_prediction


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positions(directory: Path, stride: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.pgn")):
        with path.open(encoding="utf-8-sig") as stream:
            game = chess.pgn.read_game(stream)
        if game is None:
            continue
        board = game.board()
        for ply, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)
            key = board.epd()
            if ply % stride or key in seen or board.is_game_over() or board.is_check():
                continue
            seen.add(key)
            rows.append({"fen": board.fen(), "game": path.name, "ply": ply})
    return rows


def label(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("frozen_baseline_eval", args.baseline)
    assert spec is not None and spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    rows = positions(args.games, args.stride)
    with chess.engine.SimpleEngine.popen_uci(str(args.stockfish)) as teacher:
        teacher.configure({"Threads": 1, "Hash": 64})
        teacher_name = teacher.id.get("name", "unknown")
        for index, row in enumerate(rows):
            board = chess.Board(row["fen"])
            # Clearing hash prevents neighbouring games/positions sharing teacher work.
            teacher.configure({"Clear Hash": None})
            result = teacher.analyse(board, chess.engine.Limit(nodes=args.nodes))
            score = result["score"].white().score()
            if score is None:
                row["skip"] = "teacher mate"
                continue
            encoded, side, _, _, _, king, other = baseline._encode_position(board)
            wk, bk = (king, other) if side == 1 else (other, king)
            value = int(baseline._evaluate(encoded, side, wk, bk))
            row.update({"teacher_cp": score, "baseline_cp": (value - 10) * side,
                        "depth": result.get("depth"), "nodes": result.get("nodes")})
            if index % 50 == 0:
                print(f"labelled {index + 1}/{len(rows)}", flush=True)
    result = {"provenance": {"teacher": teacher_name,
              "teacher_sha256": digest(args.stockfish), "nodes_per_position": args.nodes,
              "baseline_sha256": digest(args.baseline), "script_sha256": digest(Path(__file__)),
              "split": "all competition source games excluded from Lichess cache",
              "elapsed_seconds": time.perf_counter() - started}, "positions": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(rows)} positions in {args.out}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    source = json.loads(args.labels.read_text(encoding="utf-8"))
    rows = [row for row in source["positions"] if "teacher_cp" in row]
    teacher = np.asarray([row["teacher_cp"] for row in rows], dtype=float)
    baseline = np.asarray([row["baseline_cp"] for row in rows], dtype=float)
    groups = np.asarray([row["game"] for row in rows])
    reports: list[dict[str, Any]] = []
    for path in args.weights:
        network = load_network(path)
        predicted = np.asarray([float(numpy_prediction(network, chess.Board(row["fen"])))
                                for row in rows])
        blends: dict[str, Any] = {}
        for blend in (0.0, 0.25, 0.5, 0.75, 1.0):
            errors = np.abs(baseline + predicted * blend - teacher)
            game_error = {str(group): float(np.mean(errors[groups == group]))
                          for group in np.unique(groups)}
            blends[str(blend)] = {"mae_cp": float(errors.mean()),
                "rmse_cp": float(np.sqrt(np.mean(errors ** 2))),
                "p90_cp": float(np.quantile(errors, .9)),
                "game_mean_mae_cp": float(np.mean(list(game_error.values()))),
                "per_game_mae_cp": game_error}
        reports.append({"weights": str(path), "sha256": digest(path), "blends": blends,
                        "prediction_abs_p99_cp": float(np.quantile(np.abs(predicted), .99))})
    output = {"label_source": str(args.labels), "label_sha256": digest(args.labels),
              "positions": len(rows), "games": len(np.unique(groups)), "models": reports}
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    labeller = subparsers.add_parser("label")
    labeller.add_argument("--stockfish", type=Path, required=True)
    labeller.add_argument("--baseline", type=Path, required=True)
    labeller.add_argument("--games", type=Path, default=Path("games"))
    labeller.add_argument("--stride", type=int, default=4)
    labeller.add_argument("--nodes", type=int, default=25_000)
    labeller.add_argument("--out", type=Path, required=True)
    evaluator = subparsers.add_parser("evaluate")
    evaluator.add_argument("--labels", type=Path, required=True)
    evaluator.add_argument("--weights", type=Path, nargs="+", required=True)
    evaluator.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "label":
        label(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
