"""Benchmark one engine checkout at a deterministic fixed search depth."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chess


def _load_engine(root: Path) -> ModuleType:
    sys.path.insert(0, str(root.resolve()))
    return importlib.import_module("fast_engine")


def _starts(path: Path, maximum: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    starts = payload.get("starts")
    if not isinstance(starts, list):
        raise ValueError(f"{path} has no starts list")
    return [item for item in starts[:maximum] if isinstance(item, dict)]


def _search(engine: ModuleType, fen: str, depth: int) -> dict[str, Any]:
    board = chess.Board(fen)
    engine.reset_for_test()
    encoded = engine._encode_position(board)
    started = time.perf_counter()
    score, move, nodes, completed = engine._root_search(
        *encoded,
        depth,
        -engine.INF,
        engine.INF,
        0,
        200_000_000,
        1,
        engine._tt_keys,
        engine._tt_scores,
        engine._tt_moves,
        engine._tt_depths,
        engine._tt_bounds,
        engine._tt_ages,
        engine._move_buffer,
        engine._score_buffer,
        engine._killers,
        engine._history,
        engine._hash_stack,
        engine._nodes,
    )
    elapsed = time.perf_counter() - started
    return {
        "move": engine._decode_move(move).uci(),
        "score": score,
        "nodes": nodes,
        "completed": completed,
        "seconds": elapsed,
        "nps": nodes / elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--starts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "games" / "public-starts.json",
    )
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--positions", type=int, default=8)
    args = parser.parse_args()
    if args.depth <= 0 or args.positions <= 0:
        raise ValueError("depth and positions must be positive")

    import_started = time.perf_counter()
    engine = _load_engine(args.root)
    import_seconds = time.perf_counter() - import_started
    results = []
    for item in _starts(args.starts, args.positions):
        result = _search(engine, str(item["fen"]), args.depth)
        result["round"] = item.get("round")
        results.append(result)
    total_nodes = sum(int(result["nodes"]) for result in results)
    total_seconds = sum(float(result["seconds"]) for result in results)
    print(
        json.dumps(
            {
                "root": str(args.root.resolve()),
                "depth": args.depth,
                "positions": len(results),
                "import_seconds": import_seconds,
                "nodes": total_nodes,
                "search_seconds": total_seconds,
                "nps": total_nodes / total_seconds,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
