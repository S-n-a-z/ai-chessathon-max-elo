"""Check deployed integer feature math against an independent NumPy reference."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import chess
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.candidate.resolve()))
    started = time.perf_counter()
    engine = importlib.import_module("fast_engine")
    neural = importlib.import_module("neural")
    import_seconds = time.perf_counter() - started
    from benchmark_king_nnue import load_network, numpy_prediction

    network = load_network(args.candidate / "weights/king_nnue.npz")
    randomizer = random.Random(26831)
    position = chess.Board()
    maximum_error = 0.0
    examples = []
    for _ in range(500):
        encoded, side, _, _, _, king, other = engine._encode_position(position)
        white_king, black_king = (king, other) if side == 1 else (other, king)
        actual = neural.correction(encoded, side, white_king, black_king)
        expected = float(numpy_prediction(network, position))
        maximum_error = max(maximum_error, abs(actual - expected))
        if abs(actual - expected) > 0.02:
            raise AssertionError(f"neural mismatch {actual} vs {expected}: {position.fen()}")
        classical = engine._evaluate_classical(encoded, side, white_king, black_king)
        combined = engine._evaluate(encoded, side, white_king, black_king)
        phase = sum(
            weight * len(position.pieces(piece, color))
            for piece, weight in (
                (chess.KNIGHT, 1),
                (chess.BISHOP, 1),
                (chess.ROOK, 2),
                (chess.QUEEN, 4),
            )
            for color in chess.COLORS
        )
        blend = engine.NN_BLEND if min(phase, 24) <= getattr(engine, "NN_MAX_PHASE", 24) else 0
        assert combined == classical + round(actual * side * blend)
        examples.append((encoded, side, white_king, black_king))
        moves = list(position.legal_moves)
        if moves and not position.is_game_over() and len(position.move_stack) < 200:
            position.push(randomizer.choice(moves))
        else:
            position.reset()
    # Search mutates and restores the same mailbox. Exercise that exact access pattern with
    # warm caches, including revisiting a king square and changing its mirrored orientation.
    position = chess.Board()
    encoded, _, _, _, _, _, _ = engine._encode_position(position)
    cached_error = 0.0
    checked_transitions = 0
    for _ in range(750):
        fresh, side, _, _, _, king, other = engine._encode_position(position)
        encoded[:128] = fresh[:128]
        white, black = (king, other) if side == 1 else (other, king)
        actual = neural.correction(encoded, side, white, black)
        expected = float(numpy_prediction(network, position))
        cached_error = max(cached_error, abs(actual - expected))
        if abs(actual - expected) > 0.02:
            raise AssertionError(f"cached mismatch after transition: {position.fen()}")
        checked_transitions += 1
        moves = list(position.legal_moves)
        if position.move_stack and (randomizer.random() < 0.3 or not moves):
            position.pop()
        elif moves:
            position.push(randomizer.choice(moves))
        else:
            position.reset()
    timings = {}
    for name in ("_evaluate_classical", "_evaluate"):
        evaluate = getattr(engine, name)
        measurements = []
        checksum = 0
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(10):
                for encoded, side, white, black in examples:
                    checksum += evaluate(encoded, side, white, black)
            measurements.append((time.perf_counter() - started) / 5000)
        timings[name] = {
            "median_us": float(np.median(measurements)) * 1e6,
            "checksum": int(checksum),
        }
    report = {
        "positions": len(examples),
        "max_error_cp": maximum_error,
        "import_seconds": import_seconds,
        "timings": timings,
        "cached_transition_checks": checked_transitions,
        "cached_max_error_cp": cached_error,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
