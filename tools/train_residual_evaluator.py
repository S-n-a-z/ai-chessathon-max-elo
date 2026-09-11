"""Train a compact linear correction for the original handcrafted evaluator.

The labels are generated offline by Stockfish. Only the fitted, color-symmetric piece-square
weights are shipped; neither the labelling engine nor its position data is used at runtime.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

import fast_engine

BASE_FEATURES = 6 * 64
FEATURES = BASE_FEATURES * 2 + 1
PHASE = (0, 0, 1, 1, 2, 4, 0)


def _starts(directory: Path) -> list[str]:
    starts = [chess.STARTING_FEN]
    for path in sorted(directory.glob("*.pgn")):
        with path.open(encoding="utf-8") as stream:
            game = chess.pgn.read_game(stream)
        if game is not None and game.headers.get("FEN"):
            starts.append(game.headers["FEN"])
    return list(dict.fromkeys(starts))


def _phase(board: chess.Board) -> int:
    return min(
        24,
        sum(PHASE[piece.piece_type] for piece in board.piece_map().values()),
    )


def encode(board: chess.Board) -> np.ndarray:
    features = np.zeros(FEATURES, dtype=np.float32)
    phase = _phase(board) / 24.0
    ending = 1.0 - phase
    for square, piece in board.piece_map().items():
        relative = square if piece.color == chess.WHITE else chess.square_mirror(square)
        index = (piece.piece_type - 1) * 64 + relative
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        features[index] += sign * phase
        features[BASE_FEATURES + index] += sign * ending
    features[-1] = 1.0 if board.turn == chess.WHITE else -1.0
    return features


def _current_white_score(board: chess.Board) -> float:
    encoded, side, _castling, _ep, _halfmove, king, other = fast_engine._encode_position(board)
    white_king = king if side == fast_engine.WHITE else other
    black_king = other if side == fast_engine.WHITE else king
    side_score = int(fast_engine._evaluate(encoded, side, white_king, black_king))
    return float((side_score - 10) * side)


def collect(
    engine: chess.engine.SimpleEngine,
    starts: list[str],
    samples: int,
    nodes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    features: list[np.ndarray] = []
    targets: list[float] = []
    game_index = 0
    while len(features) < samples:
        board = chess.Board(starts[game_index % len(starts)])
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < 120 and len(features) < samples:
            analysis = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=4)
            infos = analysis if isinstance(analysis, list) else [analysis]
            principal = infos[0]
            score = principal["score"].pov(chess.WHITE).score(mate_score=2_500)
            if score is not None:
                residual = float(np.clip(score - _current_white_score(board), -700, 700))
                features.append(encode(board))
                targets.append(residual)
                if len(features) % 1_000 == 0:
                    print(f"labelled {len(features)}/{samples}", flush=True)

            choices = [info["pv"][0] for info in infos if info.get("pv")]
            if not choices:
                break
            roll = rng.random()
            choice = 0 if roll < 0.56 else 1 if roll < 0.80 else 2 if roll < 0.94 else 3
            board.push(choices[min(choice, len(choices) - 1)])
            plies += 1
        game_index += 1
    return np.stack(features), np.asarray(targets, dtype=np.float64)


def fit_huber_ridge(
    features: np.ndarray, targets: np.ndarray, ridge: float
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(20_260_906)
    order = rng.permutation(len(features))
    validation_count = max(1_000, len(features) // 10)
    validation = order[:validation_count]
    training = order[validation_count:]
    x_train = features[training].astype(np.float64)
    y_train = targets[training]
    x_val = features[validation].astype(np.float64)
    y_val = targets[validation]

    weights = np.zeros(FEATURES, dtype=np.float64)
    sample_weights = np.ones(len(x_train), dtype=np.float64)
    regularizer = np.eye(FEATURES, dtype=np.float64) * ridge
    regularizer[-1, -1] = ridge * 0.1
    for iteration in range(4):
        weighted = x_train * sample_weights[:, None]
        weights = np.linalg.solve(
            x_train.T @ weighted + regularizer, x_train.T @ (sample_weights * y_train)
        )
        errors = x_train @ weights - y_train
        sample_weights = np.minimum(1.0, 140.0 / np.maximum(1.0, np.abs(errors)))
        print(
            f"fit {iteration + 1}/4: train MAE {np.mean(np.abs(errors)):.1f} cp",
            flush=True,
        )

    before = float(np.mean(np.abs(y_val)))
    predictions = x_val @ weights
    after = float(np.mean(np.abs(y_val - predictions)))
    return weights, {
        "validation_residual_mae_before_cp": before,
        "validation_residual_mae_after_cp": after,
    }


def _table(coefficients: np.ndarray) -> np.ndarray:
    table = np.zeros((7, 128), dtype=np.int16)
    for piece_type in range(1, 7):
        for square in chess.SQUARES:
            source = (piece_type - 1) * 64 + square
            target = chess.square_rank(square) * 16 + chess.square_file(square)
            table[piece_type, target] = int(np.clip(round(coefficients[source]), -1_500, 1_500))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--games", type=Path, default=Path("games"))
    parser.add_argument("--samples", type=int, default=30_000)
    parser.add_argument("--nodes", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=20_260_906)
    parser.add_argument("--ridge", type=float, default=30.0)
    parser.add_argument("--out", type=Path, default=Path("weights/eval_residual.npz"))
    args = parser.parse_args()

    started = time.perf_counter()
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    engine_name = engine.id.get("name", "unknown")
    try:
        engine.configure({"Threads": 1, "Hash": 128})
        features, targets = collect(
            engine,
            _starts(args.games),
            args.samples,
            args.nodes,
            args.seed,
        )
    finally:
        engine.quit()
    weights, metrics = fit_huber_ridge(features, targets, args.ridge)
    metadata = {
        "architecture": "color-symmetric tapered linear residual",
        "engine_labeler": engine_name,
        "positions": args.samples,
        "nodes_per_label": args.nodes,
        "seed": args.seed,
        "ridge": args.ridge,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        **metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        mg=_table(weights[:BASE_FEATURES]),
        eg=_table(weights[BASE_FEATURES : 2 * BASE_FEATURES]),
        tempo=np.int16(round(weights[-1])),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
