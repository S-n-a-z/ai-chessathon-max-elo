"""Train our own compact evaluator from positions labelled offline by Stockfish.

The competition permits engine-labelled training data but forbids shipping the labelling engine,
published networks, or a runtime lookup database. This script therefore emits only weights learned
from the generated examples. Neither Stockfish nor the examples enter ``submission.zip``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

INPUTS = 12 * 64 + 1


def encode(board: chess.Board) -> np.ndarray:
    features = np.zeros(INPUTS, dtype=np.float32)
    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        index = (color_offset + piece.piece_type - 1) * 64 + square
        features[index] = 1.0
    features[-1] = 1.0 if board.turn == chess.WHITE else -1.0
    return features


def color_swap_permutation() -> np.ndarray:
    permutation = np.zeros(12 * 64, dtype=np.int32)
    for color_index in range(2):
        for piece_index in range(6):
            for square in chess.SQUARES:
                old = (color_index * 6 + piece_index) * 64 + square
                new = ((1 - color_index) * 6 + piece_index) * 64 + chess.square_mirror(square)
                permutation[old] = new
    return permutation


def collect(
    engine: chess.engine.SimpleEngine, samples: int, nodes: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    examples: list[np.ndarray] = []
    targets: list[float] = []
    game_index = 0
    while len(examples) < samples:
        board = chess.Board()
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < 180 and len(examples) < samples:
            analysis = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=3)
            if not isinstance(analysis, list):
                analysis = [analysis]
            principal = analysis[0]
            score = principal["score"].pov(chess.WHITE).score(mate_score=2_500)
            if score is not None and ply >= 6:
                examples.append(encode(board))
                targets.append(float(np.clip(score, -2_000, 2_000)) / 1_000.0)
                if len(examples) % 500 == 0:
                    print(f"labelled {len(examples)}/{samples}", flush=True)

            candidates = [item["pv"][0] for item in analysis if item.get("pv")]
            if not candidates:
                break
            roll = rng.random()
            if ply < 24 and len(candidates) >= 3:
                choice = 0 if roll < 0.58 else 1 if roll < 0.86 else 2
            elif len(candidates) >= 2 and roll > 0.88:
                choice = 1
            else:
                choice = 0
            board.push(candidates[choice])
            ply += 1
        game_index += 1
        print(f"completed source game {game_index}", flush=True)
    return np.stack(examples), np.asarray(targets, dtype=np.float32)


def augment(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    permutation = color_swap_permutation()
    swapped = np.zeros_like(features)
    swapped[:, permutation] = features[:, : 12 * 64]
    swapped[:, -1] = -features[:, -1]
    return np.concatenate((features, swapped)), np.concatenate((targets, -targets))


def train(
    features: np.ndarray,
    targets: np.ndarray,
    hidden: int,
    epochs: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    validation_count = max(256, len(features) // 10)
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    x_val, y_val = features[validation_indices], targets[validation_indices]
    x_train, y_train = features[training_indices], targets[training_indices]

    w1 = rng.normal(0.0, 0.035, size=(INPUTS, hidden)).astype(np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = rng.normal(0.0, 0.035, size=hidden).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)
    parameters = (w1, b1, w2, b2)
    first_moments = tuple(np.zeros_like(parameter) for parameter in parameters)
    second_moments = tuple(np.zeros_like(parameter) for parameter in parameters)
    step = 0
    batch_size = 256
    learning_rate = 0.0025

    for epoch in range(epochs):
        epoch_order = rng.permutation(len(x_train))
        for start in range(0, len(epoch_order), batch_size):
            step += 1
            indices = epoch_order[start : start + batch_size]
            x_batch = x_train[indices]
            y_batch = y_train[indices]
            hidden_pre = x_batch @ w1 + b1
            hidden_values = np.maximum(hidden_pre, 0.0)
            predictions = hidden_values @ w2 + b2[0]
            errors = predictions - y_batch
            clipped = np.clip(errors, -0.35, 0.35) / len(indices)

            grad_w2 = hidden_values.T @ clipped + 1e-5 * w2
            grad_b2 = np.array([clipped.sum()], dtype=np.float32)
            hidden_gradient = clipped[:, None] * w2[None, :]
            hidden_gradient[hidden_pre <= 0.0] = 0.0
            grad_w1 = x_batch.T @ hidden_gradient + 1e-5 * w1
            grad_b1 = hidden_gradient.sum(axis=0)
            gradients = (grad_w1, grad_b1, grad_w2, grad_b2)

            for parameter, gradient, first, second in zip(
                parameters, gradients, first_moments, second_moments, strict=True
            ):
                first *= 0.9
                first += 0.1 * gradient
                second *= 0.999
                second += 0.001 * gradient * gradient
                first_corrected = first / (1.0 - 0.9**step)
                second_corrected = second / (1.0 - 0.999**step)
                parameter -= learning_rate * first_corrected / (np.sqrt(second_corrected) + 1e-8)

        validation_predictions = np.maximum(x_val @ w1 + b1, 0.0) @ w2 + b2[0]
        mae = float(np.mean(np.abs(validation_predictions - y_val)) * 1_000)
        sign_accuracy = float(np.mean(np.sign(validation_predictions) == np.sign(y_val)))
        print(
            f"epoch {epoch + 1:02d}/{epochs}: validation MAE {mae:.1f} cp, "
            f"sign {sign_accuracy:.1%}",
            flush=True,
        )

    metrics = {"validation_mae_cp": mae, "validation_sign_accuracy": sign_accuracy}
    weights = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return weights, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small, original Chessathon evaluator.")
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--samples", type=int, default=6_000)
    parser.add_argument("--nodes", type=int, default=2_000)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20_260_904)
    parser.add_argument("--out", type=Path, default=Path("weights/evaluator.npz"))
    args = parser.parse_args()

    started = time.perf_counter()
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine_name = engine.id.get("name", "unknown")
    try:
        engine.configure({"Threads": 1, "Hash": 64})
        features, targets = collect(engine, args.samples, args.nodes, args.seed)
    finally:
        engine.quit()
    features, targets = augment(features, targets)
    weights, metrics = train(features, targets, args.hidden, args.epochs, args.seed)

    metadata = {
        "architecture": f"{INPUTS}-{args.hidden}-1 ReLU",
        "engine_labeler": engine_name,
        "positions_before_symmetry": args.samples,
        "nodes_per_label": args.nodes,
        "seed": args.seed,
        "epochs": args.epochs,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        **metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **weights, metadata=json.dumps(metadata, sort_keys=True))
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
