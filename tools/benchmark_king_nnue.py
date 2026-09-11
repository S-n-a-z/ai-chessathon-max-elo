"""Validate and benchmark a quantized king-conditioned residual network.

This tool is deliberately separate from the playing engine.  It independently reconstructs the
feature layout exported by :mod:`train_king_nnue`, checks the integer runtime path against the
trainer's NumPy predictor, and measures the cost of recomputing both accumulators from a 0x88
board.  It does not decide whether a network is strong enough to ship; that requires game-level
A/B testing.

Example::

    python tools/benchmark_king_nnue.py weights/king_nnue_smoke.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.pgn
import numpy as np
from numba import njit
from numpy.typing import NDArray

KING_BUCKETS = 32
PIECE_PLANES = 12
SQUARES = 64
INPUT_FEATURES = KING_BUCKETS * PIECE_PLANES * SQUARES


@dataclass(frozen=True)
class QuantizedNetwork:
    """Validated arrays needed by a dependency-light runtime."""

    feature_weights: NDArray[np.int8]
    feature_bias: NDArray[np.int32]
    feature_scale: NDArray[np.float32]
    activation_clip: NDArray[np.int32]
    output_weights: NDArray[np.int16]
    output_scale: np.float32
    tempo: np.float32
    target_scale_cp: np.float32
    metadata: dict[str, Any]

    @property
    def hidden(self) -> int:
        return int(self.feature_bias.shape[0])

    @property
    def head_cp(self) -> NDArray[np.float32]:
        """Output coefficients fused into centipawns for the Numba kernel."""

        output = self.output_weights.astype(np.float32) * self.output_scale
        return np.ascontiguousarray(
            output * self.feature_scale * self.target_scale_cp, dtype=np.float32
        )

    @property
    def tempo_cp(self) -> np.float32:
        return np.float32(self.tempo * self.target_scale_cp)


def _scalar(array: NDArray[Any], name: str, dtype: np.dtype[Any]) -> Any:
    if array.shape != () or array.dtype != dtype:
        raise ValueError(f"{name} must be a scalar {dtype}, got {array.shape} {array.dtype}")
    return array.item()


def load_network(path: Path) -> QuantizedNetwork:
    """Load an archive without pickle and reject incompatible layouts early."""

    expected = {
        "feature_weights",
        "feature_bias",
        "feature_scale",
        "activation_clip",
        "output_weights",
        "output_scale",
        "tempo",
        "target_scale_cp",
        "metadata",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = expected.difference(archive.files)
        unexpected = set(archive.files).difference(expected)
        if missing or unexpected:
            raise ValueError(
                f"archive members differ: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        feature_weights = np.ascontiguousarray(archive["feature_weights"])
        feature_bias = np.ascontiguousarray(archive["feature_bias"])
        feature_scale = np.ascontiguousarray(archive["feature_scale"])
        activation_clip = np.ascontiguousarray(archive["activation_clip"])
        output_weights = np.ascontiguousarray(archive["output_weights"])
        output_scale = np.float32(
            _scalar(archive["output_scale"], "output_scale", np.dtype(np.float32))
        )
        tempo = np.float32(_scalar(archive["tempo"], "tempo", np.dtype(np.float32)))
        target_scale_cp = np.float32(
            _scalar(archive["target_scale_cp"], "target_scale_cp", np.dtype(np.float32))
        )
        metadata_array = archive["metadata"]
        if metadata_array.shape != () or metadata_array.dtype.kind != "U":
            raise ValueError("metadata must be a scalar Unicode JSON string")
        metadata = json.loads(str(metadata_array[()]))

    hidden = int(feature_bias.shape[0])
    exact_shapes_and_dtypes = (
        ("feature_weights", feature_weights, (INPUT_FEATURES, hidden), np.dtype(np.int8)),
        ("feature_bias", feature_bias, (hidden,), np.dtype(np.int32)),
        ("feature_scale", feature_scale, (hidden,), np.dtype(np.float32)),
        ("activation_clip", activation_clip, (hidden,), np.dtype(np.int32)),
        ("output_weights", output_weights, (hidden,), np.dtype(np.int16)),
    )
    for name, array, shape, dtype in exact_shapes_and_dtypes:
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"{name} must be {shape} {dtype}, got {array.shape} {array.dtype}")
    if hidden <= 0:
        raise ValueError("network must have at least one hidden unit")
    if not np.all(np.isfinite(feature_scale)) or np.any(feature_scale <= 0.0):
        raise ValueError("feature_scale must contain finite positive values")
    if np.any(activation_clip <= 0):
        raise ValueError("activation_clip must contain positive values")
    if not np.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("output_scale must be finite and positive")
    if not np.isfinite(tempo) or not np.isfinite(target_scale_cp) or target_scale_cp <= 0.0:
        raise ValueError("tempo/target scale are invalid")

    layout = metadata.get("feature_layout")
    if not isinstance(layout, dict):
        raise ValueError("metadata has no feature_layout object")
    required_metadata = {
        "format_version": 1,
        "king_buckets": KING_BUCKETS,
        "piece_planes": PIECE_PLANES,
        "squares": SQUARES,
        "input_features": INPUT_FEATURES,
        "hidden": hidden,
    }
    if metadata.get("format_version") != required_metadata["format_version"]:
        raise ValueError(f"unsupported format_version: {metadata.get('format_version')!r}")
    for key in ("king_buckets", "piece_planes", "squares", "input_features", "hidden"):
        if layout.get(key) != required_metadata[key]:
            raise ValueError(
                f"metadata feature_layout.{key}={layout.get(key)!r}, "
                f"expected {required_metadata[key]!r}"
            )
    if layout.get("horizontal_king_canonicalization") is not True:
        raise ValueError("runtime requires horizontal king canonicalization")
    if layout.get("perspective_weight_sharing") is not True:
        raise ValueError("runtime requires perspective weight sharing")
    metadata_scale = metadata.get("training", {}).get("target_scale_cp")
    if metadata_scale is None or np.float32(metadata_scale) != target_scale_cp:
        raise ValueError("archive target_scale_cp differs from metadata")

    return QuantizedNetwork(
        feature_weights,
        feature_bias,
        feature_scale,
        activation_clip,
        output_weights,
        output_scale,
        tempo,
        target_scale_cp,
        metadata,
    )


def sparse_features(board: chess.Board, perspective: chess.Color) -> tuple[int, ...]:
    """Independent implementation of the exporter's sparse feature contract."""

    king = board.king(perspective)
    if king is None:
        raise ValueError("position is missing a king")
    oriented_king = king if perspective == chess.WHITE else king ^ 56
    flip_files = chess.square_file(oriented_king) >= 4
    if flip_files:
        oriented_king ^= 7
    king_bucket = chess.square_rank(oriented_king) * 4 + chess.square_file(oriented_king)

    active: list[int] = []
    for square, piece in board.piece_map().items():
        relation = 0 if piece.color == perspective else 6
        plane = relation + piece.piece_type - 1
        oriented_square = square if perspective == chess.WHITE else square ^ 56
        if flip_files:
            oriented_square ^= 7
        active.append((king_bucket * PIECE_PLANES + plane) * SQUARES + oriented_square)
    active.sort()
    return tuple(active)


def numpy_accumulators(
    network: QuantizedNetwork, board: chess.Board
) -> tuple[np.ndarray, np.ndarray]:
    """Reference integer accumulators, including bias and clipped ReLU."""

    white_features = sparse_features(board, chess.WHITE)
    black_features = sparse_features(board, chess.BLACK)
    white = network.feature_bias + network.feature_weights[list(white_features)].sum(
        axis=0, dtype=np.int32
    )
    black = network.feature_bias + network.feature_weights[list(black_features)].sum(
        axis=0, dtype=np.int32
    )
    white = np.minimum(np.maximum(white, 0), network.activation_clip).astype(
        np.int32, copy=False
    )
    black = np.minimum(np.maximum(black, 0), network.activation_clip).astype(
        np.int32, copy=False
    )
    return white, black


def numpy_prediction(network: QuantizedNetwork, board: chess.Board) -> np.float32:
    """Mirror ``train_king_nnue._predict_quantized`` operation-for-operation."""

    white, black = numpy_accumulators(network, board)
    difference = (white - black).astype(np.float32) * network.feature_scale
    output = network.output_weights.astype(np.float32) * float(network.output_scale)
    side_to_move = 1 if board.turn == chess.WHITE else -1
    scaled = float(difference @ output) + float(network.tempo) * side_to_move
    return np.float32(scaled * float(network.target_scale_cp))


def encode_0x88(board: chess.Board) -> tuple[np.ndarray, int, int, int]:
    """Encode only the state needed for static residual inference."""

    encoded: NDArray[np.int8] = np.zeros(128, dtype=np.int8)
    white_king = -1
    black_king = -1
    for square, piece in board.piece_map().items():
        square_0x88 = chess.square_rank(square) * 16 + chess.square_file(square)
        value = piece.piece_type if piece.color == chess.WHITE else -piece.piece_type
        encoded[square_0x88] = value
        if piece.piece_type == chess.KING:
            if piece.color == chess.WHITE:
                white_king = square_0x88
            else:
                black_king = square_0x88
    if white_king < 0 or black_king < 0:
        raise ValueError("position is missing a king")
    side = 1 if board.turn == chess.WHITE else -1
    return encoded, white_king, black_king, side


def runtime_sparse_features_0x88(
    board: np.ndarray, king: int, perspective: int
) -> tuple[int, ...]:
    """Independently enumerate the feature IDs that the runtime kernel addresses."""

    king_file = king & 7
    king_rank = king >> 4
    if perspective < 0:
        king_rank = 7 - king_rank
    flip_files = king_file >= 4
    if flip_files:
        king_file = 7 - king_file
    king_bucket = king_rank * 4 + king_file
    active: list[int] = []
    for rank_index in range(8):
        for file_index in range(8):
            square = rank_index * 16 + file_index
            piece = int(board[square])
            if piece == 0:
                continue
            oriented_rank = rank_index if perspective > 0 else 7 - rank_index
            oriented_file = 7 - file_index if flip_files else file_index
            relation = 0 if piece * perspective > 0 else 6
            plane = relation + abs(piece) - 1
            active.append(
                (king_bucket * PIECE_PLANES + plane) * SQUARES
                + oriented_rank * 8
                + oriented_file
            )
    active.sort()
    return tuple(active)


@njit(cache=False, inline="always")  # type: ignore[untyped-decorator]
def _feature_index_0x88(
    square: int, piece: int, perspective: int, king_bucket: int, flip_files: bool
) -> int:
    file_index = square & 7
    rank_index = square >> 4
    if perspective < 0:
        rank_index = 7 - rank_index
    if flip_files:
        file_index = 7 - file_index
    relation = 0 if piece * perspective > 0 else 6
    plane = relation + abs(piece) - 1
    return (king_bucket * PIECE_PLANES + plane) * SQUARES + rank_index * 8 + file_index


@njit(cache=False, inline="always")  # type: ignore[untyped-decorator]
def _king_geometry_0x88(king: int, perspective: int) -> tuple[int, bool]:
    file_index = king & 7
    rank_index = king >> 4
    if perspective < 0:
        rank_index = 7 - rank_index
    flip_files = file_index >= 4
    if flip_files:
        file_index = 7 - file_index
    return rank_index * 4 + file_index, flip_files


@njit(cache=False)  # type: ignore[untyped-decorator]
def runtime_accumulators(
    board: np.ndarray,
    white_king: int,
    black_king: int,
    feature_weights: np.ndarray,
    feature_bias: np.ndarray,
    activation_clip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Full accumulator recomputation shaped for direct use by ``fast_engine``."""

    hidden = feature_bias.shape[0]
    white = np.empty(hidden, dtype=np.int32)
    black = np.empty(hidden, dtype=np.int32)
    for unit in range(hidden):
        white[unit] = feature_bias[unit]
        black[unit] = feature_bias[unit]

    white_bucket, white_flip = _king_geometry_0x88(white_king, 1)
    black_bucket, black_flip = _king_geometry_0x88(black_king, -1)
    for rank_index in range(8):
        for file_index in range(8):
            square = rank_index * 16 + file_index
            piece = int(board[square])
            if piece == 0:
                continue
            white_feature = _feature_index_0x88(
                square, piece, 1, white_bucket, white_flip
            )
            black_feature = _feature_index_0x88(
                square, piece, -1, black_bucket, black_flip
            )
            for unit in range(hidden):
                white[unit] += int(feature_weights[white_feature, unit])
                black[unit] += int(feature_weights[black_feature, unit])

    for unit in range(hidden):
        white_value = white[unit]
        black_value = black[unit]
        if white_value < 0:
            white_value = 0
        elif white_value > activation_clip[unit]:
            white_value = activation_clip[unit]
        if black_value < 0:
            black_value = 0
        elif black_value > activation_clip[unit]:
            black_value = activation_clip[unit]
        white[unit] = white_value
        black[unit] = black_value
    return white, black


@njit(cache=False)  # type: ignore[untyped-decorator]
def runtime_prediction(
    board: np.ndarray,
    white_king: int,
    black_king: int,
    side: int,
    feature_weights: np.ndarray,
    feature_bias: np.ndarray,
    activation_clip: np.ndarray,
    head_cp: np.ndarray,
    tempo_cp: np.float32,
) -> float:
    """Fused full-recompute inference; output is a white-centric residual in cp."""

    white, black = runtime_accumulators(
        board,
        white_king,
        black_king,
        feature_weights,
        feature_bias,
        activation_clip,
    )
    result = float(tempo_cp) * side
    for unit in range(feature_bias.shape[0]):
        result += (int(white[unit]) - int(black[unit])) * float(head_cp[unit])
    return result


@njit(cache=False)  # type: ignore[untyped-decorator]
def _benchmark_network_batch(
    boards: np.ndarray,
    white_kings: np.ndarray,
    black_kings: np.ndarray,
    sides: np.ndarray,
    feature_weights: np.ndarray,
    feature_bias: np.ndarray,
    activation_clip: np.ndarray,
    head_cp: np.ndarray,
    tempo_cp: np.float32,
    iterations: int,
) -> float:
    checksum = 0.0
    for _iteration in range(iterations):
        for index in range(boards.shape[0]):
            checksum += runtime_prediction(
                boards[index],
                int(white_kings[index]),
                int(black_kings[index]),
                int(sides[index]),
                feature_weights,
                feature_bias,
                activation_clip,
                head_cp,
                tempo_cp,
            )
    return checksum


def _position_key(board: chess.Board) -> str:
    return str(board.epd())


def load_positions(
    public_starts: Path, games_directory: Path, pgn_stride: int, maximum: int
) -> list[chess.Board]:
    """Load a deterministic mix of public starts and played-game positions."""

    positions: list[chess.Board] = []
    seen: set[str] = set()

    def append(board: chess.Board) -> None:
        if len(positions) >= maximum or not board.is_valid():
            return
        key = _position_key(board)
        if key not in seen:
            seen.add(key)
            positions.append(board.copy(stack=False))

    if public_starts.is_file():
        payload = json.loads(public_starts.read_text(encoding="utf-8"))
        starts = payload.get("starts", [])
        if not isinstance(starts, list):
            raise ValueError(f"{public_starts} has no starts list")
        for item in starts:
            if isinstance(item, dict) and isinstance(item.get("fen"), str):
                append(chess.Board(item["fen"]))

    if games_directory.is_dir():
        for path in sorted(games_directory.rglob("*.pgn")):
            with path.open(encoding="utf-8-sig", errors="replace") as stream:
                while game := chess.pgn.read_game(stream):
                    board = game.board()
                    append(board)
                    for ply, move in enumerate(game.mainline_moves(), start=1):
                        board.push(move)
                        if ply % pgn_stride == 0:
                            append(board)
                        if len(positions) >= maximum:
                            break
                    if len(positions) >= maximum:
                        break
            if len(positions) >= maximum:
                break

    if not positions:
        positions = [chess.Board()]
    return positions


def _trainer_parity(
    network: QuantizedNetwork, positions: Sequence[chess.Board]
) -> tuple[int, int]:
    """Compare against the actual training module, not just a copied formula."""

    tools_directory = str(Path(__file__).resolve().parent)
    if tools_directory not in sys.path:
        sys.path.insert(0, tools_directory)
    import train_king_nnue

    examples = [
        train_king_nnue.Example(
            white_features=train_king_nnue.sparse_features(board, chess.WHITE),
            black_features=train_king_nnue.sparse_features(board, chess.BLACK),
            side_to_move=1 if board.turn == chess.WHITE else -1,
            target_cp=0.0,
            teacher_cp=0.0,
            baseline_cp=0.0,
            group_id="parity",
        )
        for board in positions
    ]
    quantized = {
        "feature_weights": network.feature_weights,
        "feature_bias": network.feature_bias,
        "feature_scale": network.feature_scale,
        "activation_clip": network.activation_clip,
        "output_weights": network.output_weights,
        "output_scale": np.asarray(network.output_scale, dtype=np.float32),
        "tempo": np.asarray(network.tempo, dtype=np.float32),
    }
    expected = train_king_nnue._predict_quantized(
        quantized, examples, list(range(len(examples))), float(network.target_scale_cp)
    )
    actual = np.asarray([numpy_prediction(network, board) for board in positions])
    feature_mismatches = 0
    for board, example in zip(positions, examples, strict=True):
        if sparse_features(board, chess.WHITE) != example.white_features:
            feature_mismatches += 1
        if sparse_features(board, chess.BLACK) != example.black_features:
            feature_mismatches += 1
    prediction_mismatches = int(
        np.count_nonzero(expected.view(np.uint32) != actual.view(np.uint32))
    )
    return feature_mismatches, prediction_mismatches


def validate_runtime(
    network: QuantizedNetwork, positions: Sequence[chess.Board], check_trainer: bool
) -> dict[str, float | int | str | bool]:
    """Assert exact integer parity and quantify harmless fused-float rounding."""

    maximum_float_error = 0.0
    exact_accumulator_positions = 0
    exact_feature_perspectives = 0
    for board in positions:
        encoded, white_king, black_king, side = encode_0x88(board)
        if runtime_sparse_features_0x88(encoded, white_king, 1) != sparse_features(
            board, chess.WHITE
        ):
            raise AssertionError(f"white feature-ID mismatch at {board.fen()}")
        if runtime_sparse_features_0x88(encoded, black_king, -1) != sparse_features(
            board, chess.BLACK
        ):
            raise AssertionError(f"black feature-ID mismatch at {board.fen()}")
        exact_feature_perspectives += 2
        expected_white, expected_black = numpy_accumulators(network, board)
        actual_white, actual_black = runtime_accumulators(
            encoded,
            white_king,
            black_king,
            network.feature_weights,
            network.feature_bias,
            network.activation_clip,
        )
        if not np.array_equal(expected_white, actual_white):
            raise AssertionError(f"white accumulator mismatch at {board.fen()}")
        if not np.array_equal(expected_black, actual_black):
            raise AssertionError(f"black accumulator mismatch at {board.fen()}")
        exact_accumulator_positions += 1
        expected = float(numpy_prediction(network, board))
        actual = runtime_prediction(
            encoded,
            white_king,
            black_king,
            side,
            network.feature_weights,
            network.feature_bias,
            network.activation_clip,
            network.head_cp,
            network.tempo_cp,
        )
        maximum_float_error = max(maximum_float_error, abs(expected - actual))

    feature_mismatches = 0
    predictor_bit_mismatches = 0
    trainer_parity_status = "skipped"
    if check_trainer:
        try:
            feature_mismatches, predictor_bit_mismatches = _trainer_parity(network, positions)
        except ModuleNotFoundError as error:
            if error.name != "torch":
                raise
            trainer_parity_status = "unavailable: torch is not installed in this interpreter"
        else:
            trainer_parity_status = "checked"
            if feature_mismatches or predictor_bit_mismatches:
                raise AssertionError(
                    f"trainer parity failed: {feature_mismatches} feature and "
                    f"{predictor_bit_mismatches} predictor mismatches"
                )
    return {
        "positions": len(positions),
        "runtime_feature_id_exact_perspectives": exact_feature_perspectives,
        "integer_accumulator_exact": exact_accumulator_positions,
        "trainer_parity_status": trainer_parity_status,
        "trainer_feature_mismatches": feature_mismatches,
        "trainer_prediction_bit_mismatches": predictor_bit_mismatches,
        "fused_runtime_max_abs_error_cp": maximum_float_error,
    }


def benchmark_runtime(
    network: QuantizedNetwork,
    positions: Sequence[chess.Board],
    iterations: int,
    rounds: int,
) -> dict[str, float | int]:
    """Measure JIT-internal inference calls, excluding Python dispatcher overhead."""

    encoded = [encode_0x88(board) for board in positions]
    boards = np.ascontiguousarray(np.stack([item[0] for item in encoded]))
    white_kings: NDArray[np.int16] = np.asarray(
        [item[1] for item in encoded], dtype=np.int16
    )
    black_kings: NDArray[np.int16] = np.asarray(
        [item[2] for item in encoded], dtype=np.int16
    )
    sides: NDArray[np.int8] = np.asarray([item[3] for item in encoded], dtype=np.int8)
    arguments = (
        boards,
        white_kings,
        black_kings,
        sides,
        network.feature_weights,
        network.feature_bias,
        network.activation_clip,
        network.head_cp,
        network.tempo_cp,
    )
    warmup_started = time.perf_counter()
    _benchmark_network_batch(*arguments, 1)
    warmup_seconds = time.perf_counter() - warmup_started
    timings: list[float] = []
    checksum = 0.0
    for _round in range(rounds):
        started = time.perf_counter()
        checksum = _benchmark_network_batch(*arguments, iterations)
        timings.append(time.perf_counter() - started)
    elapsed = statistics.median(timings)
    evaluations = iterations * len(positions)
    return {
        "hidden": network.hidden,
        "evaluations": evaluations,
        "jit_warmup_seconds": warmup_seconds,
        "median_seconds": elapsed,
        "nanoseconds_per_evaluation": elapsed * 1e9 / evaluations,
        "evaluations_per_second": evaluations / elapsed,
        "checksum": checksum,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_sidecar(path: Path, metadata: dict[str, Any]) -> bool | None:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    return bool(json.loads(sidecar.read_text(encoding="utf-8")) == metadata)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--public-starts", type=Path, default=Path("games/public-starts.json"))
    parser.add_argument("--games", type=Path, default=Path("games"))
    parser.add_argument("--max-positions", type=int, default=256)
    parser.add_argument("--pgn-stride", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--skip-trainer-parity",
        action="store_true",
        help="skip importing the Torch-based trainer for an independent bit-parity check",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in ("max_positions", "pgn_stride", "iterations", "rounds"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    load_started = time.perf_counter()
    network = load_network(args.weights)
    load_seconds = time.perf_counter() - load_started
    positions = load_positions(
        args.public_starts, args.games, args.pgn_stride, args.max_positions
    )
    parity = validate_runtime(network, positions, not args.skip_trainer_parity)
    benchmark = benchmark_runtime(network, positions, args.iterations, args.rounds)
    report = {
        "archive": str(args.weights),
        "archive_sha256": _sha256(args.weights),
        "archive_bytes": args.weights.stat().st_size,
        "tensor_bytes_uncompressed": sum(
            array.nbytes
            for array in (
                network.feature_weights,
                network.feature_bias,
                network.feature_scale,
                network.activation_clip,
                network.output_weights,
            )
        ),
        "archive_load_seconds": load_seconds,
        "sidecar_metadata_exact": _check_sidecar(args.weights, network.metadata),
        "parity": parity,
        "benchmark": benchmark,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
