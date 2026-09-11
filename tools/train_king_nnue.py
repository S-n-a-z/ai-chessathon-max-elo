"""Train an original compact, king-conditioned residual evaluator.

This is an offline training tool.  Stockfish supplies labels, but neither Stockfish nor the
labelled positions are exported.  The output contains only weights learned by this project.

The network is deliberately small and sparse.  For each colour it sums embeddings for every
piece, conditioned on that colour's king square, applies clipped ReLU, and subtracts the two
accumulators.  The two perspectives share all weights, making colour-swap antisymmetry exact.
It predicts a correction to ``fast_engine``'s handcrafted static evaluation rather than trying
to relearn material from scratch.

Validation is held out by complete source game.  Positions descended from a validation start or
PGN never appear in training, which prevents the overly optimistic position-level split used by
many quick evaluator experiments.

Example pilot (small enough to validate the pipeline, not enough to produce a strong net)::

    python tools/train_king_nnue.py --stockfish C:/path/to/stockfish.exe \
        --samples 128 --nodes 100 --max-groups 8 --hidden 8 --epochs 2 \
        --out weights/king_nnue_pilot.npz

For a serious experiment, use millions of diverse positions from thousands of independent
games, stronger teacher labels, and a sealed opening-family test set.  Gate the quantized runtime
integration in a paired game-level SPRT before submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

KING_BUCKETS = 32
PIECE_PLANES = 12
SQUARES = 64
INPUT_FEATURES = KING_BUCKETS * PIECE_PLANES * SQUARES
ACTIVATION_CLIP = 1.0
MATE_SCORE_CP = 32_000


@dataclass
class PositionGroup:
    """One leakage boundary: a competition game and all positions derived from it."""

    group_id: str
    seeds: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    result: str = "unknown"


@dataclass(frozen=True)
class Example:
    """A sparse position and its white-centric residual label."""

    white_features: tuple[int, ...]
    black_features: tuple[int, ...]
    side_to_move: int
    target_cp: float
    teacher_cp: float
    baseline_cp: float
    group_id: str


@dataclass(frozen=True)
class AnalysisRecord:
    teacher_cp: float
    moves: tuple[str, ...]


class KingResidual(nn.Module):
    """Shared sparse accumulators with an exactly antisymmetric output."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.feature_weights = nn.EmbeddingBag(INPUT_FEATURES, hidden, mode="sum")
        self.feature_bias = nn.Parameter(torch.zeros(hidden))
        self.output = nn.Linear(hidden, 1, bias=False)
        self.tempo = nn.Parameter(torch.zeros(()))
        # Rows absent from the training corpus must stay neutral.  Random unseen
        # embeddings can otherwise inject arbitrary scores when a new king
        # bucket/piece-square combination first appears in a rated game.
        nn.init.zeros_(self.feature_weights.weight)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.04)

    def forward(
        self,
        white_indices: Tensor,
        white_offsets: Tensor,
        black_indices: Tensor,
        black_offsets: Tensor,
        side_to_move: Tensor,
    ) -> Tensor:
        white = self.feature_weights(white_indices, white_offsets) + self.feature_bias
        black = self.feature_weights(black_indices, black_offsets) + self.feature_bias
        white = torch.clamp(white, 0.0, ACTIVATION_CLIP)
        black = torch.clamp(black, 0.0, ACTIVATION_CLIP)
        positional = self.output(white - black).squeeze(1)
        return cast(Tensor, positional + self.tempo * side_to_move)


def _canonical_position(board: chess.Board) -> str:
    """Return a clock-independent position key suitable for de-duplication."""

    return str(board.epd())


def _append_seed(group: PositionGroup, board: chess.Board) -> None:
    if board.is_valid():
        fen = board.fen()
        key = _canonical_position(board)
        if all(_canonical_position(chess.Board(existing)) != key for existing in group.seeds):
            group.seeds.append(fen)


def _round_key(value: object) -> str:
    text = str(value).strip()
    try:
        return str(int(text))
    except ValueError:
        return text


def load_position_groups(
    public_starts: Path,
    games_directory: Path,
    opening_corpus: Path | None,
    anchor_stride: int,
    max_source_plies: int,
) -> tuple[list[PositionGroup], str]:
    """Load public starts and merge matching PGN mainlines into the same game group."""

    if not public_starts.is_file():
        raise FileNotFoundError(f"public starts file not found: {public_starts}")
    raw = public_starts.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    starts = payload.get("starts")
    if not isinstance(starts, list):
        raise ValueError(f"{public_starts} must contain a JSON 'starts' list")

    groups_by_id: dict[str, PositionGroup] = {}
    match_by_round_and_start: dict[tuple[str, str], str] = {}
    digest = hashlib.sha256()
    digest.update(public_starts.as_posix().encode())
    digest.update(raw)

    for index, item in enumerate(starts):
        if not isinstance(item, dict) or not isinstance(item.get("fen"), str):
            raise ValueError(f"invalid public start at index {index}")
        board = chess.Board(item["fen"])
        game_id = str(item.get("game_id") or f"public-{index}")
        group_id = f"game:{game_id}"
        group = groups_by_id.setdefault(group_id, PositionGroup(group_id))
        _append_seed(group, board)
        group.sources.append(f"{public_starts.as_posix()}#{index}")
        group.result = str(item.get("result", group.result))
        match_by_round_and_start[
            (_round_key(item.get("round", "")), _canonical_position(board))
        ] = group_id

    if opening_corpus is not None:
        if not opening_corpus.is_dir():
            raise FileNotFoundError(f"opening corpus directory not found: {opening_corpus}")
        for path in sorted(opening_corpus.glob("?.tsv")):
            data = path.read_bytes()
            digest.update(path.as_posix().encode())
            digest.update(data)
            reader = csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t")
            for row_index, row in enumerate(reader):
                eco = str(row.get("eco", "")).strip().upper()
                pgn = row.get("pgn")
                if not eco or not pgn:
                    continue
                game = chess.pgn.read_game(io.StringIO(pgn))
                if game is None:
                    continue
                group_id = f"opening:{eco}"
                group = groups_by_id.setdefault(group_id, PositionGroup(group_id))
                group.sources.append(f"{path.as_posix()}#{row_index}")
                board = game.board()
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    if ply > max_source_plies:
                        break
                    board.push(move)
                    if ply >= 4 and ply % anchor_stride == 0:
                        _append_seed(group, board)
                _append_seed(group, board)

    for path in sorted(games_directory.rglob("*.pgn")):
        data = path.read_bytes()
        digest.update(path.as_posix().encode())
        digest.update(data)
        with path.open(encoding="utf-8-sig", errors="replace") as stream:
            game_index = 0
            while game := chess.pgn.read_game(stream):
                root = game.board()
                round_number = _round_key(game.headers.get("Round", ""))
                matched = match_by_round_and_start.get(
                    (round_number, _canonical_position(root))
                )
                if matched is None:
                    relative = path.relative_to(games_directory).as_posix()
                    matched = f"pgn:{relative}:{game_index}"
                group = groups_by_id.setdefault(matched, PositionGroup(matched))
                source = f"{path.as_posix()}#{game_index}"
                if source not in group.sources:
                    group.sources.append(source)
                if group.result == "unknown":
                    group.result = game.headers.get("Result", "unknown")
                _append_seed(group, root)

                board = root.copy(stack=False)
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    if ply > max_source_plies:
                        break
                    board.push(move)
                    if ply % anchor_stride == 0:
                        _append_seed(group, board)
                game_index += 1

    groups = [group for group in groups_by_id.values() if group.seeds]
    if len(groups) < 2:
        raise ValueError("at least two non-empty source game groups are required")
    return sorted(groups, key=lambda group: group.group_id), digest.hexdigest()


def _orient_square(square: int, perspective: chess.Color, flip_files: bool) -> int:
    oriented = square if perspective == chess.WHITE else chess.square_mirror(square)
    return oriented ^ 7 if flip_files else oriented


def sparse_features(board: chess.Board, perspective: chess.Color) -> tuple[int, ...]:
    """Encode one perspective with its king bucket and optional horizontal canonicalisation."""

    king = board.king(perspective)
    if king is None:
        raise ValueError("position has no king for one perspective")
    oriented_king = king if perspective == chess.WHITE else chess.square_mirror(king)
    flip_files = chess.square_file(oriented_king) >= 4
    if flip_files:
        oriented_king ^= 7
    king_bucket = chess.square_rank(oriented_king) * 4 + chess.square_file(oriented_king)

    active: list[int] = []
    for square, piece in board.piece_map().items():
        relation = 0 if piece.color == perspective else 6
        plane = relation + piece.piece_type - 1
        oriented_square = _orient_square(square, perspective, flip_files)
        active.append((king_bucket * PIECE_PLANES + plane) * SQUARES + oriented_square)
    active.sort()
    return tuple(active)


_FAST_ENGINE: ModuleType | None = None


def _current_white_score(board: chess.Board) -> float:
    """Evaluate with the current handcrafted evaluator, excluding its side-to-move tempo."""

    global _FAST_ENGINE
    if _FAST_ENGINE is None:
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        import fast_engine

        _FAST_ENGINE = fast_engine
    encoded, side, _castling, _ep, _halfmove, king, other = _FAST_ENGINE._encode_position(
        board
    )
    white_king = king if side == _FAST_ENGINE.WHITE else other
    black_king = other if side == _FAST_ENGINE.WHITE else king
    side_score = int(_FAST_ENGINE._evaluate(encoded, side, white_king, black_king))
    return float((side_score - 10) * side)


def _analyse(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    multipv: int,
) -> AnalysisRecord | None:
    result = engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=multipv)
    infos = result if isinstance(result, list) else [result]
    if not infos:
        return None
    score = infos[0]["score"].pov(chess.WHITE).score(mate_score=MATE_SCORE_CP)
    if score is None:
        return None
    moves = tuple(info["pv"][0].uci() for info in infos if info.get("pv"))
    return AnalysisRecord(float(score), moves)


def _choose_move(
    board: chess.Board,
    moves: Sequence[str],
    rng: random.Random,
    random_move_rate: float,
) -> chess.Move | None:
    legal = list(board.legal_moves)
    if not legal:
        return None
    if rng.random() < random_move_rate:
        return rng.choice(legal)
    candidates = [chess.Move.from_uci(uci) for uci in moves]
    candidates = [move for move in candidates if move in board.legal_moves]
    if not candidates:
        return rng.choice(legal)
    rank_weights = (0.56, 0.25, 0.12, 0.07)
    weights = [rank_weights[min(index, len(rank_weights) - 1)] for index in range(len(candidates))]
    return rng.choices(candidates, weights=weights, k=1)[0]


def _balanced_quotas(groups: Sequence[PositionGroup], samples: int) -> dict[str, int]:
    base, remainder = divmod(samples, len(groups))
    return {
        group.group_id: base + (1 if index < remainder else 0)
        for index, group in enumerate(groups)
    }


def collect_examples(
    engine: chess.engine.SimpleEngine,
    groups: Sequence[PositionGroup],
    samples: int,
    nodes: int,
    multipv: int,
    line_plies: int,
    random_move_rate: float,
    residual_clip_cp: float,
    max_teacher_abs_cp: float,
    seed: int,
) -> list[Example]:
    """Generate balanced stochastic teacher lines while preserving source group IDs."""

    rng = random.Random(seed)
    quotas = _balanced_quotas(groups, samples)
    examples: list[Example] = []
    seen: set[str] = set()
    cache: dict[str, AnalysisRecord | None] = {}

    for group_index, group in enumerate(groups, start=1):
        target = quotas[group.group_id]
        collected = 0
        episode = 0
        maximum_episodes = max(16, target * 8)
        while collected < target and episode < maximum_episodes:
            seed_fen = group.seeds[(episode + rng.randrange(len(group.seeds))) % len(group.seeds)]
            board = chess.Board(seed_fen)
            episode += 1
            for _ply in range(line_plies):
                if collected >= target or board.is_game_over(claim_draw=True):
                    break
                key = _canonical_position(board)
                record = cache.get(key)
                if key not in cache:
                    record = _analyse(engine, board, nodes, multipv)
                    cache[key] = record
                if record is None:
                    break

                if key not in seen and abs(record.teacher_cp) <= max_teacher_abs_cp:
                    baseline = _current_white_score(board)
                    residual = float(
                        np.clip(record.teacher_cp - baseline, -residual_clip_cp, residual_clip_cp)
                    )
                    examples.append(
                        Example(
                            white_features=sparse_features(board, chess.WHITE),
                            black_features=sparse_features(board, chess.BLACK),
                            side_to_move=1 if board.turn == chess.WHITE else -1,
                            target_cp=residual,
                            teacher_cp=record.teacher_cp,
                            baseline_cp=baseline,
                            group_id=group.group_id,
                        )
                    )
                    seen.add(key)
                    collected += 1
                    if len(examples) % 100 == 0 or len(examples) == samples:
                        print(f"labelled {len(examples)}/{samples}", flush=True)

                move = _choose_move(board, record.moves, rng, random_move_rate)
                if move is None:
                    break
                board.push(move)

        if collected != target:
            raise RuntimeError(
                f"could only collect {collected}/{target} unique positions for {group.group_id}; "
                "increase --line-plies or reduce --samples"
            )
        print(
            f"group {group_index}/{len(groups)}: {group.group_id} ({collected} positions)",
            flush=True,
        )
    return examples


def select_and_split_groups(
    groups: Sequence[PositionGroup],
    samples: int,
    max_groups: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[PositionGroup], set[str]]:
    """Select pilot groups and reserve whole games for validation."""

    rng = random.Random(seed)
    selected = list(groups)
    rng.shuffle(selected)
    requested_groups = max_groups if max_groups > 0 else len(selected)
    active_count = min(len(selected), requested_groups, samples)
    if active_count < 2:
        raise ValueError("need at least two samples/source groups for group-held-out validation")
    selected = selected[:active_count]
    validation_count = max(1, round(active_count * validation_fraction))
    validation_count = min(active_count - 1, validation_count)
    # Keep complete games as leakage boundaries, but represent each available result class.
    # With this small public corpus, a purely random split can otherwise contain only wins or
    # only losses and move the apparent validation error by hundreds of centipawns.
    by_result: dict[str, list[PositionGroup]] = {}
    for group in selected:
        result = group.result.strip().lower()
        category = result if result in {"win", "draw", "loss"} else "other"
        by_result.setdefault(category, []).append(group)
    for bucket in by_result.values():
        rng.shuffle(bucket)

    validation: list[PositionGroup] = []
    for category in ("win", "loss", "draw"):
        bucket = by_result.get(category, [])
        if bucket and len(validation) < validation_count:
            validation.append(bucket.pop())
    remainder = [
        group
        for category in ("win", "loss", "draw")
        for group in by_result.get(category, [])
    ]
    rng.shuffle(remainder)
    validation.extend(remainder[: validation_count - len(validation)])
    if len(validation) < validation_count:
        validation.extend(by_result.get("other", [])[: validation_count - len(validation)])
    validation_groups = {group.group_id for group in validation}
    return selected, validation_groups


def _batch_tensors(
    examples: Sequence[Example],
    indices: Sequence[int],
    target_scale_cp: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    white_lengths: NDArray[np.int64] = np.fromiter(
        (len(examples[index].white_features) for index in indices), dtype=np.int64
    )
    black_lengths: NDArray[np.int64] = np.fromiter(
        (len(examples[index].black_features) for index in indices), dtype=np.int64
    )
    white_offsets: NDArray[np.int64] = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(white_lengths[:-1]))
    )
    black_offsets: NDArray[np.int64] = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(black_lengths[:-1]))
    )
    white_indices: NDArray[np.int64] = np.fromiter(
        (feature for index in indices for feature in examples[index].white_features),
        dtype=np.int64,
    )
    black_indices: NDArray[np.int64] = np.fromiter(
        (feature for index in indices for feature in examples[index].black_features),
        dtype=np.int64,
    )
    side: NDArray[np.float32] = np.fromiter(
        (examples[index].side_to_move for index in indices), dtype=np.float32
    )
    targets: NDArray[np.float32] = np.fromiter(
        (examples[index].target_cp / target_scale_cp for index in indices),
        dtype=np.float32,
    )
    return (
        torch.from_numpy(white_indices),
        torch.from_numpy(white_offsets),
        torch.from_numpy(black_indices),
        torch.from_numpy(black_offsets),
        torch.from_numpy(side),
        torch.from_numpy(targets),
    )


def _predict_float(
    model: KingResidual,
    examples: Sequence[Example],
    indices: Sequence[int],
    batch_size: int,
    target_scale_cp: float,
) -> NDArray[np.float32]:
    predictions: list[NDArray[np.float32]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            white, white_offsets, black, black_offsets, side, _targets = _batch_tensors(
                examples, batch_indices, target_scale_cp
            )
            values = model(white, white_offsets, black, black_offsets, side)
            predictions.append(values.cpu().numpy() * target_scale_cp)
    return np.concatenate(predictions).astype(np.float32, copy=False)


def _metrics(
    examples: Sequence[Example],
    indices: Sequence[int],
    predictions_cp: NDArray[np.float32],
) -> dict[str, float]:
    objective_targets: NDArray[np.float64] = np.asarray(
        [examples[index].target_cp for index in indices], dtype=np.float64
    )
    teacher_residuals: NDArray[np.float64] = np.asarray(
        [examples[index].teacher_cp - examples[index].baseline_cp for index in indices],
        dtype=np.float64,
    )
    objective_errors = predictions_cp.astype(np.float64) - objective_targets
    teacher_errors = predictions_cp.astype(np.float64) - teacher_residuals
    absolute = np.abs(teacher_errors)
    baseline_objective_mae = float(np.mean(np.abs(objective_targets)))
    objective_mae = float(np.mean(np.abs(objective_errors)))
    return {
        "baseline_objective_mae_cp": baseline_objective_mae,
        "objective_mae_cp": objective_mae,
        "objective_mae_gain_cp": baseline_objective_mae - objective_mae,
        "baseline_teacher_mae_cp": float(np.mean(np.abs(teacher_residuals))),
        "corrected_teacher_mae_cp": float(np.mean(absolute)),
        "corrected_teacher_rmse_cp": float(np.sqrt(np.mean(teacher_errors * teacher_errors))),
        "corrected_teacher_p90_error_cp": float(np.percentile(absolute, 90)),
    }


def _feature_coverage(
    examples: Sequence[Example],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> dict[str, float | int]:
    training_rows = {
        feature
        for index in train_indices
        for feature in examples[index].white_features + examples[index].black_features
    }
    validation_rows = {
        feature
        for index in validation_indices
        for feature in examples[index].white_features + examples[index].black_features
    }
    unseen_validation = validation_rows - training_rows
    return {
        "training_feature_rows": len(training_rows),
        "training_feature_fraction": len(training_rows) / INPUT_FEATURES,
        "validation_feature_rows": len(validation_rows),
        "validation_rows_unseen_in_training": len(unseen_validation),
        "validation_unseen_fraction": (
            len(unseen_validation) / len(validation_rows) if validation_rows else 0.0
        ),
    }


def train_model(
    examples: Sequence[Example],
    validation_groups: set[str],
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    target_scale_cp: float,
    seed: int,
) -> tuple[KingResidual, dict[str, Any], list[int], list[int]]:
    train_indices = [
        index for index, example in enumerate(examples) if example.group_id not in validation_groups
    ]
    validation_indices = [
        index for index, example in enumerate(examples) if example.group_id in validation_groups
    ]
    if not train_indices or not validation_indices:
        raise ValueError("group split produced an empty training or validation set")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = KingResidual(hidden)
    training_features = sorted(
        {
            feature
            for index in train_indices
            for feature in (
                examples[index].white_features + examples[index].black_features
            )
        }
    )
    with torch.no_grad():
        feature_indices = torch.tensor(training_features, dtype=torch.long)
        initial_values = torch.empty((len(training_features), hidden))
        nn.init.normal_(initial_values, mean=0.0, std=0.008)
        model.feature_weights.weight.index_copy_(0, feature_indices, initial_values)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05
    )
    loss_function = nn.SmoothL1Loss(beta=140.0 / target_scale_cp)
    best_mae = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_indices).tolist()
        running_loss = 0.0
        trained = 0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            white, white_offsets, black, black_offsets, side, targets = _batch_tensors(
                examples, batch_indices, target_scale_cp
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(white, white_offsets, black, black_offsets, side)
            loss = loss_function(predictions, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += float(loss.detach()) * len(batch_indices)
            trained += len(batch_indices)
        scheduler.step()

        validation_predictions = _predict_float(
            model, examples, validation_indices, batch_size, target_scale_cp
        )
        validation = _metrics(examples, validation_indices, validation_predictions)
        validation_mae = validation["objective_mae_cp"]
        history.append(
            {
                "epoch": float(epoch),
                "train_huber": running_loss / max(1, trained),
                "validation_mae_cp": validation_mae,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        print(
            f"epoch {epoch:02d}/{epochs}: train Huber {running_loss / trained:.5f}, "
            f"held-out objective MAE {validation_mae:.1f} cp",
            flush=True,
        )

        if validation_mae < best_mae - 0.05:
            best_mae = validation_mae
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    float_predictions = _predict_float(
        model, examples, validation_indices, batch_size, target_scale_cp
    )
    training_predictions = _predict_float(
        model, examples, train_indices, batch_size, target_scale_cp
    )
    training_metrics = _metrics(examples, train_indices, training_predictions)
    validation_metrics = _metrics(examples, validation_indices, float_predictions)
    report: dict[str, Any] = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "training": training_metrics,
        "validation_float": validation_metrics,
        "feature_coverage": _feature_coverage(
            examples, train_indices, validation_indices
        ),
    }
    return model, report, train_indices, validation_indices


def quantize_model(model: KingResidual) -> dict[str, np.ndarray]:
    """Quantize sparse inputs to int8 and the output head to int16."""

    features = model.feature_weights.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.feature_bias.detach().cpu().numpy().astype(np.float32)
    output = model.output.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
    tempo = float(model.tempo.detach().cpu())

    feature_max = np.max(np.abs(features), axis=0)
    feature_scale = np.maximum(feature_max / 127.0, np.float32(1e-8)).astype(np.float32)
    feature_quantized = np.clip(np.rint(features / feature_scale), -127, 127).astype(np.int8)
    bias_quantized = np.rint(bias / feature_scale).astype(np.int32)
    activation_clip_quantized = np.maximum(
        1, np.rint(ACTIVATION_CLIP / feature_scale)
    ).astype(np.int32)

    output_scale = np.float32(max(float(np.max(np.abs(output))) / 32_767.0, 1e-10))
    output_quantized = np.clip(np.rint(output / output_scale), -32_767, 32_767).astype(
        np.int16
    )
    return {
        "feature_weights": feature_quantized,
        "feature_bias": bias_quantized,
        "feature_scale": feature_scale,
        "activation_clip": activation_clip_quantized,
        "output_weights": output_quantized,
        "output_scale": np.asarray(output_scale, dtype=np.float32),
        "tempo": np.asarray(tempo, dtype=np.float32),
    }


def _predict_quantized(
    quantized: dict[str, np.ndarray],
    examples: Sequence[Example],
    indices: Sequence[int],
    target_scale_cp: float,
) -> NDArray[np.float32]:
    feature_weights = quantized["feature_weights"]
    feature_bias = quantized["feature_bias"]
    feature_scale = quantized["feature_scale"]
    activation_clip = quantized["activation_clip"]
    output = quantized["output_weights"].astype(np.float32) * float(
        quantized["output_scale"]
    )
    tempo = float(quantized["tempo"])
    predictions: NDArray[np.float32] = np.empty(len(indices), dtype=np.float32)
    for output_index, example_index in enumerate(indices):
        example = examples[example_index]
        white = feature_bias + feature_weights[list(example.white_features)].sum(
            axis=0, dtype=np.int32
        )
        black = feature_bias + feature_weights[list(example.black_features)].sum(
            axis=0, dtype=np.int32
        )
        white = np.minimum(np.maximum(white, 0), activation_clip)
        black = np.minimum(np.maximum(black, 0), activation_clip)
        difference = (white - black).astype(np.float32) * feature_scale
        scaled = float(difference @ output) + tempo * example.side_to_move
        predictions[output_index] = scaled * target_scale_cp
    return predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.stockfish.is_file():
        raise FileNotFoundError(f"Stockfish binary not found: {args.stockfish}")
    integer_positive = (
        "samples",
        "nodes",
        "multipv",
        "line_plies",
        "anchor_stride",
        "max_source_plies",
        "hidden",
        "epochs",
        "batch_size",
        "patience",
        "threads",
    )
    for name in integer_positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if not 0.0 <= args.random_move_rate <= 1.0:
        raise ValueError("--random-move-rate must be between 0 and 1")
    if (
        args.target_scale_cp <= 0.0
        or args.residual_clip_cp <= 0.0
        or args.max_teacher_abs_cp <= 0.0
    ):
        raise ValueError("target, residual, and teacher cutoff scales must be positive")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--public-starts", type=Path, default=Path("games/public-starts.json"))
    parser.add_argument("--games", type=Path, default=Path("games"))
    parser.add_argument(
        "--opening-corpus",
        type=Path,
        help="optional lichess-org/chess-openings TSV directory, grouped by ECO code",
    )
    parser.add_argument("--samples", type=int, default=80_000)
    parser.add_argument("--nodes", type=int, default=8_000)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--line-plies", type=int, default=32)
    parser.add_argument("--anchor-stride", type=int, default=3)
    parser.add_argument("--max-source-plies", type=int, default=160)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="limit source games for a quick pilot; 0 uses every group",
    )
    parser.add_argument("--random-move-rate", type=float, default=0.025)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-scale-cp", type=float, default=400.0)
    parser.add_argument("--residual-clip-cp", type=float, default=1_200.0)
    parser.add_argument(
        "--max-teacher-abs-cp",
        type=float,
        default=2_000.0,
        help="skip won or mated positions whose score would dominate residual training",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_906)
    parser.add_argument("--out", type=Path, default=Path("weights/king_nnue.npz"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)

    groups, source_fingerprint = load_position_groups(
        args.public_starts,
        args.games,
        args.opening_corpus,
        args.anchor_stride,
        args.max_source_plies,
    )
    selected, validation_groups = select_and_split_groups(
        groups,
        args.samples,
        args.max_groups,
        args.validation_fraction,
        args.seed,
    )
    print(
        f"loaded {len(groups)} source groups; using {len(selected)} "
        f"({len(validation_groups)} held out)",
        flush=True,
    )

    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    engine_name = engine.id.get("name", "unknown")
    try:
        options: dict[str, int] = {"Threads": args.threads, "Hash": 128}
        engine.configure(options)
        examples = collect_examples(
            engine,
            selected,
            args.samples,
            args.nodes,
            args.multipv,
            args.line_plies,
            args.random_move_rate,
            args.residual_clip_cp,
            args.max_teacher_abs_cp,
            args.seed,
        )
    finally:
        engine.quit()

    model, training_report, train_indices, validation_indices = train_model(
        examples,
        validation_groups,
        args.hidden,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.patience,
        args.target_scale_cp,
        args.seed,
    )
    quantized = quantize_model(model)
    quantized_predictions = _predict_quantized(
        quantized, examples, validation_indices, args.target_scale_cp
    )
    quantized_metrics = _metrics(examples, validation_indices, quantized_predictions)
    training_report["validation_quantized"] = quantized_metrics
    project_root = Path(__file__).resolve().parents[1]

    group_counts: dict[str, int] = {}
    for example in examples:
        group_counts[example.group_id] = group_counts.get(example.group_id, 0) + 1
    metadata: dict[str, Any] = {
        "format_version": 1,
        "architecture": "shared symmetric king-bucket sparse residual",
        "feature_layout": {
            "king_buckets": KING_BUCKETS,
            "piece_planes": PIECE_PLANES,
            "squares": SQUARES,
            "input_features": INPUT_FEATURES,
            "horizontal_king_canonicalization": True,
            "perspective_weight_sharing": True,
            "activation": f"clipped ReLU [0,{ACTIVATION_CLIP}]",
            "hidden": args.hidden,
            "output_bias": False,
        },
        "purpose": "white-centric correction to the current handcrafted static evaluator",
        "provenance": {
            "trainer_sha256": _sha256(Path(__file__).resolve()),
            "baseline_evaluator": "fast_engine.py",
            "baseline_evaluator_sha256": _sha256(project_root / "fast_engine.py"),
        },
        "teacher": {
            "name": engine_name,
            "binary_sha256": _sha256(args.stockfish),
            "nodes_per_position": args.nodes,
            "multipv": args.multipv,
            "mate_score_cp": MATE_SCORE_CP,
        },
        "data": {
            "source_fingerprint_sha256": source_fingerprint,
            "source_groups_available": len(groups),
            "source_groups_used": len(selected),
            "positions": len(examples),
            "train_positions": len(train_indices),
            "validation_positions": len(validation_indices),
            "validation_split": "whole source games; no position-level split",
            "validation_groups": sorted(validation_groups),
            "positions_per_group": group_counts,
            "line_plies": args.line_plies,
            "random_move_rate": args.random_move_rate,
            "residual_clip_cp": args.residual_clip_cp,
            "max_teacher_abs_cp": args.max_teacher_abs_cp,
            "labelled_positions_exported": False,
        },
        "training": {
            "seed": args.seed,
            "target_scale_cp": args.target_scale_cp,
            "hidden": args.hidden,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "loss": "SmoothL1/Huber, 140 cp transition",
            **training_report,
        },
        "quantization": {
            "feature_weights": "symmetric int8, per hidden channel",
            "feature_bias": "int32 in feature accumulator units",
            "output_weights": "symmetric int16, one scale",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    archive = {
        **quantized,
        "target_scale_cp": np.asarray(args.target_scale_cp, dtype=np.float32),
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    with args.out.open("wb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], archive))
    metadata_path = args.out.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["training"]["validation_quantized"], indent=2), flush=True)
    print(
        f"wrote {args.out} ({args.out.stat().st_size / (1024 * 1024):.2f} MiB) "
        f"and {metadata_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
