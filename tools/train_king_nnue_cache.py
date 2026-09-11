"""Train the compact king NNUE from a streamed, sharded evaluation cache.

Use :mod:`build_lichess_eval_cache` to create the cache. This trainer loads one compressed
shard at a time, shuffles rows inside that shard, and discards it before opening the next one.
Dataset size is therefore limited by disk rather than Python object memory.

The test split is sealed by default. Iterate hyperparameters against validation, then use
``--report-test`` once for a candidate already selected by paired engine games. Chessathon
games are not any cache split; they remain an independent competition-domain check.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from build_lichess_eval_cache import (
    CACHE_KIND,
    CACHE_VERSION,
    MAX_ACTIVE_FEATURES,
    SPLIT_NAMES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
)
from numpy.typing import NDArray
from torch import Tensor, nn
from train_king_nnue import (
    ACTIVATION_CLIP,
    INPUT_FEATURES,
    KING_BUCKETS,
    PIECE_PLANES,
    SQUARES,
    KingResidual,
    _sha256,
    quantize_model,
)

REQUIRED_ARRAYS = {
    "white_features",
    "black_features",
    "white_lengths",
    "black_lengths",
    "side_to_move",
    "target_cp",
    "teacher_cp",
    "baseline_cp",
    "depth",
    "knodes",
    "position_hash",
    "split",
    "bucket",
}


@dataclass(frozen=True)
class CacheInfo:
    """Validated cache manifest and resolved shard list."""

    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    shards: tuple[Path, ...]
    split_counts: dict[str, int]


@dataclass
class MetricAccumulator:
    """Streaming error statistics with a one-centipawn p90 histogram."""

    count: int = 0
    objective_absolute_sum: float = 0.0
    baseline_objective_absolute_sum: float = 0.0
    teacher_absolute_sum: float = 0.0
    baseline_teacher_absolute_sum: float = 0.0
    teacher_squared_sum: float = 0.0
    absolute_histogram: NDArray[np.int64] | None = None

    def update(
        self,
        predictions_cp: NDArray[np.float32],
        targets_cp: NDArray[np.float32],
        teacher_cp: NDArray[np.float32],
        baseline_cp: NDArray[np.float32],
    ) -> None:
        objective_error = predictions_cp.astype(np.float64) - targets_cp
        teacher_residual = teacher_cp.astype(np.float64) - baseline_cp
        teacher_error = predictions_cp.astype(np.float64) - teacher_residual
        absolute = np.abs(teacher_error)
        self.count += len(predictions_cp)
        self.objective_absolute_sum += float(np.abs(objective_error).sum())
        self.baseline_objective_absolute_sum += float(np.abs(targets_cp).sum())
        self.teacher_absolute_sum += float(absolute.sum())
        self.baseline_teacher_absolute_sum += float(np.abs(teacher_residual).sum())
        self.teacher_squared_sum += float(np.square(teacher_error).sum())
        bins = np.minimum(np.rint(absolute), 32_767).astype(np.int64)
        histogram = np.bincount(bins, minlength=32_768)
        if self.absolute_histogram is None:
            self.absolute_histogram = histogram
        else:
            self.absolute_histogram += histogram

    def metrics(self) -> dict[str, float]:
        if self.count == 0 or self.absolute_histogram is None:
            raise ValueError("cannot report metrics for an empty split")
        cutoff = math.ceil(self.count * 0.9)
        p90 = int(np.searchsorted(np.cumsum(self.absolute_histogram), cutoff))
        baseline_objective = self.baseline_objective_absolute_sum / self.count
        objective = self.objective_absolute_sum / self.count
        return {
            "positions": float(self.count),
            "baseline_objective_mae_cp": baseline_objective,
            "objective_mae_cp": objective,
            "objective_mae_gain_cp": baseline_objective - objective,
            "baseline_teacher_mae_cp": self.baseline_teacher_absolute_sum / self.count,
            "corrected_teacher_mae_cp": self.teacher_absolute_sum / self.count,
            "corrected_teacher_rmse_cp": math.sqrt(self.teacher_squared_sum / self.count),
            "corrected_teacher_p90_error_cp": float(p90),
        }


def _load_cache(
    path: Path, verify_shards: bool, baseline_evaluator: Path | None = None
) -> CacheInfo:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest not found: {manifest_path}")
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("kind") != CACHE_KIND or manifest.get("format_version") != CACHE_VERSION:
        raise ValueError(f"unsupported cache format: {manifest_path}")
    if not manifest.get("complete"):
        raise ValueError("cache is not complete; finish or checkpoint the builder first")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("cache has no evaluator provenance")
    project_root = Path(__file__).resolve().parents[1]
    expected_baseline = provenance.get("baseline_evaluator_sha256")
    baseline_path = baseline_evaluator or project_root / "fast_engine.py"
    actual_baseline = _sha256(baseline_path)
    if expected_baseline != actual_baseline:
        raise ValueError(
            "fast_engine.py changed after residual labels were cached; rebuild the cache "
            "before training"
        )

    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("cache contains no shards")
    shards: list[Path] = []
    for index, raw in enumerate(raw_shards):
        if not isinstance(raw, dict) or not isinstance(raw.get("file"), str):
            raise ValueError(f"invalid shard metadata at index {index}")
        shard = path / raw["file"]
        if not shard.is_file():
            raise FileNotFoundError(f"cache shard not found: {shard}")
        if verify_shards and _sha256(shard) != raw.get("sha256"):
            raise ValueError(f"cache shard checksum mismatch: {shard}")
        shards.append(shard)

    progress = manifest.get("progress")
    if not isinstance(progress, dict) or not isinstance(progress.get("split_counts"), dict):
        raise ValueError("cache manifest has no split counts")
    split_counts = {name: int(progress["split_counts"].get(name, 0)) for name in SPLIT_NAMES}
    if split_counts["train"] <= 0 or split_counts["validation"] <= 0:
        raise ValueError("cache needs non-empty training and validation splits")
    return CacheInfo(
        directory=path,
        manifest=manifest,
        manifest_sha256=_sha256(manifest_path),
        shards=tuple(shards),
        split_counts=split_counts,
    )


def _load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(sorted(missing))}")
        arrays = {name: archive[name] for name in REQUIRED_ARRAYS}
    _validate_shard(path, arrays)
    return arrays


def _validate_shard(path: Path, arrays: dict[str, np.ndarray]) -> None:
    rows = len(arrays["split"])
    for name, values in arrays.items():
        if len(values) != rows:
            raise ValueError(f"{path}: {name} has {len(values)} rows, expected {rows}")
    expected_shape = (rows, MAX_ACTIVE_FEATURES)
    for name in ("white_features", "black_features"):
        if arrays[name].shape != expected_shape:
            raise ValueError(f"{path}: {name} shape is not {expected_shape}")
    for name in ("white_lengths", "black_lengths"):
        if np.any(arrays[name] > MAX_ACTIVE_FEATURES):
            raise ValueError(f"{path}: invalid active feature count in {name}")
    if np.any(arrays["split"] > int(SPLIT_TEST)):
        raise ValueError(f"{path}: invalid split code")


def _split_rows(arrays: dict[str, np.ndarray], split: int) -> NDArray[np.int64]:
    return np.flatnonzero(arrays["split"] == split).astype(np.int64, copy=False)


def _embedding_bag(
    features: np.ndarray,
    lengths: np.ndarray,
    rows: NDArray[np.int64],
) -> tuple[Tensor, Tensor]:
    selected_features = features[rows]
    selected_lengths = lengths[rows].astype(np.int64, copy=False)
    active = np.arange(MAX_ACTIVE_FEATURES)[None, :] < selected_lengths[:, None]
    flat = selected_features[active].astype(np.int64, copy=False)
    offsets: NDArray[np.int64] = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(selected_lengths[:-1]))
    )
    return torch.from_numpy(flat), torch.from_numpy(offsets)


def _batch_tensors(
    arrays: dict[str, np.ndarray],
    rows: NDArray[np.int64],
    target_scale_cp: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    white, white_offsets = _embedding_bag(arrays["white_features"], arrays["white_lengths"], rows)
    black, black_offsets = _embedding_bag(arrays["black_features"], arrays["black_lengths"], rows)
    side = arrays["side_to_move"][rows].astype(np.float32, copy=False)
    targets = (arrays["target_cp"][rows] / target_scale_cp).astype(np.float32, copy=False)
    return (
        white,
        white_offsets,
        black,
        black_offsets,
        torch.from_numpy(side),
        torch.from_numpy(targets),
    )


def _batches(rows: NDArray[np.int64], batch_size: int) -> Iterator[NDArray[np.int64]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def _feature_coverage(cache: CacheInfo) -> tuple[dict[str, float | int], NDArray[np.bool_]]:
    active = np.zeros((3, INPUT_FEATURES), dtype=np.bool_)
    counted = np.zeros(3, dtype=np.int64)
    for path in cache.shards:
        arrays = _load_shard(path)
        for split in range(3):
            rows = _split_rows(arrays, split)
            counted[split] += len(rows)
            if not len(rows):
                continue
            for features, lengths in (
                (arrays["white_features"], arrays["white_lengths"]),
                (arrays["black_features"], arrays["black_lengths"]),
            ):
                selected = features[rows]
                selected_lengths = lengths[rows]
                mask = np.arange(MAX_ACTIVE_FEATURES)[None, :] < selected_lengths[:, None]
                values = selected[mask]
                if np.any(values < 0) or np.any(values >= INPUT_FEATURES):
                    raise ValueError(f"{path}: feature index outside [0, {INPUT_FEATURES})")
                active[split, values] = True
    for split, name in enumerate(SPLIT_NAMES):
        if counted[split] != cache.split_counts[name]:
            raise ValueError(
                f"manifest {name} count {cache.split_counts[name]} != shard count {counted[split]}"
            )
    validation_unseen = active[int(SPLIT_VALIDATION)] & ~active[int(SPLIT_TRAIN)]
    validation_rows = int(np.count_nonzero(active[int(SPLIT_VALIDATION)]))
    report: dict[str, float | int] = {
        "training_feature_rows": int(np.count_nonzero(active[int(SPLIT_TRAIN)])),
        "training_feature_fraction": float(np.mean(active[int(SPLIT_TRAIN)])),
        "validation_feature_rows": validation_rows,
        "validation_rows_unseen_in_training": int(np.count_nonzero(validation_unseen)),
        "validation_unseen_fraction": (
            float(np.count_nonzero(validation_unseen)) / validation_rows if validation_rows else 0.0
        ),
    }
    return report, active[int(SPLIT_TRAIN)]


def _evaluate_float(
    model: KingResidual,
    cache: CacheInfo,
    split: int,
    batch_size: int,
    target_scale_cp: float,
) -> dict[str, float]:
    accumulator = MetricAccumulator()
    model.eval()
    with torch.no_grad():
        for path in cache.shards:
            arrays = _load_shard(path)
            for rows in _batches(_split_rows(arrays, split), batch_size):
                white, white_offsets, black, black_offsets, side, _target = _batch_tensors(
                    arrays, rows, target_scale_cp
                )
                predictions = (
                    model(white, white_offsets, black, black_offsets, side).cpu().numpy()
                    * target_scale_cp
                ).astype(np.float32, copy=False)
                accumulator.update(
                    predictions,
                    arrays["target_cp"][rows],
                    arrays["teacher_cp"][rows],
                    arrays["baseline_cp"][rows],
                )
    return accumulator.metrics()


def _quantized_batch(
    quantized: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    rows: NDArray[np.int64],
    target_scale_cp: float,
) -> NDArray[np.float32]:
    feature_weights = quantized["feature_weights"]
    bias = quantized["feature_bias"]
    clip = quantized["activation_clip"]

    def accumulator(features: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        selected = features[rows].copy()
        selected_lengths = lengths[rows]
        active = np.arange(MAX_ACTIVE_FEATURES)[None, :] < selected_lengths[:, None]
        selected[~active] = 0
        embedded = feature_weights[selected].astype(np.int32)
        embedded *= active[:, :, None]
        values = embedded.sum(axis=1, dtype=np.int32) + bias
        return cast(np.ndarray, np.minimum(np.maximum(values, 0), clip))

    white = accumulator(arrays["white_features"], arrays["white_lengths"])
    black = accumulator(arrays["black_features"], arrays["black_lengths"])
    feature_scale = quantized["feature_scale"]
    output = quantized["output_weights"].astype(np.float32) * float(quantized["output_scale"])
    difference = (white - black).astype(np.float32) * feature_scale
    tempo = float(quantized["tempo"])
    side = arrays["side_to_move"][rows].astype(np.float32)
    return cast(
        NDArray[np.float32],
        ((difference @ output) + tempo * side) * target_scale_cp,
    )


def _evaluate_quantized(
    quantized: dict[str, np.ndarray],
    cache: CacheInfo,
    split: int,
    batch_size: int,
    target_scale_cp: float,
) -> dict[str, float]:
    accumulator = MetricAccumulator()
    for path in cache.shards:
        arrays = _load_shard(path)
        for rows in _batches(_split_rows(arrays, split), batch_size):
            predictions = _quantized_batch(quantized, arrays, rows, target_scale_cp).astype(
                np.float32, copy=False
            )
            accumulator.update(
                predictions,
                arrays["target_cp"][rows],
                arrays["teacher_cp"][rows],
                arrays["baseline_cp"][rows],
            )
    return accumulator.metrics()


def _initialise_model(hidden: int, training_features: NDArray[np.bool_], seed: int) -> KingResidual:
    torch.manual_seed(seed)
    model = KingResidual(hidden)
    feature_indices = np.flatnonzero(training_features).astype(np.int64, copy=False)
    with torch.no_grad():
        indices = torch.from_numpy(feature_indices)
        initial_values = torch.empty((len(feature_indices), hidden))
        nn.init.normal_(initial_values, mean=0.0, std=0.008)
        model.feature_weights.weight.index_copy_(0, indices, initial_values)
    return model


def train_cached(
    cache: CacheInfo,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    target_scale_cp: float,
    seed: int,
    best_checkpoint: Path | None = None,
) -> tuple[KingResidual, dict[str, Any]]:
    coverage, training_features = _feature_coverage(cache)
    model = _initialise_model(hidden, training_features, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05
    )
    loss_function = nn.SmoothL1Loss(beta=140.0 / target_scale_cp)
    rng = np.random.default_rng(seed)
    best_mae = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        shard_order = rng.permutation(len(cache.shards))
        loss_sum = 0.0
        trained = 0
        for shard_index in shard_order:
            arrays = _load_shard(cache.shards[int(shard_index)])
            rows = _split_rows(arrays, int(SPLIT_TRAIN))
            rng.shuffle(rows)
            for batch_rows in _batches(rows, batch_size):
                white, white_offsets, black, black_offsets, side, targets = _batch_tensors(
                    arrays, batch_rows, target_scale_cp
                )
                optimizer.zero_grad(set_to_none=True)
                predictions = model(white, white_offsets, black, black_offsets, side)
                loss = loss_function(predictions, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_rows)
                trained += len(batch_rows)
        scheduler.step()

        validation = _evaluate_float(
            model, cache, int(SPLIT_VALIDATION), batch_size, target_scale_cp
        )
        validation_mae = validation["objective_mae_cp"]
        history.append(
            {
                "epoch": float(epoch),
                "train_huber": loss_sum / max(1, trained),
                "validation_mae_cp": validation_mae,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        print(
            f"epoch {epoch:02d}/{epochs}: train Huber {loss_sum / max(1, trained):.5f}, "
            f"held-out objective MAE {validation_mae:.1f} cp",
            flush=True,
        )
        if validation_mae < best_mae - 0.05:
            best_mae = validation_mae
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            if best_checkpoint is not None:
                best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = best_checkpoint.with_suffix(".partial.pt")
                torch.save(
                    {
                        "model_state": best_state,
                        "hidden": hidden,
                        "epoch": epoch,
                        "validation_mae_cp": best_mae,
                        "cache_manifest_sha256": cache.manifest_sha256,
                        "seed": seed,
                        "history": history,
                    },
                    temporary,
                )
                temporary.replace(best_checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation = _evaluate_float(model, cache, int(SPLIT_VALIDATION), batch_size, target_scale_cp)
    return model, {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "validation_float": validation,
        "feature_coverage": coverage,
    }


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("hidden", "epochs", "batch_size", "patience", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0.0 or args.target_scale_cp <= 0.0:
        raise ValueError("learning rate and target scale must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight decay cannot be negative")


def _save(
    args: argparse.Namespace,
    cache: CacheInfo,
    model: KingResidual,
    report: dict[str, Any],
    started: float,
) -> None:
    quantized = quantize_model(model)
    report["validation_quantized"] = _evaluate_quantized(
        quantized,
        cache,
        int(SPLIT_VALIDATION),
        args.batch_size,
        args.target_scale_cp,
    )
    if args.report_test:
        report["test_float"] = _evaluate_float(
            model, cache, int(SPLIT_TEST), args.batch_size, args.target_scale_cp
        )
        report["test_quantized"] = _evaluate_quantized(
            quantized, cache, int(SPLIT_TEST), args.batch_size, args.target_scale_cp
        )
    project_root = Path(__file__).resolve().parents[1]
    baseline_path = args.baseline_evaluator or project_root / "fast_engine.py"
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
            "baseline_evaluator": str(baseline_path),
            "baseline_evaluator_sha256": _sha256(baseline_path),
            "cache_manifest_sha256": cache.manifest_sha256,
            "cache_directory": cache.directory.as_posix(),
            "cache_source": cache.manifest.get("config", {}).get("source"),
            "cache_source_sha256": cache.manifest.get("config", {}).get("source_sha256"),
            "source_license": cache.manifest.get("source_license"),
        },
        "data": {
            "split_counts": cache.split_counts,
            "validation_split": "stable canonical-FEN hash; competition positions excluded",
            "test_split_reported": args.report_test,
            "test_split_policy": "sealed unless --report-test is explicitly supplied",
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
            **report,
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
    print(json.dumps(report["validation_quantized"], indent=2), flush=True)
    print(f"wrote {args.out} and {metadata_path}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--baseline-evaluator",
        type=Path,
        help="immutable source snapshot matching the cache's baseline SHA256",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--target-scale-cp", type=float, default=400.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20_260_907)
    parser.add_argument("--verify-shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--report-test",
        action="store_true",
        help="evaluate the sealed test split (only for a final preselected candidate)",
    )
    parser.add_argument("--out", type=Path, default=Path("weights/king_nnue.npz"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cache = _load_cache(args.cache, args.verify_shards, args.baseline_evaluator)
    model, report = train_cached(
        cache,
        args.hidden,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.patience,
        args.target_scale_cp,
        args.seed,
        args.out.with_suffix(".best.pt"),
    )
    _save(args, cache, model, report, started)


if __name__ == "__main__":
    main()
