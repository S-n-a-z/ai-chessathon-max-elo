"""Stream the Lichess CC0 evaluation dump into bounded NNUE training shards.

The source dump is intentionally *not* downloaded by this tool.  Point ``--source`` at a
local ``.jsonl[.zst]`` file, ``-`` for stdin, or the official HTTPS URL.  The input is read
once and fixed-size compressed NumPy shards are checkpointed as they fill, so hundreds of
millions of source rows never have to fit in RAM.

Lichess documents the dump at https://database.lichess.org/#evals.  For every position this
tool selects the evaluation with the greatest depth (then greatest node count) and its first
PV, as recommended there.  Dump ``cp`` values are already from White's point of view; do not
flip them for Black-to-move FENs.

Known AI Chessathon starts and game mainlines are excluded from every split.  They remain an
external, competition-specific test set and cannot silently leak into training.

Example bounded pilot (streams compressed bytes without keeping the 20+ GiB download)::

    python tools/build_lichess_eval_cache.py \
        --source https://database.lichess.org/lichess_db_eval.jsonl.zst \
        --out C:/datasets/blockshark-eval-pilot --max-accepted 250000

An interrupted local-file run can be continued with ``--resume``.  Resuming a zstd stream
must replay and discard the already checkpointed prefix because ordinary zstd files are not
seekable, but no accepted rows or labels are recomputed.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import importlib.util
import io
import json
import lzma
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, TextIO, cast
from urllib.request import Request, urlopen

import chess
import chess.pgn
import numpy as np
import train_king_nnue
from numpy.typing import NDArray
from train_king_nnue import _current_white_score, _sha256, sparse_features

CACHE_KIND = "blockshark.king-nnue-labelled-shards"
CACHE_VERSION = 1
MAX_ACTIVE_FEATURES = 32
SPLIT_TRAIN = np.uint8(0)
SPLIT_VALIDATION = np.uint8(1)
SPLIT_TEST = np.uint8(2)
SPLIT_NAMES = ("train", "validation", "test")
LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
LICHESS_DATABASE_URL = "https://database.lichess.org/#evals"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


@dataclass(frozen=True)
class LabelledPosition:
    """One fixed-size cache row."""

    white_features: tuple[int, ...]
    black_features: tuple[int, ...]
    side_to_move: int
    target_cp: float
    teacher_cp: float
    baseline_cp: float
    depth: int
    knodes: int
    position_hash: int
    split: int
    bucket: int


@dataclass(frozen=True)
class SelectedEvaluation:
    """The deepest available first-PV score in one Lichess row."""

    cp: float
    depth: int
    knodes: int
    best_move: str | None = None


@dataclass(frozen=True)
class SourceRecord:
    """One source position after collapsing all of its PV/evaluation rows."""

    rows_processed: int
    fen: str | None
    evaluation: SelectedEvaluation | None


def _is_url(value: str) -> bool:
    return value.startswith("https://") or value.startswith("http://")


def _source_description(source: str) -> dict[str, Any]:
    if source == "-":
        return {"kind": "stdin", "name": "stdin"}
    if _is_url(source):
        return {"kind": "url", "url": source}
    path = Path(source).resolve()
    stat = path.stat()
    return {
        "kind": "file",
        "path": path.as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _compression(source: str, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = source.lower()
    if lowered.endswith(".zst"):
        return "zstd"
    if lowered.endswith(".gz"):
        return "gzip"
    if lowered.endswith(".bz2"):
        return "bz2"
    if lowered.endswith(".xz") or lowered.endswith(".lzma"):
        return "xz"
    return "none"


class _StreamingText(AbstractContextManager[TextIO]):
    """Open local, stdin, or HTTP text, optionally through a bounded zstd pipe."""

    def __init__(self, source: str, compression: str, zstd: Path | None) -> None:
        self.source = source
        self.compression = _compression(source, compression)
        self.zstd = zstd
        self.text: TextIO | None = None
        self.binary: BinaryIO | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.feeder: threading.Thread | None = None
        self.feeder_error: list[BaseException] = []
        self.exhausted = False

    def _open_binary(self) -> BinaryIO:
        if self.source == "-":
            return sys.stdin.buffer
        if _is_url(self.source):
            request = Request(self.source, headers={"User-Agent": "BlockShark-NNUE/1"})
            return cast(BinaryIO, urlopen(request, timeout=60))
        return Path(self.source).open("rb")

    def _feed_decoder(self) -> None:
        assert self.binary is not None
        assert self.process is not None
        assert self.process.stdin is not None
        try:
            while chunk := self.binary.read(1024 * 1024):
                self.process.stdin.write(chunk)
            self.process.stdin.close()
        except (BrokenPipeError, OSError, ValueError) as error:
            if self.exhausted:
                self.feeder_error.append(error)
        except BaseException as error:  # pragma: no cover - preserves worker failure
            self.feeder_error.append(error)

    def __enter__(self) -> TextIO:
        if self.compression == "none":
            self.binary = self._open_binary()
            self.text = io.TextIOWrapper(self.binary, encoding="utf-8", errors="replace")
            return self.text
        if self.compression in {"gzip", "bz2", "xz"}:
            if self.source == "-" or _is_url(self.source):
                raise ValueError(
                    f"streamed {self.compression} is not supported; use zstd or a local file"
                )
            openers: dict[str, Any] = {"gzip": gzip.open, "bz2": bz2.open, "xz": lzma.open}
            self.text = cast(
                TextIO,
                openers[self.compression](
                    self.source, mode="rt", encoding="utf-8", errors="replace"
                ),
            )
            return self.text
        if self.compression != "zstd":
            raise ValueError(f"unsupported compression: {self.compression}")

        executable = str(self.zstd) if self.zstd is not None else shutil.which("zstd")
        if executable is None:
            raise FileNotFoundError(
                "zstd executable not found; install zstd or pass --zstd C:/path/to/zstd.exe"
            )
        if self.source != "-" and not _is_url(self.source):
            self.process = subprocess.Popen(
                [executable, "-dcq", "--", str(Path(self.source).resolve())],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            self.binary = self._open_binary()
            self.process = subprocess.Popen(
                [executable, "-dcq"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.feeder = threading.Thread(
                target=self._feed_decoder, name="zstd-input", daemon=True
            )
            self.feeder.start()
        assert self.process.stdout is not None
        self.text = io.TextIOWrapper(self.process.stdout, encoding="utf-8", errors="replace")
        return self.text

    def mark_exhausted(self) -> None:
        """Tell the context manager that EOF, rather than a requested early stop, was reached."""

        self.exhausted = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self.text is not None:
            self.text.close()
        if self.binary is not None and self.binary is not sys.stdin.buffer:
            self.binary.close()
        if self.process is None:
            return None

        if not self.exhausted and self.process.poll() is None:
            self.process.terminate()
        if self.feeder is not None:
            self.feeder.join(timeout=10)
        try:
            return_code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return_code = self.process.wait()
        assert self.process.stderr is not None
        stderr = self.process.stderr.read()
        if exc_type is None and self.exhausted:
            if self.feeder_error:
                raise RuntimeError("input feeder failed") from self.feeder_error[0]
            if return_code != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"zstd exited with {return_code}: {detail}")
        return None


def _canonical(board: chess.Board) -> str:
    return str(board.epd())


def _known_competition_positions(public_starts: Path, games: Path) -> tuple[set[str], str]:
    positions: set[str] = set()
    digest = hashlib.sha256()
    if public_starts.is_file():
        raw = public_starts.read_bytes()
        digest.update(public_starts.as_posix().encode())
        digest.update(raw)
        payload = json.loads(raw.decode("utf-8"))
        starts = payload.get("starts", [])
        if isinstance(starts, list):
            for item in starts:
                if isinstance(item, dict) and isinstance(item.get("fen"), str):
                    try:
                        positions.add(_canonical(chess.Board(item["fen"])))
                    except ValueError:
                        continue

    if games.is_dir():
        for path in sorted(games.rglob("*.pgn")):
            raw = path.read_bytes()
            digest.update(path.as_posix().encode())
            digest.update(raw)
            with path.open(encoding="utf-8-sig", errors="replace") as stream:
                while game := chess.pgn.read_game(stream):
                    board = game.board()
                    positions.add(_canonical(board))
                    for move in game.mainline_moves():
                        board.push(move)
                        positions.add(_canonical(board))
    return positions, digest.hexdigest()


def _select_evaluation(payload: dict[str, Any]) -> SelectedEvaluation | None:
    raw_evaluations = payload.get("evals")
    if not isinstance(raw_evaluations, list):
        return None
    selected_rank: tuple[int, int] | None = None
    selected: SelectedEvaluation | None = None
    for raw in raw_evaluations:
        if not isinstance(raw, dict):
            continue
        pvs = raw.get("pvs")
        if not isinstance(pvs, list) or not pvs or not isinstance(pvs[0], dict):
            continue
        depth = raw.get("depth", 0)
        knodes = raw.get("knodes", 0)
        if isinstance(depth, bool) or not isinstance(depth, int):
            continue
        if isinstance(knodes, bool) or not isinstance(knodes, int):
            knodes = 0
        node_count = max(0, knodes)
        rank = (depth, node_count)
        if selected_rank is not None and rank <= selected_rank:
            continue
        selected_rank = rank
        # Decide from the selected evaluation's PV1.  If it is a mate, return no
        # centipawn label rather than silently falling back to a weaker analysis.
        cp = pvs[0].get("cp")
        selected = (
            SelectedEvaluation(float(cp), depth, node_count, _first_move(pvs[0].get("line")))
            if isinstance(cp, (int, float)) and not isinstance(cp, bool)
            else None
        )
    return selected


def _first_move(line: object) -> str | None:
    moves = line.split() if isinstance(line, str) else []
    return moves[0] if moves else None


def _parquet_candidate(
    cp: object, depth: object, knodes: object, line: object = None
) -> SelectedEvaluation | None:
    if isinstance(cp, bool) or not isinstance(cp, (int, float)):
        return None
    if isinstance(depth, bool) or not isinstance(depth, int):
        return None
    node_count = knodes if isinstance(knodes, int) and not isinstance(knodes, bool) else 0
    return SelectedEvaluation(float(cp), depth, max(0, node_count), _first_move(line))


def _is_quiet_label(board: chess.Board, selected: SelectedEvaluation) -> bool:
    """Keep labels suited to static evaluation, before a quiet teacher move.

    This is a filter, not a claim that the position contains no tactics. The
    original FEN and score remain unchanged, so the canonical split stays stable.
    """
    if board.is_check() or selected.best_move is None:
        return False
    try:
        move = chess.Move.from_uci(selected.best_move)
    except ValueError:
        return False
    return (
        move in board.legal_moves
        and not move.promotion
        and not board.is_capture(move)
        and not board.gives_check(move)
    )


def _iter_jsonl_records(stream: TextIO) -> Iterator[SourceRecord]:
    for line_number, line in enumerate(stream, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            yield SourceRecord(line_number, None, None)
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("fen"), str):
            yield SourceRecord(line_number, None, None)
            continue
        yield SourceRecord(line_number, raw["fen"], _select_evaluation(raw))


def _iter_parquet_records(
    path: Path, batch_size: int, include_pv: bool = False
) -> Iterator[SourceRecord]:
    """Collapse contiguous flattened Parquet PV rows without loading a row group at once."""

    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - depends on training workstation
        raise RuntimeError(
            "Parquet input needs the optional offline dependency: pip install pyarrow"
        ) from error

    parquet_file = parquet.ParquetFile(path)
    required = {"fen", "depth", "knodes", "cp", "mate"}
    if include_pv:
        required.add("line")
    available = set(parquet_file.schema.names)
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Parquet source is missing columns: {', '.join(missing)}")

    pending_fen: str | None = None
    pending_best: SelectedEvaluation | None = None
    pending_rank: tuple[int, int] | None = None
    pending_last_row = 0
    raw_row = 0
    columns = ["fen", "depth", "knodes", "cp"]
    if include_pv:
        columns.append("line")
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        values = cast(dict[str, list[object]], batch.to_pydict())
        lines = values.get("line", [None] * len(values["fen"]))
        for fen_raw, depth, knodes, cp, line in zip(
            values["fen"],
            values["depth"],
            values["knodes"],
            values["cp"],
            lines,
            strict=True,
        ):
            raw_row += 1
            fen = fen_raw if isinstance(fen_raw, str) else None
            if pending_fen is not None and fen != pending_fen:
                yield SourceRecord(pending_last_row, pending_fen, pending_best)
                pending_best = None
                pending_rank = None
            if fen is None:
                yield SourceRecord(raw_row, None, None)
                pending_fen = None
                pending_last_row = raw_row
                continue
            if pending_fen is None or fen != pending_fen:
                pending_fen = fen
            candidate = _parquet_candidate(cp, depth, knodes, line)
            raw_depth = depth if isinstance(depth, int) and not isinstance(depth, bool) else -1
            raw_knodes = knodes if isinstance(knodes, int) and not isinstance(knodes, bool) else 0
            rank = (raw_depth, max(0, raw_knodes))
            if pending_rank is None or rank > pending_rank:
                # Equal-depth/equal-node rows are later PVs from the same analysis;
                # retaining the first occurrence is precisely PV1.  A selected mate
                # intentionally leaves ``pending_best`` empty instead of falling back.
                pending_rank = rank
                pending_best = candidate
            pending_last_row = raw_row
    if pending_fen is not None:
        yield SourceRecord(pending_last_row, pending_fen, pending_best)


def _input_format(source: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "parquet" if source.lower().endswith(".parquet") else "jsonl"


def _hashes(fen: str, seed: int) -> tuple[int, int]:
    digest = hashlib.sha256(f"{seed}:{fen}".encode()).digest()
    return (
        int.from_bytes(digest[:8], "little", signed=False),
        int.from_bytes(digest[8:16], "little", signed=False),
    )


def _split(split_hash: int, validation_fraction: float, test_fraction: float) -> int:
    unit = split_hash / 2**64
    if unit < test_fraction:
        return int(SPLIT_TEST)
    if unit < test_fraction + validation_fraction:
        return int(SPLIT_VALIDATION)
    return int(SPLIT_TRAIN)


def _phase_bucket(board: chess.Board) -> int:
    phase = 0
    for piece_type, weight in (
        (chess.KNIGHT, 1),
        (chess.BISHOP, 1),
        (chess.ROOK, 2),
        (chess.QUEEN, 4),
    ):
        phase += weight * len(board.pieces(piece_type, chess.WHITE))
        phase += weight * len(board.pieces(piece_type, chess.BLACK))
    if phase >= 18:
        return 0
    if phase >= 7:
        return 1
    return 2


def _balance_bucket(board: chess.Board, cp: float) -> int:
    if cp < -400.0:
        score_bucket = 0
    elif cp < -100.0:
        score_bucket = 1
    elif cp <= 100.0:
        score_bucket = 2
    elif cp <= 400.0:
        score_bucket = 3
    else:
        score_bucket = 4
    return _phase_bucket(board) * 5 + score_bucket


def _bucket_quotas(max_accepted: int) -> NDArray[np.int64]:
    quotas = np.full(15, max_accepted // 15, dtype=np.int64)
    quotas[: max_accepted % 15] += 1
    return quotas


class ShardBuffer:
    """Preallocated bounded-memory writer for the cache's fixed-size row schema."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.count = 0
        self.white_features = np.full((capacity, MAX_ACTIVE_FEATURES), -1, dtype=np.int32)
        self.black_features = np.full((capacity, MAX_ACTIVE_FEATURES), -1, dtype=np.int32)
        self.white_lengths = np.empty(capacity, dtype=np.uint8)
        self.black_lengths = np.empty(capacity, dtype=np.uint8)
        self.side_to_move = np.empty(capacity, dtype=np.int8)
        self.target_cp = np.empty(capacity, dtype=np.float32)
        self.teacher_cp = np.empty(capacity, dtype=np.float32)
        self.baseline_cp = np.empty(capacity, dtype=np.float32)
        self.depth = np.empty(capacity, dtype=np.uint16)
        self.knodes = np.empty(capacity, dtype=np.uint32)
        self.position_hash = np.empty(capacity, dtype=np.uint64)
        self.split = np.empty(capacity, dtype=np.uint8)
        self.bucket = np.empty(capacity, dtype=np.uint8)

    def append(self, row: LabelledPosition) -> None:
        if self.count >= self.capacity:
            raise RuntimeError("shard buffer is full")
        if len(row.white_features) > MAX_ACTIVE_FEATURES or len(row.black_features) > 32:
            raise ValueError("a chess position cannot contain more than 32 active pieces")
        index = self.count
        self.white_features[index, : len(row.white_features)] = row.white_features
        self.black_features[index, : len(row.black_features)] = row.black_features
        self.white_lengths[index] = len(row.white_features)
        self.black_lengths[index] = len(row.black_features)
        self.side_to_move[index] = row.side_to_move
        self.target_cp[index] = row.target_cp
        self.teacher_cp[index] = row.teacher_cp
        self.baseline_cp[index] = row.baseline_cp
        self.depth[index] = min(row.depth, int(np.iinfo(np.uint16).max))
        self.knodes[index] = min(row.knodes, int(np.iinfo(np.uint32).max))
        self.position_hash[index] = row.position_hash
        self.split[index] = row.split
        self.bucket[index] = row.bucket
        self.count += 1

    def arrays(self) -> dict[str, np.ndarray]:
        end = self.count
        return {
            "white_features": self.white_features[:end],
            "black_features": self.black_features[:end],
            "white_lengths": self.white_lengths[:end],
            "black_lengths": self.black_lengths[:end],
            "side_to_move": self.side_to_move[:end],
            "target_cp": self.target_cp[:end],
            "teacher_cp": self.teacher_cp[:end],
            "baseline_cp": self.baseline_cp[:end],
            "depth": self.depth[:end],
            "knodes": self.knodes[:end],
            "position_hash": self.position_hash[:end],
            "split": self.split[:end],
            "bucket": self.bucket[:end],
        }

    def reset(self) -> None:
        self.count = 0
        self.white_features.fill(-1)
        self.black_features.fill(-1)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_shard(out: Path, number: int, buffer: ShardBuffer) -> dict[str, Any]:
    filename = f"shard-{number:05d}.npz"
    target = out / filename
    temporary = out / f".{filename}.tmp"
    arrays = buffer.arrays()
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], arrays))
    os.replace(temporary, target)
    splits = arrays["split"]
    buckets = arrays["bucket"]
    return {
        "file": filename,
        "sha256": _sha256(target),
        "positions": buffer.count,
        "split_counts": {
            name: int(np.count_nonzero(splits == index)) for index, name in enumerate(SPLIT_NAMES)
        },
        "bucket_counts": np.bincount(buckets, minlength=15).astype(int).tolist(),
    }


def _config(args: argparse.Namespace, competition_fingerprint: str) -> dict[str, Any]:
    return {
        "source": _source_description(args.source),
        "source_sha256": args.source_sha256,
        "input_format": _input_format(args.source, args.input_format),
        "compression": (
            _compression(args.source, args.compression)
            if _input_format(args.source, args.input_format) == "jsonl"
            else "parquet-internal"
        ),
        "sample_rate": args.sample_rate,
        "balanced": args.balanced,
        "min_depth": args.min_depth,
        "max_teacher_abs_cp": args.max_teacher_abs_cp,
        "residual_clip_cp": args.residual_clip_cp,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "competition_exclusion_sha256": competition_fingerprint,
        "score_convention": "Lichess dump cp is white-centric; no turn-dependent sign flip",
        "selection": "greatest depth, then greatest knodes, first PV",
        **(
            {"quiet_only": "not checked; PV1 legal, non-capture, non-promotion, non-check"}
            if args.quiet_only
            else {}
        ),
    }


def _new_manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    baseline = args.baseline_evaluator or project_root / "fast_engine.py"
    return {
        "format_version": CACHE_VERSION,
        "kind": CACHE_KIND,
        "source_license": "CC0",
        "source_documentation": LICHESS_DATABASE_URL,
        "official_source_url": LICHESS_EVAL_URL,
        "provenance": {
            "builder_sha256": _sha256(Path(__file__).resolve()),
            "baseline_evaluator": str(baseline),
            "baseline_evaluator_sha256": _sha256(baseline),
        },
        "config": config,
        "progress": {
            "rows_processed": 0,
            "accepted": 0,
            "invalid_json": 0,
            "invalid_fen": 0,
            "missing_cp": 0,
            "below_min_depth": 0,
            "outside_score_cutoff": 0,
            "competition_positions_excluded": 0,
            "hash_sample_rejected": 0,
            "balance_bucket_full": 0,
            "bucket_counts": [0] * 15,
            "split_counts": {name: 0 for name in SPLIT_NAMES},
        },
        "shards": [],
        "complete": False,
        "trainer_contract": {
            "feature_layout": "train_king_nnue.sparse_features v1",
            "target": "clipped white-centric teacher cp minus fast_engine static cp",
            "labelled_positions_exported_with_submission": False,
        },
    }


def _load_or_create_manifest(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    path = args.out / "manifest.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"cache already exists: {path}; pass --resume to continue")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("kind") != CACHE_KIND or manifest.get("format_version") != CACHE_VERSION:
            raise ValueError(f"unsupported cache manifest: {path}")
        if manifest.get("config") != config:
            raise ValueError("resume configuration or source metadata differs from manifest")
        if manifest.get("complete"):
            raise ValueError("cache is already marked complete")
        return cast(dict[str, Any], manifest)
    if args.resume:
        raise FileNotFoundError(f"cannot resume; manifest does not exist: {path}")
    args.out.mkdir(parents=True, exist_ok=False)
    return _new_manifest(args, config)


def _flush(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    buffer: ShardBuffer,
    progress: dict[str, Any],
) -> None:
    if buffer.count:
        shards = cast(list[dict[str, Any]], manifest["shards"])
        shards.append(_write_shard(args.out, len(shards), buffer))
        buffer.reset()
    manifest["progress"] = progress
    _atomic_json(args.out / "manifest.json", manifest)


def _validate_args(args: argparse.Namespace) -> None:
    if args.source != "-" and not _is_url(args.source) and not Path(args.source).is_file():
        raise FileNotFoundError(f"source not found: {args.source}")
    if args.source == "-" and args.resume:
        raise ValueError("stdin cannot be resumed")
    if _input_format(args.source, args.input_format) == "parquet" and (
        args.source == "-" or _is_url(args.source)
    ):
        raise ValueError("Parquet sources must be local files (HTTP JSONL.zst can stream directly)")
    if not 0.0 < args.sample_rate <= 1.0:
        raise ValueError("--sample-rate must be in (0, 1]")
    if args.max_rows < 0 or args.max_accepted < 0:
        raise ValueError("--max-rows and --max-accepted cannot be negative")
    if (
        args.min_depth < 0
        or args.shard_size <= 0
        or args.checkpoint_rows <= 0
        or args.parquet_batch_size <= 0
    ):
        raise ValueError("depth, shard size, and checkpoint rows must be positive")
    if args.max_teacher_abs_cp <= 0.0 or args.residual_clip_cp <= 0.0:
        raise ValueError("score cutoffs must be positive")
    if args.validation_fraction < 0.0 or args.test_fraction < 0.0:
        raise ValueError("split fractions cannot be negative")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation + test fractions must be below 1")
    if args.source_sha256 and (
        len(args.source_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in args.source_sha256)
    ):
        raise ValueError("--source-sha256 must contain exactly 64 hexadecimal characters")


def build_cache(args: argparse.Namespace) -> None:
    _validate_args(args)
    competition_positions, competition_fingerprint = _known_competition_positions(
        args.public_starts, args.competition_games
    )
    config = _config(args, competition_fingerprint)
    manifest = _load_or_create_manifest(args, config)
    progress = cast(dict[str, Any], manifest["progress"])
    resume_rows = int(progress["rows_processed"])
    bucket_counts = np.asarray(progress["bucket_counts"], dtype=np.int64)
    split_counts = cast(dict[str, int], progress["split_counts"])
    buffer = ShardBuffer(args.shard_size)
    quotas = _bucket_quotas(args.max_accepted) if args.balanced and args.max_accepted else None
    accepted_start = int(progress["accepted"])
    next_checkpoint = resume_rows + args.checkpoint_rows
    stopped_early = False

    input_format = _input_format(args.source, args.input_format)
    source: _StreamingText | None = None
    stream: TextIO | None = None
    if input_format == "parquet":
        records = _iter_parquet_records(Path(args.source), args.parquet_batch_size, args.quiet_only)
    else:
        source = _StreamingText(args.source, args.compression, args.zstd)
        stream = source.__enter__()
        records = _iter_jsonl_records(stream)

    source_exhausted = False
    try:
        for record in records:
            row_number = record.rows_processed
            if row_number <= resume_rows:
                continue
            if args.max_rows and row_number > args.max_rows:
                stopped_early = True
                break
            progress["rows_processed"] = row_number
            if record.fen is None:
                progress["invalid_json"] += 1
                continue
            selected = record.evaluation
            if selected is None:
                progress["missing_cp"] += 1
                continue
            if selected.depth < args.min_depth:
                progress["below_min_depth"] += 1
                continue
            if abs(selected.cp) > args.max_teacher_abs_cp:
                progress["outside_score_cutoff"] += 1
                continue
            try:
                board = chess.Board(record.fen)
            except ValueError:
                progress["invalid_fen"] += 1
                continue
            if (
                not board.is_valid()
                or board.king(chess.WHITE) is None
                or board.king(chess.BLACK) is None
            ):
                progress["invalid_fen"] += 1
                continue
            canonical = _canonical(board)
            if canonical in competition_positions:
                progress["competition_positions_excluded"] += 1
                continue
            sample_hash, split_hash = _hashes(canonical, args.seed)
            if sample_hash / 2**64 >= args.sample_rate:
                progress["hash_sample_rejected"] += 1
                continue
            bucket = _balance_bucket(board, selected.cp)
            if quotas is not None and bucket_counts[bucket] >= quotas[bucket]:
                progress["balance_bucket_full"] += 1
                continue

            if args.quiet_only and not _is_quiet_label(board, selected):
                progress["nonquiet_rejected"] = progress.get("nonquiet_rejected", 0) + 1
                continue

            baseline_cp = _current_white_score(board)
            target_cp = float(
                np.clip(
                    selected.cp - baseline_cp,
                    -args.residual_clip_cp,
                    args.residual_clip_cp,
                )
            )
            split = _split(split_hash, args.validation_fraction, args.test_fraction)
            buffer.append(
                LabelledPosition(
                    white_features=sparse_features(board, chess.WHITE),
                    black_features=sparse_features(board, chess.BLACK),
                    side_to_move=1 if board.turn == chess.WHITE else -1,
                    target_cp=target_cp,
                    teacher_cp=selected.cp,
                    baseline_cp=baseline_cp,
                    depth=selected.depth,
                    knodes=selected.knodes,
                    position_hash=sample_hash,
                    split=split,
                    bucket=bucket,
                )
            )
            progress["accepted"] += 1
            bucket_counts[bucket] += 1
            split_counts[SPLIT_NAMES[split]] += 1
            progress["bucket_counts"] = bucket_counts.astype(int).tolist()
            progress["split_counts"] = split_counts

            accepted_this_run = int(progress["accepted"]) - accepted_start
            if accepted_this_run % 1_000 == 0:
                print(
                    f"accepted {progress['accepted']:,} after {row_number:,} rows",
                    flush=True,
                )
            if buffer.count >= args.shard_size or row_number >= next_checkpoint:
                _flush(args, manifest, buffer, progress)
                next_checkpoint = row_number + args.checkpoint_rows

            if args.max_accepted and int(progress["accepted"]) >= args.max_accepted:
                stopped_early = True
                break
        else:
            source_exhausted = True
            if source is not None:
                source.mark_exhausted()
    finally:
        close_records = getattr(records, "close", None)
        if callable(close_records):
            close_records()
        if source is not None:
            source.__exit__(*sys.exc_info())

    _flush(args, manifest, buffer, progress)
    manifest["complete"] = source_exhausted or stopped_early
    manifest["completion_reason"] = (
        "source exhausted" if source_exhausted else "requested row/acceptance limit reached"
    )
    _atomic_json(args.out / "manifest.json", manifest)
    print(
        f"cache complete: {progress['accepted']:,} positions in "
        f"{len(manifest['shards'])} shards at {args.out}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="JSONL[.zst] path, HTTP(S) URL, or -")
    parser.add_argument("--out", type=Path, required=True, help="new cache directory")
    parser.add_argument(
        "--baseline-evaluator",
        type=Path,
        help="immutable evaluator source snapshot used for residual labels",
    )
    parser.add_argument("--input-format", choices=("auto", "jsonl", "parquet"), default="auto")
    parser.add_argument(
        "--compression", choices=("auto", "none", "zstd", "gzip", "bz2", "xz"), default="auto"
    )
    parser.add_argument("--zstd", type=Path, help="zstd executable used for .zst streams")
    parser.add_argument(
        "--source-sha256",
        default="",
        help="optional expected official source digest recorded as provenance",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="stable FEN-hash sampling rate; scan the full source for an unbiased subset",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0, help="stop after this source row; 0 is all"
    )
    parser.add_argument(
        "--max-accepted", type=int, default=0, help="stop after this many cached rows; 0 is all"
    )
    parser.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="balance a bounded pilot over 3 material phases x 5 score bands",
    )
    parser.add_argument("--min-depth", type=int, default=24)
    parser.add_argument(
        "--quiet-only",
        action="store_true",
        help="retain positions not in check whose deepest PV1 starts with a quiet non-check",
    )
    parser.add_argument("--max-teacher-abs-cp", type=float, default=2_000.0)
    parser.add_argument("--residual-clip-cp", type=float, default=1_200.0)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=65_536,
        help="maximum flattened Parquet rows materialised at once",
    )
    parser.add_argument(
        "--checkpoint-rows",
        type=int,
        default=2_000_000,
        help="flush a partial shard after this many input rows",
    )
    parser.add_argument("--public-starts", type=Path, default=Path("games/public-starts.json"))
    parser.add_argument("--competition-games", type=Path, default=Path("games"))
    parser.add_argument("--seed", type=int, default=20_260_907)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.baseline_evaluator:
        spec = importlib.util.spec_from_file_location("cache_baseline", args.baseline_evaluator)
        if spec is None or spec.loader is None:
            raise ValueError("cannot load baseline evaluator")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        train_king_nnue._FAST_ENGINE = module
    build_cache(args)


if __name__ == "__main__":
    main()
