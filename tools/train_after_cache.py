"""Wait for a complete training cache, then run one bounded original training job.

The small supervisor uses only the standard library while waiting, so it does not hold
PyTorch in memory during data preparation. All outputs remain outside the submission.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--min-positions", type=int, default=400_000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.min_positions < 1 or args.threads < 1:
        parser.error("minimum positions and threads must be positive")
    manifest_path = args.cache / "manifest.json"
    deadline = time.monotonic() + 6 * 3600
    print(f"Waiting for complete cache: {args.cache}", flush=True)
    while True:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("complete"):
                accepted = manifest["progress"]["accepted"]
                if accepted < args.min_positions:
                    raise RuntimeError(f"larger cache too small: {accepted}")
                print(f"Cache complete: {accepted:,} positions; starting training", flush=True)
                break
        if time.monotonic() >= deadline:
            raise TimeoutError("cache did not finish within six hours; inspect builder log")
        time.sleep(30)
    if args.out.exists():
        raise FileExistsError(f"preserving existing model: {args.out}")
    trainer = Path(__file__).with_name("train_king_nnue_cache.py")
    command = [
        sys.executable,
        str(trainer),
        "--cache",
        str(args.cache),
        "--baseline-evaluator",
        str(args.baseline),
        "--hidden",
        str(args.hidden),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        "1024",
        "--threads",
        str(args.threads),
        "--out",
        str(args.out),
    ]
    subprocess.run(command, check=True)
    print(f"Training complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
