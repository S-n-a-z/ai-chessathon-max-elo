"""Build an isolated original neural candidate without changing the promoted engine."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

WRAPPER = """

@njit(cache=False)
def _evaluate(board: np.ndarray, side: int, white_king: int, black_king: int) -> int:
    classical = _evaluate_classical(board, side, white_king, black_king)
    if NN_MAX_PHASE < 24:
        phase = 0
        for rank in range(8):
            for file in range(8):
                phase += PHASE_VALUE[abs(int(board[rank * 16 + file]))]
        if phase > NN_MAX_PHASE:
            return classical
    correction = _neural_correction(board, side, white_king, black_king)
    return classical + int(round(correction * side * NN_BLEND))

"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blend", type=float, default=0.5)
    parser.add_argument("--numba-opt", type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--accumulator-cache", action="store_true")
    parser.add_argument(
        "--max-phase",
        type=int,
        choices=range(25),
        default=24,
        help="use the residual only up to this material phase (6 = endgames)",
    )
    args = parser.parse_args()
    if not 0 <= args.blend <= 1:
        parser.error("blend must be between zero and one")
    if args.out.resolve() == args.source.resolve():
        parser.error("candidate must be isolated from source")
    args.out.mkdir(parents=True, exist_ok=False)
    for path in args.source.glob("*.py"):
        shutil.copy2(path, args.out / path.name)
    shutil.copytree(args.source / "weights", args.out / "weights")
    shutil.copy2(args.network, args.out / "weights" / "king_nnue.npz")
    runtime = Path(__file__).with_name("neural_runtime.py")
    shutil.copy2(runtime, args.out / "neural.py")
    if args.accumulator_cache:
        shutil.copy2(runtime, args.out / "neural_full.py")
        shutil.copy2(Path(__file__).with_name("neural_cached_runtime.py"), args.out / "neural.py")
    engine = (args.source / "fast_engine.py").read_text(encoding="utf-8")
    if args.numba_opt is not None:
        engine = engine.replace(
            "import time\n",
            f"import os\nimport time\nos.environ['NUMBA_OPT'] = '{args.numba_opt}'\n",
            1,
        )
    if "def _evaluate_classical(" in engine:
        raise ValueError("source already has neural integration")
    engine = engine.replace("def _evaluate(", "def _evaluate_classical(", 1)
    marker = "@njit(cache=False)\ndef _is_repeated("
    if marker not in engine:
        raise ValueError("cannot locate evaluator wrapper insertion point")
    engine = engine.replace(marker, WRAPPER + marker, 1)
    engine = engine.replace(
        "from numba import njit, objmode",
        "from numba import njit, objmode\n"
        "from neural import correction as _neural_correction\n"
        f"NN_BLEND = {args.blend!r}\nNN_MAX_PHASE = {args.max_phase}",
    )
    if args.accumulator_cache:
        engine = engine.replace(
            "from neural import correction as _neural_correction",
            "from neural import BOARD_SIZE as NN_BOARD_SIZE\n"
            "from neural import correction as _neural_correction",
        )
        engine = engine.replace(
            "encoded = np.zeros(128, dtype=np.int8)",
            "encoded = np.zeros(NN_BOARD_SIZE, dtype=np.int32)",
        )
    (args.out / "fast_engine.py").write_text(engine, encoding="utf-8")
    print(f"Built {args.out}, blend={args.blend:g}, weights={args.network}", flush=True)


if __name__ == "__main__":
    main()
