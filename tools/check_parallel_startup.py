"""Repeat genuine fresh-process startup gates on two separate local CPU cores."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import chess

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.sandbox import local  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mask", type=int)
    args = parser.parse_args()
    if args.mask is None:
        args.out.mkdir(parents=True, exist_ok=False)
        environment = dict(os.environ)
        thread_variables = (
            "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"
        )
        for name in thread_variables:
            environment[name] = "1"
        processes = []
        logs = []
        for mask in (1, 4):
            stream = (args.out / f"core-{mask}.log").open("w", encoding="utf-8")
            logs.append(stream)
            processes.append(subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--candidate", str(args.candidate), "--out", str(args.out), "--mask", str(mask)],
                cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            ))
        codes = [process.wait() for process in processes]
        for stream in logs:
            stream.close()
        reports = [json.loads((args.out / f"core-{mask}.json").read_text()) for mask in (1, 4)]
        results = [item for report in reports for item in report]
        summary = {
            "candidate": str(args.candidate.resolve()), "exit_codes": codes,
            "passed": codes == [0, 0] and len(results) == 6 and all(x["passed"] for x in results),
            "attempts": results,
        }
        (args.out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        raise SystemExit(0 if summary["passed"] else 1)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), args.mask):
        raise ctypes.WinError(ctypes.get_last_error())
    starts = json.loads((REPO / "games/revamp/unbooked-starts.json").read_text())["starts"]
    results = []
    for attempt in range(3):
        runner = local(args.candidate)
        started = time.perf_counter()
        report = {"mask": args.mask, "attempt": attempt + 1, "passed": False}
        try:
            runner.start(90)
            report["init_seconds"] = time.perf_counter() - started
            fen = starts[attempt]["fen"]
            before = time.perf_counter()
            move = runner.move(fen, 1000)
            duration = time.perf_counter() - before
            report["move"] = move
            report["move_seconds"] = duration
            assert duration < 1.0, "first-move flag"
            assert chess.Move.from_uci(move) in chess.Board(fen).legal_moves
            report["passed"] = True
        except Exception as error:
            report["error"] = repr(error)
            report["elapsed_seconds"] = time.perf_counter() - started
        finally:
            runner.stop()
            (args.out / f"core-{args.mask}-attempt-{attempt + 1}.stderr.log").write_text(
                runner.stderr_tail, encoding="utf-8"
            )
        if "Compiled search initialization failed:" in runner.stderr_tail:
            report["passed"] = False
            report["error"] = "compiled search failed; fallback is not accepted"
        results.append(report)
        (args.out / f"core-{args.mask}.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        print(json.dumps(report), flush=True)
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)


if __name__ == "__main__":
    main()
