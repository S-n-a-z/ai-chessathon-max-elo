"""Run the project regression suite against an isolated runtime directory."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.candidate.resolve()))
    started = time.perf_counter()
    agent = importlib.import_module("agent")
    import_seconds = time.perf_counter() - started
    compiled = agent._fast_choose_move is not None
    print(f"Imported candidate in {import_seconds:.2f}s; compiled search: {compiled}", flush=True)
    suite = unittest.defaultTestLoader.discover(str(REPO / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "candidate": str(args.candidate.resolve()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            package: importlib.metadata.version(package) for package in ("numpy", "numba", "chess")
        },
        "import_seconds": import_seconds,
        "local_90_second_import_passed": import_seconds <= 90,
        "compiled_search": compiled,
        "tests_run": result.testsRun,
        "tests_passed": result.wasSuccessful(),
        "failures": [str(test) for test, _ in result.failures],
        "errors": [str(test) for test, _ in result.errors],
        "limitation": (
            "Local OS and CPU differ from platform; upload validation remains authoritative"
        ),
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not compiled or not result.wasSuccessful() or import_seconds > 90:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
