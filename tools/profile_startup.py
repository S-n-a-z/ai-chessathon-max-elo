"""Measure cold JIT startup, compiler hotspots and deterministic search speed."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark_search import _load_engine, _search, _starts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    engine = _load_engine(args.candidate)
    elapsed = time.perf_counter() - started
    print(f"Cold import: {elapsed:.2f}s", flush=True)
    compiles = []
    for name, function in vars(engine).items():
        if hasattr(function, "get_metadata"):
            for signature, metadata in function.get_metadata().items():
                seconds = sum(
                    entry.run
                    for pipeline in metadata.get("pipeline_times", {}).values()
                    for entry in pipeline.values()
                )
                compiles.append({"name": name, "signature": str(signature), "seconds": seconds})
    results = []
    for row in _starts(Path("games/revamp/unbooked-starts.json"), 6):
        results.append({"fen": row["fen"], **_search(engine, row["fen"], 6)})
    report = {
        "candidate": str(args.candidate.resolve()), "import_seconds": elapsed,
        "compiler_hotspots_inclusive": sorted(compiles, key=lambda x: -x["seconds"]),
        "results": results,
        "nps": sum(row["nodes"] for row in results) / sum(row["seconds"] for row in results),
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Search: {report['nps']:.0f} nodes/s; report {args.out}", flush=True)


if __name__ == "__main__":
    main()
