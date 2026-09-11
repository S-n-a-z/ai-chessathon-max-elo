"""Run two isolated full-clock match lanes and save incremental combined results.

Local Windows test orchestration only; never included in a submission. Each lane inherits
one logical processor on a distinct physical performance core. The existing fresh-process
match tool and platform harness enforce startup and move deadlines without modification.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import runpy
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]


def save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def lane(directory: Path, number: int) -> None:
    config = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    mask = config["lane_cpu_masks"][number]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())
    print(f"Lane {number}: affinity mask {mask}; fresh 120+0.5 games", flush=True)
    sys.argv = [
        "revamp_match.py", "--candidate", config["candidate"],
        "--baseline", config["baseline"], "--starts", str(directory / f"starts-{number}.json"),
        "--pairs", "6", "--fresh", "--base-ms", "120000", "--increment-ms", "500",
        "--init-seconds", "90", "--out", str(directory / f"lane-{number}"),
    ]
    runpy.run_path(str(REPO / "tools/revamp_match.py"), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=REPO / "games/revamp/release-v3")
    parser.add_argument("--baseline", type=Path, default=REPO / "games/revamp/submitted-v2")
    parser.add_argument("--lane", type=int, choices=(0, 1))
    args = parser.parse_args()
    directory = args.out.resolve()
    if args.lane is not None:
        lane(directory, args.lane)
        return
    directory.mkdir(parents=True, exist_ok=False)
    public = json.loads((REPO / "games/public-starts.json").read_text())["starts"]
    unbooked = json.loads((REPO / "games/revamp/unbooked-starts.json").read_text())["starts"]
    # Skip the opening already tested at full clock and remove duplicate FENs.
    seen = {public[0]["fen"]}
    selected = []
    for row in public[1:]:
        if row["fen"] not in seen:
            selected.append({**row, "cohort": "opening"})
            seen.add(row["fen"])
        if len(selected) == 8:
            break
    assert len(selected) == 8 and len(unbooked) >= 4
    selected.extend({**row, "cohort": "unbooked"} for row in unbooked[:4])
    for number in range(2):
        # Each lane has four opening pairs and two unbooked pairs, interleaved.
        rows = selected[number::2]
        rows = [rows[0], rows[4], rows[1], rows[2], rows[5], rows[3]]
        save(directory / f"starts-{number}.json", {"starts": rows})
    config = {
        "started_utc": datetime.now(UTC).isoformat(),
        "candidate": str(args.candidate.resolve()),
        "baseline": str(args.baseline.resolve()),
        "planned_games": 24,
        "lane_cpu_masks": [1, 4],
        "topology": "i3-1215U physical P-core masks 3 and 12; one sibling per core",
        "clock": "120 seconds + 0.5 seconds per move; fresh process; 90-second import",
        "limitations": (
            "Local Windows laptop, shared RAM/thermal budget; public starts, not held out"
        ),
    }
    save(directory / "manifest.json", config)
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[key] = "1"
    processes = []
    streams = []
    for number in range(2):
        stream = (directory / f"lane-{number}.log").open("w", encoding="utf-8")
        streams.append(stream)
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--out", str(directory),
             "--lane", str(number)],
            cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT,
        )
        processes.append(process)
    config["lane_pids"] = [process.pid for process in processes]
    save(directory / "manifest.json", config)
    previous = None
    try:
        while True:
            games = []
            for number in range(2):
                try:
                    report = json.loads(
                        (directory / f"lane-{number}/results.json").read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    continue
                for game in report["games"]:
                    cohort = "unbooked" if int(game["fen"].split()[-1]) > 20 else "opening"
                    games.append({**game, "lane": number, "cohort": cohort})
            codes = [process.poll() for process in processes]
            done = all(code is not None for code in codes)
            summary = {}
            for cohort in ("all", "opening", "unbooked"):
                points = [game["candidate_points"] for game in games
                          if cohort == "all" or game["cohort"] == cohort]
                summary[cohort] = {
                    "games": len(points), "wins": points.count(1.0),
                    "draws": points.count(0.5), "losses": points.count(0.0),
                    "score": sum(points) / len(points) if points else None,
                }
            state = "complete" if done and codes == [0, 0] and len(games) == 24 else (
                "failed" if done else "running"
            )
            save(directory / "results.json", {
                "state": state, "updated_utc": datetime.now(UTC).isoformat(),
                "planned_games": 24, "lane_exit_codes": codes,
                "summary": summary, "games": games,
            })
            progress = (len(games), tuple(codes))
            if progress != previous:
                print(f"{state}: {summary['all']}; lane exit codes {codes}", flush=True)
                previous = progress
            if done:
                break
            time.sleep(10)
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
