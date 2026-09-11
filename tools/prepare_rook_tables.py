"""Download a bounded, dependency-complete WDL set for rook-and-pawn vs rook."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import chess
import chess.syzygy


def main() -> None:
    root = Path("games/revamp/rook-tables")
    root.mkdir(exist_ok=True)
    report = []
    names = ("KRPvKR", "KQRvKR", "KRRvKR", "KRBvKR", "KRNvKR")
    for name in names:
        for kind, extension in (("wdl", "rtbw"), ("dtz", "rtbz")):
            # KRRvKR's large DTZ file does not fit the size budget. Its WDL file
            # is needed to resolve rook underpromotions; that root class searches.
            if name == "KRRvKR" and kind == "dtz":
                continue
            filename = f"{name}.{extension}"
            index = Path(f"games/revamp/syzygy-{kind}-index.html").read_text()
            match = re.search(r'href="' + re.escape(filename) + r'"[^\n]+?\s(\d+)\r?$', index, re.M)
            if match is None:
                raise ValueError(f"file absent from public mirror index: {filename}")
            size = int(match.group(1))
            url = f"https://tablebase.lichess.ovh/tables/standard/3-4-5-{kind}/{filename}"
            path = root / filename
            if not path.exists():
                partial = path.with_suffix(path.suffix + ".partial")
                with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                partial.replace(path)
            data = path.read_bytes()
            magic = chess.Board.tbw_magic if kind == "wdl" else chess.Board.tbz_magic
            if len(data) != size or data[:4] != magic:
                raise ValueError(f"download size/header mismatch: {path}")
            report.append(
                {
                    "file": filename,
                    "bytes": size,
                    "url": url,
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            print(f"Verified {filename}: {size:,} bytes", flush=True)
    missing = [
        name
        for name in chess.syzygy.all_dependencies(["KRPvKR"])
        if name not in {"KvK", "KBvK", "KNvK"}
        and not (root / f"{name}.rtbw").exists()
        and not (Path("weights/syzygy") / f"{name}.rtbw").exists()
    ]
    if missing:
        raise ValueError(f"missing WDL dependencies: {missing}")
    (root / "ROOK_TABLES.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Complete: {sum(item['bytes'] for item in report):,} bytes", flush=True)


if __name__ == "__main__":
    main()
