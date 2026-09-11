"""Package a tested release and audit the exact submission archive."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harness.package import build, members  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.candidate.resolve()
    if args.out.exists():
        raise FileExistsError(f"preserving existing archive: {args.out}")
    entries = list(members(root, ("weights",)))
    total = sum(path.stat().st_size for path, _ in entries)
    if total > 50_000_000:
        raise ValueError(f"unzipped data exceeds competition limit: {total}")
    allowed = {".py", ".npz", ".rtbw", ".rtbz", ".md", ".json"}
    for path, name in entries:
        if path.suffix not in allowed or path.is_symlink():
            raise ValueError(f"unexpected release member: {name}")
    book = json.loads((root / "weights/opening_book.json").read_text(encoding="utf-8"))
    assert book["competition_preparation"]["complete"]
    assert book["competition_preparation"]["maximum_fullmove"] == 20
    network = root / "weights/king_nnue.npz"
    assert hashlib.sha256(network.read_bytes()).hexdigest() == (
        "789721e8e5a5618c74a2ecb7a150bb9d5b066f3beb979b6f00c6faa39a92ea62"
    )
    built = build(root, args.out, ("weights",))
    with zipfile.ZipFile(args.out) as archive:
        names = archive.namelist()
        assert "agent.py" in names and len(names) == len(set(names))
        assert all("\\" not in name and ".." not in Path(name).parts for name in names)
        assert archive.testzip() is None
        assert sum(info.file_size for info in archive.infolist()) == total
        for path, name in entries:
            assert archive.read(name.replace("\\", "/")) == path.read_bytes()
    report = {
        "archive": str(args.out.resolve()),
        "candidate": str(root),
        "zip_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "compressed_bytes": args.out.stat().st_size,
        "uncompressed_bytes": total,
        "files": len(built),
        "book_positions": len(book["moves"]),
        "network_sha256": hashlib.sha256(network.read_bytes()).hexdigest(),
        "checks": "root agent.py; <=50MB; no native executable; CRC/readback and source hashes",
        "file_sha256": {
            name.replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path, name in entries
        },
    }
    args.out.with_suffix(".audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in report.items() if key != "file_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
