"""Finish book assembly, then run sequential clock-controlled release comparisons."""

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.polyglot


def main() -> None:
    area = Path("games/revamp")
    book_path = area / "competition-book-v3.json"
    match_path = area / "release-v3-v2-unbooked/results.json"
    deadline = time.monotonic() + 3600
    while True:
        try:
            book = json.loads(book_path.read_text(encoding="utf-8"))
            match = json.loads(match_path.read_text(encoding="utf-8"))
            if book["competition_preparation"]["complete"] and len(match["games"]) == 16:
                break
        except (OSError, ValueError, KeyError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError("preparation did not complete within one hour")
        time.sleep(15)
    # Let the completed workers exit before measuring fresh imports and clocks.
    time.sleep(5)
    audit = json.loads(book_path.with_suffix(".audit.json").read_text(encoding="utf-8"))
    for row in audit:
        board = chess.Board(row["fen"])
        assert board.fullmove_number <= 20 and board.is_valid()
        assert row["key"] == f"{chess.polyglot.zobrist_hash(board):016x}"
        assert row["move"] == book["moves"][row["key"]]
        assert chess.Move.from_uci(row["move"]) in board.legal_moves
    release = area / "release-v3"
    shutil.copy2(book_path, release / "weights/opening_book.json")
    (release / "weights/OPENING_BOOK.md").write_text(
        "# Opening preparation\n\n"
        f"{len(book['moves']):,} unique opening positions. The original 5,474-position CC0 "
        "Lichess opening book is supplemented by 8,000 offline analyses with Stockfish18 "
        "at 50,000 nodes and two principal variations per position. Preparation starts "
        "from public Chessathon games through round100 and explores up to eight more plies, "
        "always at move20 or earlier. The runtime rejects book lookups after move20.\n\n"
        "The teacher executable, teacher network, analysis scores and source PGNs are not "
        "shipped. Only permitted opening moves are stored. File metadata records the "
        "teacher identity, public source hashes and preparation settings.\n\n"
        "Sources: https://github.com/lichess-org/chess-openings and public team game PGNs "
        "from https://aichessathon.com/\n",
        encoding="utf-8",
    )
    table_note = release / "weights/syzygy/README.md"
    table_note.write_text(
        table_note.read_text(encoding="utf-8")
        + "\nAdditional five-piece WDL/DTZ coverage: rook-and-pawn versus rook, "
        "queen-and-rook versus rook, bishop-and-rook versus rook, knight-and-rook versus rook. "
        "KRRvKR has WDL only as a promotion dependency; that root class uses search. "
        "All necessary WDL capture/promotion dependencies are present. Sizes, source URLs "
        "and SHA256 hashes are recorded in ROOK_TABLES.json.\n",
        encoding="utf-8",
    )
    print(
        f"Audited {len(audit)} prepared positions; assembled {len(book['moves'])}-position book",
        flush=True,
    )
    base = [
        sys.executable,
        "tools/revamp_match.py",
        "--candidate",
        str(release),
        "--baseline",
        str(area / "submitted-v2"),
    ]
    trials = [
        ("release-v3-v2-timed", ["--pairs", "12", "--base-ms", "10000", "--increment-ms", "100"]),
        (
            "release-v3-v2-full",
            ["--pairs", "1", "--fresh", "--base-ms", "120000", "--increment-ms", "500"],
        ),
    ]
    for name, arguments in trials:
        print(f"Starting {name}", flush=True)
        with (area / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                base + arguments + ["--out", str(area / name)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"Completed {name}", flush=True)
    print("All final trial jobs complete; inspect results before promotion", flush=True)


if __name__ == "__main__":
    main()
