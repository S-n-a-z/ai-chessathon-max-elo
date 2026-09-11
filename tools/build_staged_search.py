"""Create an isolated lazy-legality experiment from this project's original search."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = (args.source / "fast_engine.py").read_text(encoding="utf-8")
    begin = source.index("@njit(cache=False)\ndef _search(")
    end = source.index("@njit(cache=False)\ndef _root_search(", begin)
    search = source[begin:end]
    before = search
    search = search.replace(
        "count = _generate_legal(\n"
        "        board, side, castling, ep_square, halfmove, king_square, ply, move_buffer\n"
        "    )\n    if count == 0:",
        "count = _generate_pseudo(\n"
        "        board, side, castling, ep_square, king_square, move_buffer[ply]\n"
        "    )\n    if count == 0 or not _has_legal_move(\n"
        "        board, side, castling, ep_square, halfmove, king_square, move_buffer[ply]\n"
        "    ):",
        1,
    )
    search = search.replace(
        "count = _generate_legal(\n"
        "            board, side, castling, ep_square, halfmove, king_square, ply, move_buffer\n"
        "        )",
        "count = _generate_pseudo(\n"
        "            board, side, castling, ep_square, king_square, move_buffer[ply]\n"
        "        )",
        1,
    )
    search = search.replace(
        "    for move_index in range(count):\n        move = int(move_buffer[ply, move_index])",
        "    legal_index = 0\n"
        "    for candidate_index in range(count):\n"
        "        move = int(move_buffer[ply, candidate_index])",
        1,
    )
    search = search.replace(
        "        hash_stack[ply + 1] = _hash_after_move(",
        "        # Test legality when a candidate is actually searched. Cutoffs often\n"
        "        # leave most quiet moves untouched. Count only legal moves for PVS/LMR.\n"
        "        if _is_attacked(board, new_king, -side):\n"
        "            _undo_move(board, move, side, captured)\n"
        "            continue\n"
        "        move_index = legal_index\n"
        "        legal_index += 1\n"
        "        hash_stack[ply + 1] = _hash_after_move(",
        1,
    )
    if search == before or "_generate_legal(" in search or "legal_index += 1" not in search:
        raise ValueError("source layout changed; review the staged search transformation")
    args.out.mkdir(parents=True, exist_ok=False)
    for path in args.source.glob("*.py"):
        shutil.copy2(path, args.out / path.name)
    shutil.copytree(args.source / "weights", args.out / "weights")
    (args.out / "fast_engine.py").write_text(
        source[:begin] + search + source[end:], encoding="utf-8"
    )
    print(f"Built staged search experiment at {args.out}", flush=True)


if __name__ == "__main__":
    main()
