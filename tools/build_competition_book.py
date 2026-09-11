"""Offline, bounded opening preparation from public competition PGNs.

Only positions at move 20 or earlier enter the book. The teacher executable is
offline equipment; it and all analysis scores remain outside the submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import chess.polyglot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, nargs="+", required=True)
    parser.add_argument("--base-book", type=Path, default=Path("weights/opening_book.json"))
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=50_000)
    parser.add_argument("--max-positions", type=int, default=8_000)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--extension-plies", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"preserving existing book: {args.out}")
    if min(args.nodes, args.max_positions, args.branches) < 1 or args.extension_plies < 0:
        parser.error("invalid preparation budget")
    payload = json.loads(args.base_book.read_text(encoding="utf-8"))
    seeds: dict[int, chess.Board] = {}
    source_hashes: dict[str, str] = {}
    for directory in args.games:
        for path in sorted(directory.glob("round-*.pgn")):
            source_hashes[path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding="utf-8-sig") as stream:
                game = chess.pgn.read_game(stream)
            if game is None or game.errors:
                continue
            board = game.board()
            for move in game.mainline_moves():
                if board.fullmove_number > 20:
                    break
                if board.is_valid() and not board.is_game_over(claim_draw=True):
                    seeds.setdefault(chess.polyglot.zobrist_hash(board), board.copy(stack=False))
                board.push(move)
    if not seeds:
        raise ValueError("no valid public opening positions")
    queue = deque((board, 0) for board in seeds.values())
    scheduled = set(seeds)
    labelled: dict[str, str] = {}
    audit: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    identity = dict(engine.id)

    def save(complete: bool) -> None:
        output = dict(payload)
        output["moves"] = {**payload["moves"], **labelled}
        output["competition_preparation"] = {
            "public_source_sha256": source_hashes,
            "teacher": identity,
            "teacher_executable_sha256": teacher_hash,
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "maximum_fullmove": 20,
            "nodes_per_position": args.nodes,
            "branches": args.branches,
            "extension_plies": args.extension_plies,
            "public_seed_positions": len(seeds),
            "labelled_positions": len(labelled),
            "complete": complete,
        }
        temporary = args.out.with_suffix(".partial.json")
        temporary.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
        temporary.replace(args.out)
        args.out.with_suffix(".audit.json").write_text(json.dumps(audit), encoding="utf-8")

    teacher_hash = hashlib.sha256(args.stockfish.read_bytes()).hexdigest()
    print(f"Preparing {len(seeds)} public seed positions with {identity}", flush=True)
    try:
        engine.configure({"Threads": 1, "Hash": 64})
        while queue and len(labelled) < args.max_positions:
            board, distance = queue.popleft()
            if board.fullmove_number > 20 or board.is_game_over(claim_draw=True):
                continue
            lines = engine.analyse(
                board,
                chess.engine.Limit(nodes=args.nodes),
                multipv=min(args.branches, board.legal_moves.count()),
            )
            if not isinstance(lines, list):
                lines = [lines]
            principal = [line["pv"][0] for line in lines if line.get("pv")]
            if not principal:
                continue
            key = f"{chess.polyglot.zobrist_hash(board):016x}"
            assert principal[0] in board.legal_moves and board.fullmove_number <= 20
            labelled[key] = principal[0].uci()
            audit.append({"key": key, "fen": board.fen(), "move": principal[0].uci()})
            if distance < args.extension_plies:
                for move in principal:
                    child = board.copy(stack=False)
                    child.push(move)
                    child_key = chess.polyglot.zobrist_hash(child)
                    if child.fullmove_number <= 20 and child_key not in scheduled:
                        scheduled.add(child_key)
                        queue.append((child, distance + 1))
            if len(labelled) % 250 == 0:
                save(False)
                print(f"Prepared {len(labelled)} positions; {len(queue)} queued", flush=True)
        save(True)
    finally:
        engine.quit()
    print(
        f"Complete: {len(labelled)} prepared; {len(set(payload['moves']) | set(labelled))} total",
        flush=True,
    )


if __name__ == "__main__":
    main()
