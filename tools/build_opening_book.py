"""Build a shallow, rules-permitted opening book from curated opening positions."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.polyglot


def opening_positions(directory: Path, max_ply: int) -> dict[int, str]:
    positions: dict[int, str] = {}
    for source in sorted(directory.glob("?.tsv")):
        with source.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                game = chess.pgn.read_game(io.StringIO(row["pgn"]))
                if game is None:
                    continue
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= max_ply:
                        break
                    key = chess.polyglot.zobrist_hash(board)
                    positions.setdefault(key, board.fen())
                    board.push(move)
    return positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("weights/opening_book.json"))
    parser.add_argument("--max-ply", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=20_000)
    parser.add_argument("--labeler", default="Stockfish 19")
    arguments = parser.parse_args()

    positions = opening_positions(arguments.openings, arguments.max_ply)
    moves: dict[str, str] = {}
    engine = chess.engine.SimpleEngine.popen_uci(str(arguments.stockfish))
    try:
        engine.configure({"Threads": 1, "Hash": 128})
        for index, (key, fen) in enumerate(sorted(positions.items()), start=1):
            board = chess.Board(fen)
            result = engine.play(board, chess.engine.Limit(nodes=arguments.nodes))
            if result.move is not None:
                moves[f"{key:016x}"] = result.move.uci()
            if index % 250 == 0:
                print(f"labelled {index}/{len(positions)}")
    finally:
        engine.quit()

    payload = {
        "format": 1,
        "source": (
            "lichess-org/chess-openings (CC0), moves labelled by " + arguments.labeler
        ),
        "max_ply": arguments.max_ply,
        "nodes_per_position": arguments.nodes,
        "moves": moves,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    print(f"wrote {len(moves)} positions to {arguments.out}")


if __name__ == "__main__":
    main()
