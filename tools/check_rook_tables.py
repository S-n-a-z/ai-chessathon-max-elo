"""Check partial-table fallback and WDL preservation for random rook endings."""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import chess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.candidate.resolve()))
    endgame = importlib.import_module("endgame")
    tablebase = endgame.TABLEBASE
    assert tablebase is not None
    rng = random.Random(20260910)
    checked = 0
    results = {"winning": 0, "drawing": 0, "losing": 0}
    while checked < 500:
        board = chess.Board(None)
        squares = rng.sample(range(64), 5)
        for square, piece in zip(squares, "KRPrk", strict=True):
            board.set_piece_at(square, chess.Piece.from_symbol(piece))
        board.turn = rng.choice(chess.COLORS)
        if not board.is_valid() or board.is_game_over():
            continue
        root_wdl = tablebase.get_wdl(board)
        assert root_wdl is not None
        uci = endgame.choose_tablebase_move(board.fen())
        assert uci is not None, board.fen()
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)
        child_wdl = tablebase.get_wdl(board)
        assert child_wdl is not None
        assert -child_wdl >= root_wdl, (board.fen(), root_wdl, child_wdl)
        checked += 1
        results["winning" if root_wdl > 0 else "losing" if root_wdl < 0 else "drawing"] += 1
    # WDL-only underpromotion dependency must search when encountered as a root.
    board = chess.Board("7k/6r1/8/8/8/8/1RR5/K7 w - - 0 1")
    assert board.is_valid()
    assert tablebase.get_wdl(board) is not None
    assert tablebase.get_dtz(board) is None
    assert endgame.choose_tablebase_move(board.fen()) is None
    report = {"random_positions": checked, "classes": results, "missing_dtz_fallback": True}
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
