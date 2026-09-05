"""Perfect three- and four-piece endgame play from the permitted Syzygy tables."""

from __future__ import annotations

from pathlib import Path

import chess
import chess.syzygy

_TABLE_DIRECTORY = Path(__file__).resolve().parent / "weights" / "syzygy"

try:
    TABLEBASE: chess.syzygy.Tablebase | None = chess.syzygy.open_tablebase(
        str(_TABLE_DIRECTORY), max_fds=32
    )
except (OSError, ValueError):
    TABLEBASE = None


def choose_tablebase_move(fen: str) -> str | None:
    """Return a WDL-optimal move in covered endings, or ``None`` outside the tables."""
    if TABLEBASE is None:
        return None
    board = chess.Board(fen)
    if len(board.piece_map()) > 4 or board.castling_rights:
        return None

    best_move: chess.Move | None = None
    best_rank: tuple[int, int, int] | None = None
    for move in board.legal_moves:
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            child_wdl = TABLEBASE.get_wdl(board)
            child_dtz = TABLEBASE.get_dtz(board)
        finally:
            board.pop()
        if child_wdl is None:
            continue

        wdl = -child_wdl
        dtz = -child_dtz if child_dtz is not None else 0
        if wdl > 0:
            distance_rank = -abs(dtz)  # Convert or mate quickly when winning.
        elif wdl < 0:
            distance_rank = abs(dtz)  # Maximise resistance when the loss is forced.
        else:
            distance_rank = int(zeroing)  # Reset the fifty-move clock in drawn endings.
        rank = (wdl, distance_rank, int(zeroing))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_move = move

    return best_move.uci() if best_move is not None else None
