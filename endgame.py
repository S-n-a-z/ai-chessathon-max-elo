"""Use permitted WDL/DTZ tables in covered four- and five-piece endings."""

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
    if len(board.piece_map()) > 5 or board.castling_rights:
        return None
    # Extra WDL files cover promotion dependencies. Only use this move selector
    # when the root also has DTZ coverage; other material classes use search.
    if TABLEBASE.get_dtz(board) is None:
        return None

    best_move: chess.Move | None = None
    best_rank: tuple[int, int, int] | None = None
    for move in board.legal_moves:
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            child_wdl = TABLEBASE.get_wdl(board)
            child_dtz = TABLEBASE.get_dtz(board)
            mate = board.is_checkmate()
        finally:
            board.pop()
        # A partly covered set of alternatives cannot establish the best move.
        if child_wdl is None or (child_dtz is None and not zeroing):
            return None

        wdl = -child_wdl
        distance = 1 if zeroing else 1 + abs(child_dtz or 0)
        if mate:
            return move.uci()
        # WDL assumes a reset halfmove clock. Reserve a ply for DTZ rounding
        # when a nominal win would reach the current fifty-move boundary.
        if wdl == 2 and not zeroing and board.halfmove_clock + distance >= 100:
            wdl = 1
        if wdl > 0:
            distance_rank = -distance  # Convert or mate quickly when winning.
        elif wdl < 0:
            distance_rank = distance  # Maximise resistance when the loss is forced.
        else:
            distance_rank = int(zeroing)  # Reset the fifty-move clock in drawn endings.
        rank = (wdl, distance_rank, int(zeroing))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_move = move

    return best_move.uci() if best_move is not None else None
