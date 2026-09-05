"""Read the shallow, rules-permitted opening book shipped with the agent."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.polyglot

_BOOK_PATH = Path(__file__).resolve().parent / "weights" / "opening_book.json"

try:
    _payload = json.loads(_BOOK_PATH.read_text(encoding="utf-8"))
    BOOK: dict[int, str] = {int(key, 16): move for key, move in _payload["moves"].items()}
except (OSError, ValueError, KeyError, TypeError):
    BOOK = {}


def choose_opening_move(fen: str) -> str | None:
    """Return a legal book move for an exact opening position, if one is present."""
    board = chess.Board(fen)
    uci = BOOK.get(chess.polyglot.zobrist_hash(board))
    if uci is None:
        return None
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    return uci if move in board.legal_moves else None
