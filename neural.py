# mypy: ignore-errors
"""Original lazy accumulator cache for the compact king-conditioned evaluator.

The first 128 cells of the board retain the engine's mailbox. Scratch cells after them
hold the last evaluated position and an accumulator for each perspective/king square.
Every evaluation compares the current board to that cache, so make/unmake needs no neural
undo stack. Castling, captures, en passant and promotions are ordinary changed squares.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from neural_full import BIAS, CLIP, HEAD, HIDDEN, TEMPO, WEIGHTS

CACHE_ENTRY = 1 + 64 + HIDDEN
BOARD_SIZE = 128 + 128 * CACHE_ENTRY


@njit(cache=False)
def _accumulate(board: np.ndarray, king: int, perspective: int) -> int:
    king_file = king & 7
    king_rank = king >> 4
    # Keep separate caches for mirrored physical king squares; their feature orientations differ.
    cache_number = king_rank * 8 + king_file + (64 if perspective < 0 else 0)
    start = 128 + cache_number * CACHE_ENTRY
    accumulator = start + 65
    if board[start] == 0:
        for unit in range(HIDDEN):
            board[accumulator + unit] = BIAS[unit]
        board[start] = 1
    flip = 7 if king_file >= 4 else 0
    oriented_rank = king_rank if perspective > 0 else 7 - king_rank
    bucket = (oriented_rank * 4 + (king_file ^ flip)) * 768
    for rank in range(8):
        oriented_rank = rank if perspective > 0 else 7 - rank
        for file in range(8):
            current = int(board[rank * 16 + file])
            previous_slot = start + 1 + rank * 8 + file
            previous = int(board[previous_slot])
            if current == previous:
                continue
            square = oriented_rank * 8 + (file ^ flip)
            if previous:
                plane = abs(previous) - 1 + (0 if previous * perspective > 0 else 6)
                feature = bucket + plane * 64 + square
                for unit in range(HIDDEN):
                    board[accumulator + unit] -= WEIGHTS[feature, unit]
            if current:
                plane = abs(current) - 1 + (0 if current * perspective > 0 else 6)
                feature = bucket + plane * 64 + square
                for unit in range(HIDDEN):
                    board[accumulator + unit] += WEIGHTS[feature, unit]
            board[previous_slot] = current
    return accumulator


@njit(cache=False)
def correction(board: np.ndarray, side: int, white_king: int, black_king: int) -> float:
    """Return exactly the same residual as full recomputation, reusing accumulators."""
    white = _accumulate(board, white_king, 1)
    black = _accumulate(board, black_king, -1)
    result = TEMPO * side
    for unit in range(HIDDEN):
        result += (
            min(max(board[white + unit], 0), CLIP[unit])
            - min(max(board[black + unit], 0), CLIP[unit])
        ) * HEAD[unit]
    return result
