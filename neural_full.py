# mypy: ignore-errors
"""Original quantized king-conditioned neural residual, trained from scratch.

This source is copied to neural.py only in experimental or selected neural builds.
Inference uses NumPy/Numba and bundled weights, with no training packages or downloads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(
        Path(__file__).resolve().parent / "weights/king_nnue.npz", allow_pickle=False
    ) as archive:
        weights = np.ascontiguousarray(archive["feature_weights"], dtype=np.int16)
        bias = np.ascontiguousarray(archive["feature_bias"], dtype=np.int32)
        clip = np.ascontiguousarray(archive["activation_clip"], dtype=np.int32)
        scale = float(archive["target_scale_cp"])
        head = np.ascontiguousarray(
            archive["output_weights"].astype(np.float32)
            * archive["output_scale"]
            * archive["feature_scale"]
            * scale
        )
        tempo = float(archive["tempo"]) * scale
    hidden = bias.size
    if weights.shape != (24576, hidden) or clip.shape != (hidden,) or head.shape != (hidden,):
        raise ValueError("incompatible neural weight shapes")
    if not np.all(np.isfinite(head)) or not np.isfinite(tempo) or np.any(clip <= 0):
        raise ValueError("invalid neural weight values")
    return weights, bias, clip, head, tempo


WEIGHTS, BIAS, CLIP, HEAD, TEMPO = _load()
HIDDEN = int(BIAS.size)


@njit(cache=False)
def correction(board: np.ndarray, side: int, white_king: int, black_king: int) -> float:
    """Return the white-centric correction, sharing weights across both perspectives."""
    white = BIAS.copy()
    black = BIAS.copy()
    white_flip = (white_king & 7) >= 4
    black_flip = (black_king & 7) >= 4
    white_file = (white_king & 7) ^ (7 if white_flip else 0)
    black_file = (black_king & 7) ^ (7 if black_flip else 0)
    white_base = ((white_king >> 4) * 4 + white_file) * 768
    black_base = ((7 - (black_king >> 4)) * 4 + black_file) * 768
    for rank in range(8):
        for file in range(8):
            piece = int(board[rank * 16 + file])
            if piece == 0:
                continue
            kind = abs(piece) - 1
            white_feature = white_base + (kind + (0 if piece > 0 else 6)) * 64
            white_feature += rank * 8 + (file ^ (7 if white_flip else 0))
            black_feature = black_base + (kind + (0 if piece < 0 else 6)) * 64
            black_feature += (7 - rank) * 8 + (file ^ (7 if black_flip else 0))
            for unit in range(HIDDEN):
                white[unit] += WEIGHTS[white_feature, unit]
                black[unit] += WEIGHTS[black_feature, unit]
    value = TEMPO * side
    for unit in range(HIDDEN):
        value += (
            min(max(white[unit], 0), CLIP[unit]) - min(max(black[unit], 0), CLIP[unit])
        ) * HEAD[unit]
    return value
