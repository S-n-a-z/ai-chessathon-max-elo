"""Run deterministic fixed-depth probes against the compiled search.

This is deliberately separate from the game harness: it makes search-heuristic changes easy to
compare by score, selected move, node count, and speed without clock noise.
"""

from __future__ import annotations

import argparse
import random
import time

import chess

import fast_engine


def probe_positions() -> list[chess.Board]:
    rng = random.Random(20260905)
    boards = [chess.Board()]
    board = chess.Board()
    targets = {8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52}
    for ply in range(1, max(targets) + 1):
        moves = list(board.legal_moves)
        if not moves:
            board.reset()
            moves = list(board.legal_moves)
        board.push(rng.choice(moves))
        if ply in targets and not board.is_game_over():
            boards.append(board.copy(stack=False))
    return boards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=9)
    arguments = parser.parse_args()

    total_nodes = 0
    total_seconds = 0.0
    for index, board in enumerate(probe_positions()):
        fast_engine.reset_for_test()
        encoded = fast_engine._encode_position(board)
        started = time.perf_counter()
        score, move, nodes, completed = fast_engine._root_search(
            *encoded,
            arguments.depth,
            -fast_engine.INF,
            fast_engine.INF,
            0,
            100_000_000,
            index + 1,
            fast_engine._tt_keys,
            fast_engine._tt_scores,
            fast_engine._tt_moves,
            fast_engine._tt_depths,
            fast_engine._tt_bounds,
            fast_engine._tt_ages,
            fast_engine._move_buffer,
            fast_engine._score_buffer,
            fast_engine._killers,
            fast_engine._history,
            fast_engine._hash_stack,
            fast_engine._nodes,
        )
        elapsed = time.perf_counter() - started
        total_nodes += nodes
        total_seconds += elapsed
        uci = fast_engine._decode_move(move).uci()
        print(index, score, uci, nodes, f"{elapsed:.4f}", completed)
    print("total", total_nodes, f"{total_seconds:.4f}", f"{total_nodes / total_seconds:.0f}")


if __name__ == "__main__":
    main()
