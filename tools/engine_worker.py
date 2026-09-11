"""Persistent local worker used by the A/B match tool.

Competition containers restart between games. This worker avoids paying Numba's local compile
cost for every test game and explicitly resets all game-persistent search state between games.
It is testing infrastructure only and is not included in the submission archive.
"""

from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import chess


def _fixed_search_move(
    agent: object,
    fast_engine: object,
    fen: str,
    depth: int,
    total_nodes: int,
) -> str:
    tablebase_move = agent.choose_tablebase_move(fen)
    if tablebase_move is not None:
        return tablebase_move
    opening_move = agent.choose_opening_move(fen)
    if opening_move is not None:
        return opening_move

    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    encoded = fast_engine._encode_position(board)
    fast_engine._generation = (fast_engine._generation + 1) & 0xFFFF
    if fast_engine._generation == 0:
        fast_engine._generation = 1
        fast_engine._tt_ages.fill(0)

    preferred = 0
    remaining_nodes = total_nodes
    maximum_depth = depth if depth else 63
    for current_depth in range(1, maximum_depth + 1):
        iteration_limit = 200_000_000 if not total_nodes else remaining_nodes
        if iteration_limit < 2_000:
            break
        score, move, searched_nodes, completed = fast_engine._root_search(
            *encoded,
            current_depth,
            -fast_engine.INF,
            fast_engine.INF,
            preferred,
            iteration_limit,
            fast_engine._generation,
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
        del score
        if total_nodes:
            remaining_nodes -= searched_nodes
        if not completed:
            break
        preferred = move

    candidate = fast_engine._decode_move(preferred) if preferred else legal[0]
    if candidate not in board.legal_moves:
        candidate = legal[0]
    if getattr(fast_engine, "USE_ROOT_POSTPROCESS", True):
        candidate = fast_engine._postprocess_candidate(board, candidate)
    return candidate.uci()


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root))
    agent = import_module("agent")
    fast_engine = import_module("fast_engine")

    print(
        json.dumps({"ready": True, "compiled_search": agent._fast_choose_move is not None}),
        flush=True,
    )
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("reset"):
            fast_engine.reset_for_test()
            print(json.dumps({"reset": True}), flush=True)
            continue
        fixed_depth = int(request.get("fixed_depth", 0))
        fixed_nodes = int(request.get("fixed_nodes", 0))
        if fixed_depth or fixed_nodes:
            move = _fixed_search_move(agent, fast_engine, request["fen"], fixed_depth, fixed_nodes)
        else:
            move = agent.get_move(request["fen"], int(request["time_left_ms"]))
        print(json.dumps({"move": move}), flush=True)


if __name__ == "__main__":
    main()
