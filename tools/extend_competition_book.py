"""Extend the genuine opening book around public AI Chessathon start positions."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.polyglot


def _load_starts(path: Path) -> tuple[list[tuple[chess.Board, chess.Color]], int]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    starts: list[tuple[chess.Board, chess.Color]] = []
    latest_round = 0
    for item in payload["starts"]:
        board = chess.Board(item["fen"])
        # Future pairings can assign either colour to a repeated curated start.
        # Building both perspectives keeps the tree symmetric: on the agent's
        # turns follow one teacher PV, and on the opponent's turns retain the
        # requested number of plausible replies.
        starts.append((board, chess.WHITE))
        starts.append((board.copy(stack=False), chess.BLACK))
        round_value = item.get("round")
        if isinstance(round_value, int):
            latest_round = max(latest_round, round_value)
    return starts, latest_round


def _principal_moves(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    nodes: int,
    count: int,
) -> list[chess.Move]:
    legal_count = board.legal_moves.count()
    if not legal_count:
        return []
    analysis = engine.analyse(
        board,
        chess.engine.Limit(nodes=nodes),
        multipv=min(count, legal_count),
    )
    lines = analysis if isinstance(analysis, list) else [analysis]
    return [line["pv"][0] for line in lines if line.get("pv")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=Path, default=Path("weights/opening_book.json"))
    parser.add_argument("--starts", type=Path, default=Path("games/public-starts.json"))
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("weights/opening_book.candidate.json"))
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--opponent-branches", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=50_000)
    arguments = parser.parse_args()

    payload: dict[str, Any] = json.loads(arguments.book.read_text(encoding="utf-8"))
    moves: dict[str, str] = dict(payload["moves"])
    starts, latest_round = _load_starts(arguments.starts)
    queue: deque[tuple[chess.Board, chess.Color, int]] = deque(
        (board, color, 0) for board, color in starts
    )
    visited: set[tuple[int, chess.Color, int]] = set()
    analysed: dict[int, list[chess.Move]] = {}
    added: set[int] = set()

    engine = chess.engine.SimpleEngine.popen_uci(str(arguments.stockfish))
    try:
        engine.configure({"Threads": 1, "Hash": 256})
        while queue:
            board, engine_color, distance = queue.popleft()
            key = chess.polyglot.zobrist_hash(board)
            state = (key, engine_color, distance)
            if state in visited or board.is_game_over(claim_draw=True):
                continue
            visited.add(state)

            principal = analysed.get(key)
            if principal is None:
                principal = _principal_moves(
                    engine, board, arguments.nodes, arguments.opponent_branches
                )
                analysed[key] = principal
            if not principal:
                continue

            moves[f"{key:016x}"] = principal[0].uci()
            added.add(key)
            if len(added) % 250 == 0:
                print(f"labelled {len(added)} unique competition opening positions", flush=True)

            if distance + 1 >= arguments.plies:
                continue
            continuations = principal[:1] if board.turn == engine_color else principal
            for move in continuations:
                child = board.copy(stack=False)
                child.push(move)
                queue.append((child, engine_color, distance + 1))
    finally:
        engine.quit()

    payload["moves"] = moves
    payload["competition_extension"] = {
        "source": f"public AI Chessathon opening positions through round {latest_round}",
        "unique_starting_positions": len(starts) // 2,
        "perspectives_per_start": 2,
        "plies_from_start": arguments.plies,
        "opponent_branches": arguments.opponent_branches,
        "nodes_per_position": arguments.nodes,
        "labeler": "Stockfish 19",
        "positions_labelled": len(added),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    print(
        f"wrote {len(moves)} total positions ({len(added)} competition positions) "
        f"to {arguments.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
