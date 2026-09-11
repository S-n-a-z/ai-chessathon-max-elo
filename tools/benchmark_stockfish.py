"""Offline strength check against a locally installed Stockfish.

Stockfish is testing equipment only. It is never included in ``submission.zip`` and is never
called by the competition agent. The Chessathon rules explicitly permit engine-labelled training;
using an engine only as a local opponent is an even narrower use.
"""

from __future__ import annotations

import argparse
import os
import time

import chess
import chess.engine

import agent
import fast_engine

OPENINGS = (
    (),
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5"),
)


def _starting_board(game_index: int) -> chess.Board:
    board = chess.Board()
    for uci in OPENINGS[game_index % len(OPENINGS)]:
        board.push_uci(uci)
    return board


def _result_for_agent(outcome: chess.Outcome, agent_color: chess.Color) -> str:
    if outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == agent_color else "loss"


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the submission against limited Stockfish.")
    parser.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH"))
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--base-ms", type=int, default=5_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    args = parser.parse_args()
    if not args.stockfish:
        raise SystemExit("Pass --stockfish or set STOCKFISH_PATH (the binary is test-only).")

    wins = draws = losses = 0
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        engine.configure(
            {"Threads": 1, "Hash": 64, "UCI_LimitStrength": True, "UCI_Elo": args.elo}
        )
        for game_index in range(args.games):
            fast_engine.reset_for_test()
            board = _starting_board(game_index)
            agent_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK
            clocks = {chess.WHITE: float(args.base_ms), chess.BLACK: float(args.base_ms)}

            while not board.is_game_over(claim_draw=True) and len(board.move_stack) < 240:
                mover = board.turn
                started = time.perf_counter()
                if mover == agent_color:
                    move = chess.Move.from_uci(agent.get_move(board.fen(), int(clocks[mover])))
                else:
                    result = engine.play(
                        board,
                        chess.engine.Limit(
                            white_clock=clocks[chess.WHITE] / 1_000,
                            black_clock=clocks[chess.BLACK] / 1_000,
                            white_inc=args.increment_ms / 1_000,
                            black_inc=args.increment_ms / 1_000,
                        ),
                    )
                    if result.move is None:
                        break
                    move = result.move
                elapsed_ms = (time.perf_counter() - started) * 1_000
                clocks[mover] -= elapsed_ms
                if clocks[mover] < 0 or move not in board.legal_moves:
                    losses += int(mover == agent_color)
                    wins += int(mover != agent_color)
                    actor = "agent" if mover == agent_color else "stockfish"
                    failure = "flag" if clocks[mover] < 0 else "illegal"
                    print(f"game {game_index + 1}: {actor} {failure}")
                    break
                board.push(move)
                clocks[mover] += args.increment_ms
            else:
                outcome = board.outcome(claim_draw=True)
                if outcome is None:
                    draws += 1
                    label = "draw"
                else:
                    label = _result_for_agent(outcome, agent_color)
                    wins += int(label == "win")
                    draws += int(label == "draw")
                    losses += int(label == "loss")
                print(f"game {game_index + 1}/{args.games}: {label}")
    finally:
        engine.quit()

    score = (wins + draws / 2) / args.games
    print(f"agent vs Stockfish UCI_Elo {args.elo}: +{wins} ={draws} -{losses}, {score:.1%}")


if __name__ == "__main__":
    main()
