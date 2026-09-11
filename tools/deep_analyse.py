"""Deep, test-only Stockfish review of our moves in one or more PGNs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn

OUR_NAMES = {"Stockfish's Nightmare", "BlockShark"}
MATE_SCORE = 100_000


@dataclass(frozen=True)
class ReviewedMove:
    number: str
    played: str
    best: str
    best_score: int
    played_score: int
    loss: int
    fen: str


def _score(info: chess.engine.InfoDict, color: chess.Color) -> int:
    return info["score"].pov(color).score(mate_score=MATE_SCORE) or 0


def review_game(
    engine: chess.engine.SimpleEngine, path: Path, nodes: int
) -> list[ReviewedMove]:
    with path.open(encoding="utf-8") as stream:
        game = chess.pgn.read_game(stream)
    if game is None:
        raise ValueError(f"Could not parse {path}")
    if game.headers.get("White") in OUR_NAMES:
        our_color = chess.WHITE
    elif game.headers.get("Black") in OUR_NAMES:
        our_color = chess.BLACK
    else:
        raise ValueError(f"Could not identify our colour in {path}")
    board = game.board()
    reviewed: list[ReviewedMove] = []
    for move in game.mainline_moves():
        if board.turn == our_color:
            best_info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
            played_info = engine.analyse(
                board,
                chess.engine.Limit(nodes=nodes),
                root_moves=[move],
            )
            best_move = best_info["pv"][0]
            best_score = _score(best_info, our_color)
            played_score = _score(played_info, our_color)
            reviewed.append(
                ReviewedMove(
                    number=f"{board.fullmove_number}{'.' if our_color else '...'}",
                    played=board.san(move),
                    best=board.san(best_move),
                    best_score=best_score,
                    played_score=played_score,
                    loss=max(0, best_score - played_score),
                    fen=board.fen(),
                )
            )
        board.push(move)
    return reviewed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("pgn", nargs="+", type=Path)
    args = parser.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    try:
        engine.configure({"Threads": 1, "Hash": 256})
        for path in args.pgn:
            print(f"\n{path} — {args.nodes:,} nodes per root search")
            reviewed = review_game(engine, path, args.nodes)
            for item in sorted(reviewed, key=lambda value: value.loss, reverse=True)[: args.top]:
                print(
                    f"{item.number:<5} {item.played:<9} best {item.best:<9} "
                    f"{item.best_score:>7} -> {item.played_score:>7} ({item.loss:>5} cp)"
                )
                if item.loss >= 75:
                    print(f"      FEN {item.fen}")
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
