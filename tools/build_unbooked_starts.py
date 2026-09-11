"""Select level, public middlegames beyond the permitted opening-book boundary."""

import argparse
import json
from pathlib import Path

import chess.engine
import chess.pgn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    starts = []
    engine = chess.engine.SimpleEngine.popen_uci(str(args.stockfish))
    try:
        engine.configure({"Threads": 1, "Hash": 32})
        for source in sorted(args.games.glob("round-*.pgn"), reverse=True):
            with source.open(encoding="utf-8") as stream:
                game = chess.pgn.read_game(stream)
            if game is None or game.errors:
                continue
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number != 21 or board.turn != chess.WHITE:
                    continue
                if not board.is_valid() or board.is_game_over():
                    break
                result = engine.analyse(board, chess.engine.Limit(nodes=50_000))
                score = result["score"].white().score()
                if score is not None and abs(score) <= 80:
                    starts.append(
                        {"fen": board.fen(), "source": str(source), "selection_teacher_cp": score}
                    )
                break
            if len(starts) >= 8:
                break
    finally:
        engine.quit()
    if len(starts) < 4:
        raise ValueError(f"insufficient level middlegames: {len(starts)}")
    args.out.write_text(json.dumps({"starts": starts}, indent=2), encoding="utf-8")
    print(f"Selected {len(starts)} level middlegames, all beyond move20", flush=True)


if __name__ == "__main__":
    main()
