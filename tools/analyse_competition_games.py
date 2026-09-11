"""Rank mistakes in downloaded Chessathon PGNs using the public game review.

The dashboard downloads remain the source of truth for the played moves and clocks.  Public game
pages include a Stockfish 16 depth-16 evaluation after every ply; this tool lines those values up
with the PGN and reports the largest evaluation swings made by our agent.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot

OUR_NAMES = {"Stockfish's Nightmare", "BlockShark"}
GAME_IDS = {
    "round-24-draw.pgn": "0e18b09a-e0ce-4579-ac2e-8fcaef72e73c",
    "round-25-loss.pgn": "827918fa-fa5d-4a3b-a822-0f206daf7641",
    "round-28-loss.pgn": "d133d71b-1f29-4347-bd1b-0d3b6f2e64d2",
    "round-29-loss.pgn": "0d2013cf-58e1-459b-9c38-36790afd15a2",
    "round-30-loss.pgn": "b48c6764-293e-4f60-9bbb-af33b0b317bb",
    "round-33-loss.pgn": "d81c09a3-3d28-40a8-86ff-510fc46f4709",
    "round-34-loss.pgn": "117a11e7-0b56-4c2c-b766-081036ea47e1",
    "round-37-loss.pgn": "bdc3361b-d82b-4091-8750-3949baac8b2e",
    "round-38-loss.pgn": "4da77f64-53b8-4dcd-b9ea-c7f94eb9c1a7",
    "round-41-loss.pgn": "cf99852d-c1e9-4375-812b-10257541d735",
    "round-43-loss.pgn": "12cb1b08-4e69-4222-af21-d8112bd343ae",
    "round-44-draw.pgn": "035016a8-1b01-42f4-9ebf-b3e3a4d3223a",
    "round-45-draw.pgn": "afaa28f3-8340-443d-990d-2f16b758b6bf",
    "round-46-loss.pgn": "3dc8014b-87b3-4901-86e1-f6812e6ad6a6",
    "round-47-loss.pgn": "6b94e1ae-f7b0-478a-b1dc-86062e3eee0d",
    "round-48-loss.pgn": "cac12c79-8676-465a-9bf6-4136b2e79af0",
    "round-49-draw.pgn": "9edba697-d232-4e66-8f07-dc28339b1339",
    "round-50-loss.pgn": "0af3032a-aab8-461d-aa15-5f9160e6dfdd",
    "round-51-draw.pgn": "4b0080c6-aae7-41e6-817c-842481c4984c",
    "round-52-loss.pgn": "a3901e17-5c4d-4e5e-839e-d77afbf00a08",
    "round-53-loss.pgn": "03b462c7-f7d0-47af-9119-d2e600173de1",
}
EVAL_RE = re.compile(
    r'\\"evals\\":\{\\"cp\\":(\[[^\]]*\]),'
    r'\\"best\\":(\[[^\]]*\]),\\"depth\\":(\d+)'
)


@dataclass(frozen=True)
class MoveLoss:
    move_number: str
    san: str
    uci: str
    best_uci: str
    before_cp: int
    after_cp: int
    loss_cp: int
    book_uci: str
    fen: str


def _review(game_id: str) -> tuple[list[int], list[str], int]:
    request = urllib.request.Request(
        f"https://aichessathon.com/game/{game_id}",
        headers={"User-Agent": "AI-Chessathon-local-analysis/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8")
    match = EVAL_RE.search(html)
    if match is None:
        raise RuntimeError(f"No public review data found for game {game_id}")
    scores = json.loads(match.group(1))
    best_moves = json.loads(match.group(2).replace(r'\"', '"'))
    return scores, best_moves, int(match.group(3))


def _move_number(board: chess.Board) -> str:
    suffix = "." if board.turn == chess.WHITE else "..."
    return f"{board.fullmove_number}{suffix}"


def analyse(
    path: Path, game_id: str, book: dict[int, str]
) -> tuple[str, int, list[MoveLoss]]:
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"Could not parse {path}")

    scores, best_moves, depth = _review(game_id)
    moves = list(game.mainline_moves())
    if len(scores) != len(moves) + 1:
        raise ValueError(
            f"Review/PGN length mismatch for {path}: {len(scores)} scores, {len(moves)} plies"
        )

    if game.headers.get("White") in OUR_NAMES:
        our_color = chess.WHITE
    elif game.headers.get("Black") in OUR_NAMES:
        our_color = chess.BLACK
    else:
        raise ValueError(f"could not identify our colour in {path}")
    board = game.board()
    losses: list[MoveLoss] = []
    for index, move in enumerate(moves):
        before = int(scores[index])
        after = int(scores[index + 1])
        if board.turn == our_color:
            loss = max(0, before - after) if our_color == chess.WHITE else max(0, after - before)
            losses.append(
                MoveLoss(
                    move_number=_move_number(board),
                    san=board.san(move),
                    uci=move.uci(),
                    best_uci=best_moves[index],
                    before_cp=before,
                    after_cp=after,
                    loss_cp=loss,
                    book_uci=book.get(chess.polyglot.zobrist_hash(board), ""),
                    fen=board.fen(),
                )
            )
        board.push(move)
    return "White" if our_color == chess.WHITE else "Black", depth, losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, default=Path("games"))
    parser.add_argument("--top", type=int, default=12)
    arguments = parser.parse_args()
    book_path = Path(__file__).resolve().parents[1] / "weights" / "opening_book.json"
    payload = json.loads(book_path.read_text(encoding="utf-8"))
    book = {int(key, 16): value for key, value in payload["moves"].items()}

    for filename, game_id in GAME_IDS.items():
        path = arguments.games / filename
        color, depth, losses = analyse(path, game_id, book)
        print(f"\n{filename}: our colour {color}; Stockfish review depth {depth}")
        print("move    played    best      book      before   after    loss")
        for item in sorted(losses, key=lambda value: value.loss_cp, reverse=True)[: arguments.top]:
            print(
                f"{item.move_number:<7} {item.san:<9} {item.best_uci:<9} {item.book_uci:<9} "
                f"{item.before_cp:>7} {item.after_cp:>7} {item.loss_cp:>6} cp"
            )
            if item.loss_cp >= 100:
                print(f"        FEN {item.fen}")


if __name__ == "__main__":
    main()
