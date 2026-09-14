import unittest

import chess

import opening


class OpeningBookTests(unittest.TestCase):
    def test_expected_book_size(self) -> None:
        self.assertGreaterEqual(len(opening.BOOK), 5_474)

    def test_starting_move_is_legal(self) -> None:
        board = chess.Board()
        uci = opening.choose_opening_move(board.fen())
        self.assertIsNotNone(uci)
        self.assertIn(chess.Move.from_uci(uci or "0000"), board.legal_moves)

    def test_non_opening_position_is_not_matched(self) -> None:
        fen = "8/8/8/8/8/2K5/4Q3/2k5 w - - 0 1"
        self.assertIsNone(opening.choose_opening_move(fen))

    def test_late_repetition_of_book_position_is_not_looked_up(self) -> None:
        board = chess.Board()
        board.fullmove_number = 20
        self.assertIsNotNone(opening.choose_opening_move(board.fen()))
        board.fullmove_number = 21
        self.assertIsNone(opening.choose_opening_move(board.fen()))


if __name__ == "__main__":
    unittest.main()
