import unittest

import chess
import chess.syzygy

import endgame


class EndgameTablebaseTests(unittest.TestCase):
    def test_complete_three_and_four_piece_tables_load(self) -> None:
        self.assertIsNotNone(endgame.TABLEBASE)
        assert endgame.TABLEBASE is not None
        required = set(chess.syzygy.tablenames(piece_count=4)) - {"KNvK", "KBvK"}
        self.assertTrue(required <= set(endgame.TABLEBASE.wdl))
        self.assertTrue(required <= set(endgame.TABLEBASE.dtz))

    def test_mate_in_one_is_selected(self) -> None:
        board = chess.Board("8/8/8/8/8/2K5/4Q3/2k5 w - - 0 1")
        move = endgame.choose_tablebase_move(board.fen())
        self.assertIsNotNone(move)
        board.push_uci(move or "0000")
        self.assertTrue(board.is_checkmate())

    def test_draw_is_preserved(self) -> None:
        board = chess.Board("8/8/8/8/8/2k5/4B3/2K5 w - - 0 1")
        move = endgame.choose_tablebase_move(board.fen())
        self.assertIsNotNone(move)
        board.push_uci(move or "0000")
        assert endgame.TABLEBASE is not None
        self.assertEqual(endgame.TABLEBASE.get_wdl(board), 0)

    def test_search_handles_positions_outside_tables(self) -> None:
        self.assertIsNone(endgame.choose_tablebase_move(chess.STARTING_FEN))


if __name__ == "__main__":
    unittest.main()
