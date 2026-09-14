import random
import unittest

import chess

import fast_engine


class FastMoveGeneratorTests(unittest.TestCase):
    def assert_generator_matches(self, board: chess.Board) -> None:
        expected = {move.uci() for move in board.legal_moves}
        actual = fast_engine.legal_moves_for_test(board.fen())
        self.assertEqual(actual, expected, board.fen())

    def test_special_positions(self) -> None:
        positions = (
            chess.STARTING_FEN,
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1",
            "4r1k1/8/8/8/8/8/8/4K3 w - - 0 1",
        )
        for fen in positions:
            with self.subTest(fen=fen):
                self.assert_generator_matches(chess.Board(fen))

    def test_random_game_positions(self) -> None:
        rng = random.Random(20260904)
        board = chess.Board()
        checked = 0
        while checked < 500:
            self.assert_generator_matches(board)
            checked += 1
            moves = list(board.legal_moves)
            if not moves or board.is_game_over() or len(board.move_stack) > 180:
                board.reset()
            else:
                board.push(rng.choice(moves))

    def test_perft_reference_positions(self) -> None:
        references = (
            (chess.STARTING_FEN, 3, 8_902),
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                2,
                2_039,
            ),
        )
        for fen, depth, expected in references:
            with self.subTest(fen=fen, depth=depth):
                self.assertEqual(fast_engine.perft_for_test(fen, depth), expected)

    def test_preserves_kingside_castling_pawn_on_equivalent_recapture(self) -> None:
        board = chess.Board(
            "r1bqkb1r/1p2pppp/p2p1B2/8/4P3/2N5/PPQ2PPP/R3KB1R b KQkq - 0 9"
        )
        candidate = chess.Move.from_uci("g7f6")
        self.assertEqual(
            fast_engine._postprocess_candidate(board, candidate),
            chess.Move.from_uci("e7f6"),
        )

    def test_rejects_an_immediate_queen_for_rook_loss(self) -> None:
        board = chess.Board(
            "4r1k1/6pp/Q2p1q2/2rP3P/P1P5/2R2p2/2R2P2/5K2 w - - 2 35"
        )
        candidate = chess.Move.from_uci("a6b5")
        replacement = fast_engine._postprocess_candidate(board, candidate)
        self.assertNotEqual(replacement, candidate)
        self.assertIn(replacement, board.legal_moves)
        self.assertFalse(fast_engine._unsafe_root_exchange(board, replacement))
        self.assertLess(fast_engine._immediate_exchange_loss(board, replacement), 120)

    def test_rejects_a_losing_knight_capture(self) -> None:
        board = chess.Board(
            "2r2rk1/3nq3/bpp1pn2/p2p1pp1/P1PP4/1PN1N1P1/R2QP1BP/5RK1 w - - 2 20"
        )
        candidate = chess.Move.from_uci("e3f5")
        self.assertGreaterEqual(fast_engine._immediate_exchange_loss(board, candidate), 120)
        replacement = fast_engine._postprocess_candidate(board, candidate)
        self.assertNotEqual(replacement, candidate)
        self.assertIn(replacement, board.legal_moves)
        self.assertFalse(fast_engine._unsafe_root_exchange(board, replacement))
        self.assertLess(fast_engine._immediate_exchange_loss(board, replacement), 120)
        self.assertEqual(replacement, chess.Move.from_uci("c4d5"))

    def test_mate_scores_round_trip_through_transposition_table(self) -> None:
        for ply in (0, 1, 17, fast_engine.MAX_PLY - 1):
            for score in (
                fast_engine.MATE - 3,
                -fast_engine.MATE + 5,
                731,
                -842,
            ):
                with self.subTest(ply=ply, score=score):
                    stored = fast_engine._to_tt(score, ply)
                    self.assertEqual(fast_engine._from_tt(stored, ply), score)

if __name__ == "__main__":
    unittest.main()
