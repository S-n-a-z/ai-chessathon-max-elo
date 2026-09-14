import random
import time
import unittest

import chess

import agent


class AgentTests(unittest.TestCase):
    def test_compiled_search_is_active(self) -> None:
        self.assertIsNotNone(agent._fast_choose_move)

    def test_returns_legal_moves_from_varied_positions(self) -> None:
        random.seed(7)
        board = chess.Board()
        for _ in range(14):
            if board.is_game_over():
                break
            move = chess.Move.from_uci(agent.get_move(board.fen(), 120))
            self.assertIn(move, board.legal_moves)
            board.push(random.choice(list(board.legal_moves)))

    def test_finds_mate_in_one(self) -> None:
        board = chess.Board("7k/p4Q2/6K1/8/8/8/P7/8 w - - 0 1")
        move = chess.Move.from_uci(agent.get_move(board.fen(), 1_000))
        self.assertIn(move, board.legal_moves)
        board.push(move)
        self.assertTrue(board.is_checkmate())

    def test_emergency_clock_returns_promptly(self) -> None:
        board = chess.Board("7k/8/8/8/8/8/P3Q3/4K2R w - - 0 1")
        started = time.perf_counter()
        move = chess.Move.from_uci(agent.get_move(board.fen(), 80))
        elapsed = time.perf_counter() - started
        self.assertIn(move, board.legal_moves)
        self.assertLess(elapsed, 0.08)

    def test_terminal_position_has_protocol_fallback(self) -> None:
        board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(board.is_checkmate())
        self.assertEqual(agent.get_move(board.fen(), 1_000), "0000")


if __name__ == "__main__":
    unittest.main()
