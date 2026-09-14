"""Correctness checks for the optimized search path and clock handling."""

import random
import time
import unittest

import chess
import numpy as np

import fast_engine as engine


class SearchCoreTests(unittest.TestCase):
    def test_recursive_search_has_no_duplicate_literal_specializations(self) -> None:
        # Root/null-move constants used to compile three complete recursive trees,
        # exhausting the cold-start budget. Import must warm one shared integer type.
        for function in (engine._search, engine._quiescence):
            self.assertEqual(len(function.signatures), 1, function.py_func.__name__)
            self.assertNotIn("Literal[", str(function.signatures[0]))

    def test_incremental_hash_matches_full_hash_and_restores_position(self) -> None:
        randomizer = random.Random(79031)
        positions = [
            chess.Board(),
            chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
            chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
            chess.Board("4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1"),
        ]
        random_board = chess.Board()
        for _ in range(150):
            moves = list(random_board.legal_moves)
            if not moves or random_board.is_game_over():
                random_board.reset()
            else:
                random_board.push(randomizer.choice(moves))
                positions.append(random_board.copy(stack=False))
        for position in positions:
            board, side, castle, ep, halfmove, king, _ = engine._encode_position(position)
            original = board.copy()
            # Python boxes a uint64 return as int; preserve the compiled caller's dtype.
            key = np.uint64(engine._position_hash(board, side, castle, ep))
            count = engine._generate_legal(
                board, side, castle, ep, halfmove, king, 0, engine._move_buffer
            )
            for index in range(count):
                move = int(engine._move_buffer[0, index])
                captured, new_castle, new_ep, _, _ = engine._make_move(
                    board, move, side, castle, halfmove, king
                )
                incremental = engine._hash_after_move(
                    board, key, move, side, captured, castle, new_castle, ep, new_ep
                )
                self.assertEqual(
                    incremental, engine._position_hash(board, -side, new_castle, new_ep)
                )
                engine._undo_move(board, move, side, captured)
                self.assertTrue((board == original).all())

    def test_tactical_generator_includes_exact_legal_captures_and_promotions(self) -> None:
        randomizer = random.Random(435)
        position = chess.Board()
        for _ in range(200):
            board, side, castle, ep, halfmove, king, _ = engine._encode_position(position)
            count = engine._generate_pseudo(
                board, side, castle, ep, king, engine._move_buffer[0], True
            )
            generated = set()
            for index in range(count):
                move = int(engine._move_buffer[0, index])
                captured, _, _, _, new_king = engine._make_move(
                    board, move, side, castle, halfmove, king
                )
                if not engine._is_attacked(board, new_king, -side):
                    generated.add(engine._decode_move(move).uci())
                engine._undo_move(board, move, side, captured)
            expected = {
                move.uci()
                for move in position.legal_moves
                if position.is_capture(move) or move.promotion
            }
            self.assertEqual(generated, expected, position.fen())
            moves = list(position.legal_moves)
            if moves and not position.is_game_over():
                position.push(randomizer.choice(moves))
            else:
                position.reset()

    def test_quiescence_recognizes_stalemate_before_stand_pat_cutoff(self) -> None:
        engine.reset_for_test()
        position = chess.Board("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
        self.assertTrue(position.is_stalemate())
        encoded = engine._encode_position(position)
        engine._hash_stack[0] = engine._position_hash(*encoded[:4])
        engine._nodes[0] = 0
        engine._nodes[1] = 100_000
        score = engine._quiescence(
            *encoded,
            -engine.INF,
            -20_000,
            0,
            engine._move_buffer,
            engine._score_buffer,
            engine._killers,
            engine._history,
            engine._hash_stack,
            engine._nodes,
        )
        self.assertEqual(score, 0)

    def test_real_deadline_aborts_even_with_large_node_budget(self) -> None:
        engine.reset_for_test()
        encoded = engine._encode_position(chess.Board())
        original = encoded[0].copy()
        engine._nodes[2] = time.perf_counter_ns() - 1
        _, _, nodes, completed = engine._root_search(
            *encoded,
            20,
            -engine.INF,
            engine.INF,
            0,
            100_000_000,
            1,
            engine._tt_keys,
            engine._tt_scores,
            engine._tt_moves,
            engine._tt_depths,
            engine._tt_bounds,
            engine._tt_ages,
            engine._move_buffer,
            engine._score_buffer,
            engine._killers,
            engine._history,
            engine._hash_stack,
            engine._nodes,
        )
        engine._nodes[2] = 0
        self.assertFalse(completed)
        self.assertLessEqual(nodes, 1024)
        self.assertTrue((encoded[0][:128] == original[:128]).all())


if __name__ == "__main__":
    unittest.main()
