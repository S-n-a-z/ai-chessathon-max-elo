"""A time-safe classical chess engine for the AI Chessathon.

The engine is deliberately self-contained and readable. Its strength comes from iterative
deepening alpha-beta search, strong move ordering, a persistent transposition table, quiescence
search, and a tapered handcrafted evaluation. It does not call or contain a third-party engine.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Hashable
from typing import NamedTuple

import chess

from endgame import choose_tablebase_move
from opening import choose_opening_move

_fast_choose_move: Callable[[str, int], str] | None
try:
    from fast_engine import choose_move as _fast_choose_move
except Exception:  # Keep a legal classical fallback if JIT initialization ever fails.
    _fast_choose_move = None

INF = 40_000
MATE = 32_000
MATE_BOUND = 31_000
MAX_PLY = 96
MAX_TT_ENTRIES = 280_000

EXACT = 0
LOWER = 1
UPPER = 2

MG_VALUE = (0, 100, 320, 335, 500, 930, 0)
EG_VALUE = (0, 120, 310, 345, 525, 900, 0)
PHASE_VALUE = (0, 0, 1, 1, 2, 4, 0)
MOBILITY_MG = (0, 0, 4, 4, 2, 1, 0)
MOBILITY_EG = (0, 0, 3, 4, 3, 2, 0)
PASSED_MG = (0, 4, 9, 18, 32, 55, 90, 0)
PASSED_EG = (0, 8, 18, 35, 62, 105, 170, 0)


class TTEntry(NamedTuple):
    depth: int
    score: int
    bound: int
    move: chess.Move | None
    generation: int


class SearchTimeout(Exception):
    """Raised internally to unwind a search after its hard deadline."""


def _build_piece_square_tables() -> tuple[list[list[int]], list[list[int]]]:
    """Build original, smooth tables instead of embedding values from another engine."""
    middle = [[0] * 64 for _ in range(7)]
    ending = [[0] * 64 for _ in range(7)]
    for square in chess.SQUARES:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        file_distance = abs(2 * file_index - 7)
        rank_distance = abs(2 * rank_index - 7)
        centre = 14 - file_distance - rank_distance
        advance = rank_index
        edge = int(file_index in (0, 7) or rank_index in (0, 7))

        middle[chess.PAWN][square] = advance * 7 + centre // 2
        ending[chess.PAWN][square] = advance * 12 + centre // 3
        middle[chess.KNIGHT][square] = centre * 6 - edge * 18
        ending[chess.KNIGHT][square] = centre * 4 - edge * 10
        middle[chess.BISHOP][square] = centre * 3 - edge * 5
        ending[chess.BISHOP][square] = centre * 3
        middle[chess.ROOK][square] = advance * 2 + centre // 3
        ending[chess.ROOK][square] = centre * 2
        middle[chess.QUEEN][square] = centre - advance
        ending[chess.QUEEN][square] = centre * 2
        middle[chess.KING][square] = -centre * 5
        ending[chess.KING][square] = centre * 6

    # Castled king locations receive a modest middlegame preference.
    middle[chess.KING][chess.G1] += 42
    middle[chess.KING][chess.C1] += 34
    return middle, ending


def _build_passed_masks() -> list[list[int]]:
    masks = [[0] * 64 for _ in range(2)]
    for color in chess.COLORS:
        for square in chess.SQUARES:
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            mask = 0
            for other_file in range(max(0, file_index - 1), min(7, file_index + 1) + 1):
                ranks = range(rank_index + 1, 8) if color == chess.WHITE else range(rank_index)
                for other_rank in ranks:
                    mask |= chess.BB_SQUARES[chess.square(other_file, other_rank)]
            masks[int(color)][square] = mask
    return masks


PST_MG, PST_EG = _build_piece_square_tables()
PASSED_MASKS = _build_passed_masks()

_tt: dict[Hashable, TTEntry] = {}
_killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]
_history: list[list[list[int]]] = [[[0] * 64 for _ in range(64)] for _ in range(2)]
_generation = 0
_nodes = 0
_hard_deadline = 0.0


def _relative_square(piece: chess.Piece, square: chess.Square) -> chess.Square:
    return square if piece.color == chess.WHITE else chess.square_mirror(square)


def _evaluate(board: chess.Board) -> int:
    """Taper a positional middlegame score into a king-and-pawn endgame score."""
    mg = 0
    eg = 0
    phase = 0
    attack_maps = [0, 0]

    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        relative = _relative_square(piece, square)
        piece_type = piece.piece_type
        mg += sign * (MG_VALUE[piece_type] + PST_MG[piece_type][relative])
        eg += sign * (EG_VALUE[piece_type] + PST_EG[piece_type][relative])
        phase += PHASE_VALUE[piece_type]

        attacks = board.attacks_mask(square)
        attack_maps[int(piece.color)] |= attacks
        if piece_type not in (chess.PAWN, chess.KING):
            mobility = (attacks & ~board.occupied_co[piece.color]).bit_count()
            mg += sign * MOBILITY_MG[piece_type] * mobility
            eg += sign * MOBILITY_EG[piece_type] * mobility

    all_pawns = board.pawns
    for color in chess.COLORS:
        sign = 1 if color == chess.WHITE else -1
        pawns = board.pieces_mask(chess.PAWN, color)
        enemy_pawns = board.pieces_mask(chess.PAWN, not color)
        file_counts = [(pawns & chess.BB_FILES[file_index]).bit_count() for file_index in range(8)]

        for count in file_counts:
            if count > 1:
                mg -= sign * 13 * (count - 1)
                eg -= sign * 19 * (count - 1)

        for square in chess.scan_forward(pawns):
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            relative_rank = rank_index if color == chess.WHITE else 7 - rank_index
            adjacent_files = 0
            if file_index > 0:
                adjacent_files |= chess.BB_FILES[file_index - 1]
            if file_index < 7:
                adjacent_files |= chess.BB_FILES[file_index + 1]

            if not pawns & adjacent_files:
                mg -= sign * 11
                eg -= sign * 9
            if not enemy_pawns & PASSED_MASKS[int(color)][square]:
                mg += sign * PASSED_MG[relative_rank]
                eg += sign * PASSED_EG[relative_rank]

            support_rank = rank_index - 1 if color == chess.WHITE else rank_index + 1
            if 0 <= support_rank < 8:
                supporters = adjacent_files & chess.BB_RANKS[support_rank]
                if pawns & supporters:
                    mg += sign * 7
                    eg += sign * 10

        bishops = board.pieces_mask(chess.BISHOP, color)
        if bishops.bit_count() >= 2:
            mg += sign * 31
            eg += sign * 42

        for square in chess.scan_forward(board.pieces_mask(chess.ROOK, color)):
            file_mask = chess.BB_FILES[chess.square_file(square)]
            if not all_pawns & file_mask:
                mg += sign * 20
                eg += sign * 14
            elif not pawns & file_mask:
                mg += sign * 11
                eg += sign * 7

        king_square = board.king(color)
        if king_square is not None:
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            direction = 1 if color == chess.WHITE else -1
            shield = 0
            for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
                for distance, value in ((1, 12), (2, 5)):
                    shield_rank = king_rank + direction * distance
                    if 0 <= shield_rank < 8:
                        shield_square = chess.square(file_index, shield_rank)
                        if pawns & chess.BB_SQUARES[shield_square]:
                            shield += value
            mg += sign * shield

            king_ring = board.attacks_mask(king_square)
            pressure = (attack_maps[int(not color)] & king_ring).bit_count()
            mg -= sign * pressure * 8

    phase = min(24, phase)
    white_score = (mg * phase + eg * (24 - phase)) // 24
    score = white_score if board.turn == chess.WHITE else -white_score
    score += 10  # Initiative/tempo.
    if board.is_check():
        score -= 28
    return score


def _check_time() -> None:
    if time.perf_counter() >= _hard_deadline:
        raise SearchTimeout


def _draw_on_line(board: chess.Board, ply: int) -> bool:
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return True
    # A repetition cycle needs at least four plies. Avoid the expensive history replay elsewhere.
    return ply >= 4 and board.halfmove_clock >= 4 and board.is_repetition(2)


def _captured_piece_type(board: chess.Board, move: chess.Move) -> chess.PieceType:
    if board.is_en_passant(move):
        return chess.PAWN
    victim = board.piece_type_at(move.to_square)
    return victim or 0


def _move_score(
    board: chess.Board, move: chess.Move, tt_move: chess.Move | None, ply: int
) -> int:
    if move == tt_move:
        return 30_000_000
    if move.promotion:
        promotion_gain = MG_VALUE[move.promotion] - MG_VALUE[chess.PAWN]
        return 20_000_000 + promotion_gain * 1_000
    if board.is_capture(move):
        victim = _captured_piece_type(board, move)
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        return 18_000_000 + MG_VALUE[victim] * 100 - MG_VALUE[attacker]
    if ply < MAX_PLY:
        if move == _killers[ply][0]:
            return 15_000_000
        if move == _killers[ply][1]:
            return 14_000_000
    if board.gives_check(move):
        return 12_000_000
    return _history[int(board.turn)][move.from_square][move.to_square]


def _ordered_moves(
    board: chess.Board,
    moves: list[chess.Move],
    tt_move: chess.Move | None,
    ply: int,
) -> list[chess.Move]:
    moves.sort(key=lambda move: _move_score(board, move, tt_move, ply), reverse=True)
    return moves


def _score_from_tt(score: int, ply: int) -> int:
    if score > MATE_BOUND:
        return score - ply
    if score < -MATE_BOUND:
        return score + ply
    return score


def _score_to_tt(score: int, ply: int) -> int:
    if score > MATE_BOUND:
        return score + ply
    if score < -MATE_BOUND:
        return score - ply
    return score


def _quiescence(board: chess.Board, alpha: int, beta: int, ply: int) -> int:
    global _nodes
    _nodes += 1
    if _nodes & 255 == 0:
        _check_time()
    if ply >= MAX_PLY - 1:
        return _evaluate(board)
    if _draw_on_line(board, ply):
        return 0

    in_check = board.is_check()
    stand_pat = _evaluate(board)
    if not in_check:
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    moves = list(board.legal_moves)
    if not moves:
        return -MATE + ply if in_check else 0
    if not in_check:
        moves = [move for move in moves if board.is_capture(move) or move.promotion]
    _ordered_moves(board, moves, None, ply)

    for move in moves:
        if not in_check and not move.promotion:
            victim = _captured_piece_type(board, move)
            if stand_pat + MG_VALUE[victim] + 180 < alpha:
                continue
        board.push(move)
        try:
            score = -_quiescence(board, -beta, -alpha, ply + 1)
        finally:
            board.pop()
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
    pieces = board.occupied_co[color] & ~(board.pawns | board.kings)
    return bool(pieces)


def _record_cutoff(move: chess.Move, color: chess.Color, depth: int, ply: int) -> None:
    if ply < MAX_PLY and move != _killers[ply][0]:
        _killers[ply][1] = _killers[ply][0]
        _killers[ply][0] = move
    old = _history[int(color)][move.from_square][move.to_square]
    _history[int(color)][move.from_square][move.to_square] = min(1_000_000, old + depth * depth)


def _store_tt(key: Hashable, entry: TTEntry) -> None:
    previous = _tt.get(key)
    if previous is None or previous.generation != _generation or entry.depth >= previous.depth - 1:
        _tt[key] = entry


def _search(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    global _nodes
    _nodes += 1
    if _nodes & 255 == 0:
        _check_time()
    if ply >= MAX_PLY - 1:
        return _evaluate(board)
    if _draw_on_line(board, ply):
        return 0
    if depth <= 0:
        return _quiescence(board, alpha, beta, ply)

    key = board._transposition_key()
    entry = _tt.get(key)
    tt_move = entry.move if entry else None
    if entry and entry.depth >= depth:
        tt_score = _score_from_tt(entry.score, ply)
        if entry.bound == EXACT:
            return tt_score
        if entry.bound == LOWER:
            alpha = max(alpha, tt_score)
        else:
            beta = min(beta, tt_score)
        if alpha >= beta:
            return tt_score

    original_alpha = alpha
    in_check = board.is_check()
    moves = list(board.legal_moves)
    if not moves:
        return -MATE + ply if in_check else 0

    static_score = _evaluate(board)
    if not in_check and depth <= 3 and static_score - 95 * depth >= beta:
        return static_score
    if not in_check and depth == 1 and static_score + 170 <= alpha:
        razor_score = _quiescence(board, alpha, beta, ply)
        if razor_score <= alpha:
            return razor_score

    if (
        depth >= 3
        and not in_check
        and static_score >= beta
        and _has_non_pawn_material(board, board.turn)
    ):
        reduction = 2 + depth // 4
        board.push(chess.Move.null())
        try:
            null_score = -_search(board, depth - 1 - reduction, -beta, -beta + 1, ply + 1)
        finally:
            board.pop()
        if null_score >= beta:
            return null_score

    _ordered_moves(board, moves, tt_move, ply)
    best_score = -INF
    best_move: chess.Move | None = None
    mover = board.turn

    for move_index, move in enumerate(moves):
        is_capture = board.is_capture(move)
        is_quiet = not is_capture and not move.promotion
        gives_check = board.gives_check(move)
        extension = 1 if in_check and depth >= 2 and ply < 12 else 0
        next_depth = depth - 1 + extension

        board.push(move)
        try:
            if move_index == 0:
                score = -_search(board, next_depth, -beta, -alpha, ply + 1)
            else:
                reduced_depth = next_depth
                if depth >= 3 and move_index >= 4 and is_quiet and not in_check and not gives_check:
                    reduction = 1 + int(depth >= 6) + int(move_index >= 10)
                    reduced_depth = max(0, next_depth - reduction)
                score = -_search(board, reduced_depth, -alpha - 1, -alpha, ply + 1)
                if score > alpha and reduced_depth < next_depth:
                    score = -_search(board, next_depth, -alpha - 1, -alpha, ply + 1)
                if score > alpha and score < beta:
                    score = -_search(board, next_depth, -beta, -alpha, ply + 1)
        finally:
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if is_quiet:
                _record_cutoff(move, mover, depth, ply)
            break

    bound = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
    _store_tt(key, TTEntry(depth, _score_to_tt(best_score, ply), bound, best_move, _generation))
    return best_score


def _search_root(
    board: chess.Board,
    depth: int,
    alpha: int,
    beta: int,
    preferred: chess.Move | None,
) -> tuple[int, chess.Move]:
    _check_time()
    key = board._transposition_key()
    entry = _tt.get(key)
    tt_move = preferred or (entry.move if entry else None)
    moves = _ordered_moves(board, list(board.legal_moves), tt_move, 0)
    if not moves:
        raise ValueError("get_move was called on a terminal position")

    original_alpha = alpha
    best_score = -INF
    best_move = moves[0]
    mover = board.turn
    for move_index, move in enumerate(moves):
        is_quiet = not board.is_capture(move) and not move.promotion
        board.push(move)
        try:
            if move_index == 0:
                score = -_search(board, depth - 1, -beta, -alpha, 1)
            else:
                score = -_search(board, depth - 1, -alpha - 1, -alpha, 1)
                if score > alpha and score < beta:
                    score = -_search(board, depth - 1, -beta, -alpha, 1)
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if is_quiet:
                _record_cutoff(move, mover, depth, 0)
            break
        _check_time()

    bound = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
    _store_tt(key, TTEntry(depth, _score_to_tt(best_score, 0), bound, best_move, _generation))
    return best_score, best_move


def _time_limits(time_left_ms: int) -> tuple[float, float]:
    remaining = max(0.001, time_left_ms / 1_000.0)
    if remaining < 1.0:
        soft = max(0.004, remaining * 0.055)
    elif remaining < 5.0:
        soft = 0.05 + remaining * 0.055
    else:
        soft = min(4.2, 0.10 + remaining / 35.0)
    hard = min(max(0.008, remaining - 0.06), soft * 1.7)
    return soft, max(soft, hard)


def _classic_get_move(fen: str, time_left_ms: int) -> str:
    """Return the best move found before a conservative wall-clock deadline."""
    global _generation, _hard_deadline, _nodes
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return "0000"
    if len(legal_moves) == 1:
        return legal_moves[0].uci()

    _generation += 1
    if len(_tt) > MAX_TT_ENTRIES:
        _tt.clear()
    _nodes = 0
    start = time.perf_counter()
    soft_seconds, hard_seconds = _time_limits(time_left_ms)
    soft_deadline = start + soft_seconds
    _hard_deadline = start + hard_seconds

    # Even on an emergency clock, the fallback is ordered rather than arbitrary.
    best_move = _ordered_moves(board, legal_moves, None, 0)[0]
    best_score = 0

    for depth in range(1, 64):
        iteration_start = time.perf_counter()
        if depth >= 3:
            window = 55
            alpha = max(-INF, best_score - window)
            beta = min(INF, best_score + window)
        else:
            alpha, beta = -INF, INF

        try:
            while True:
                score, move = _search_root(board, depth, alpha, beta, best_move)
                if score <= alpha and alpha > -INF:
                    alpha = max(-INF, alpha - max(140, (beta - alpha) * 2))
                    continue
                if score >= beta and beta < INF:
                    beta = min(INF, beta + max(140, (beta - alpha) * 2))
                    continue
                best_score, best_move = score, move
                break
        except SearchTimeout:
            break

        now = time.perf_counter()
        previous_iteration_seconds = now - iteration_start
        if abs(best_score) > MATE_BOUND:
            break
        if now >= soft_deadline:
            break
        # A completed iteration predicts the next one better than a fixed maximum depth.
        if previous_iteration_seconds * 2.4 >= _hard_deadline - now:
            break

    return best_move.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Use the compiled engine, retaining the proven Python engine as an import fallback."""
    tablebase_move = choose_tablebase_move(fen)
    if tablebase_move is not None:
        return tablebase_move
    opening_move = choose_opening_move(fen)
    if opening_move is not None:
        return opening_move
    if _fast_choose_move is not None:
        return _fast_choose_move(fen, time_left_ms)
    return _classic_get_move(fen, time_left_ms)
