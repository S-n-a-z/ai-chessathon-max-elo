# mypy: ignore-errors
"""Original 0x88/Numba search core used by :mod:`agent`.

Python-chess remains the contract/parser and the final legality backstop.  The hot path below uses
our own readable move generator and evaluator so Numba can compile the full recursive search.
No third-party engine code, network, subprocess, or external data is used at runtime.
"""

from __future__ import annotations

import threading
import time

import chess
import numpy as np
from numba import njit

WHITE = 1
BLACK = -1
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

INF = 40_000
MATE = 32_000
MATE_BOUND = 31_000
ABORT = 50_000
MAX_PLY = 96
MAX_MOVES = 256

EXACT = 0
LOWER = 1
UPPER = 2

CASTLE_WK = 1
CASTLE_WQ = 2
CASTLE_BK = 4
CASTLE_BQ = 8

FLAG_EP = 1 << 17
FLAG_CASTLE = 1 << 18
FLAG_DOUBLE = 1 << 19

TT_BITS = 19
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1

KNIGHT_OFFSETS = np.array((33, 31, 18, 14, -14, -18, -31, -33), dtype=np.int16)
KING_OFFSETS = np.array((16, -16, 1, -1, 17, 15, -15, -17), dtype=np.int16)
BISHOP_DIRECTIONS = np.array((17, 15, -15, -17), dtype=np.int16)
ROOK_DIRECTIONS = np.array((16, -16, 1, -1), dtype=np.int16)

MG_VALUE = np.array((0, 100, 320, 335, 500, 930, 0), dtype=np.int32)
EG_VALUE = np.array((0, 120, 310, 345, 525, 900, 0), dtype=np.int32)
PHASE_VALUE = np.array((0, 0, 1, 1, 2, 4, 0), dtype=np.int32)
MOBILITY_MG = np.array((0, 0, 4, 4, 2, 1, 0), dtype=np.int32)
MOBILITY_EG = np.array((0, 0, 3, 4, 3, 2, 0), dtype=np.int32)
PASSED_MG = np.array((0, 4, 9, 18, 32, 55, 90, 0), dtype=np.int32)
PASSED_EG = np.array((0, 8, 18, 35, 62, 105, 170, 0), dtype=np.int32)


def _piece_square_tables() -> tuple[np.ndarray, np.ndarray]:
    middle = np.zeros((7, 128), dtype=np.int16)
    ending = np.zeros((7, 128), dtype=np.int16)
    for rank_index in range(8):
        for file_index in range(8):
            square = rank_index * 16 + file_index
            file_distance = abs(2 * file_index - 7)
            rank_distance = abs(2 * rank_index - 7)
            centre = 14 - file_distance - rank_distance
            edge = int(file_index in (0, 7) or rank_index in (0, 7))
            middle[PAWN, square] = rank_index * 7 + centre // 2
            ending[PAWN, square] = rank_index * 12 + centre // 3
            middle[KNIGHT, square] = centre * 6 - edge * 18
            ending[KNIGHT, square] = centre * 4 - edge * 10
            middle[BISHOP, square] = centre * 3 - edge * 5
            ending[BISHOP, square] = centre * 3
            middle[ROOK, square] = rank_index * 2 + centre // 3
            ending[ROOK, square] = centre * 2
            middle[QUEEN, square] = centre - rank_index
            ending[QUEEN, square] = centre * 2
            middle[KING, square] = -centre * 5
            ending[KING, square] = centre * 6
    middle[KING, 6] += 42
    middle[KING, 2] += 34
    return middle, ending


def _splitmix64(seed: int) -> tuple[int, int]:
    seed = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = seed
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return seed, (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


def _zobrist_tables() -> tuple[np.ndarray, np.uint64, np.ndarray, np.ndarray]:
    seed = 0xC0DEC0FFEE123456
    pieces = np.zeros((12, 128), dtype=np.uint64)
    for piece in range(12):
        for square in range(128):
            seed, value = _splitmix64(seed)
            pieces[piece, square] = value
    seed, side = _splitmix64(seed)
    castles = np.zeros(16, dtype=np.uint64)
    for index in range(16):
        seed, value = _splitmix64(seed)
        castles[index] = value
    ep_files = np.zeros(8, dtype=np.uint64)
    for index in range(8):
        seed, value = _splitmix64(seed)
        ep_files[index] = value
    return pieces, np.uint64(side), castles, ep_files


PST_MG, PST_EG = _piece_square_tables()
Z_PIECES, Z_SIDE, Z_CASTLES, Z_EP_FILES = _zobrist_tables()

_tt_keys = np.zeros(TT_SIZE, dtype=np.uint64)
_tt_scores = np.zeros(TT_SIZE, dtype=np.int32)
_tt_moves = np.zeros(TT_SIZE, dtype=np.int32)
_tt_depths = np.full(TT_SIZE, -1, dtype=np.int8)
_tt_bounds = np.zeros(TT_SIZE, dtype=np.int8)
_tt_ages = np.zeros(TT_SIZE, dtype=np.uint16)
_history = np.zeros((2, 128, 128), dtype=np.int32)
_killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
_move_buffer = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
_score_buffer = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
_hash_stack = np.zeros(MAX_PLY, dtype=np.uint64)
_nodes = np.zeros(3, dtype=np.int64)
_generation = 1
_ponder_thread: threading.Thread | None = None
_ponder_stop = threading.Event()


@njit(cache=False, inline="always")
def _encode_move(from_square: int, to_square: int, promotion: int, flags: int) -> int:
    return from_square | (to_square << 7) | (promotion << 14) | flags


@njit(cache=False, inline="always")
def _from_square(move: int) -> int:
    return move & 127


@njit(cache=False, inline="always")
def _to_square(move: int) -> int:
    return (move >> 7) & 127


@njit(cache=False, inline="always")
def _promotion(move: int) -> int:
    return (move >> 14) & 7


@njit(cache=False, inline="always")
def _on_board(square: int) -> bool:
    return square >= 0 and square < 128 and (square & 0x88) == 0


@njit(cache=False, inline="always")
def _piece_index(piece: int) -> int:
    return piece - 1 if piece > 0 else 6 + (-piece - 1)


@njit(cache=False)
def _position_hash(board: np.ndarray, side: int, castling: int, ep_square: int) -> np.uint64:
    key = np.uint64(0)
    for square in range(128):
        if square & 0x88:
            continue
        piece = int(board[square])
        if piece:
            key ^= Z_PIECES[_piece_index(piece), square]
    if side == BLACK:
        key ^= Z_SIDE
    key ^= Z_CASTLES[castling]
    if ep_square >= 0:
        key ^= Z_EP_FILES[ep_square & 7]
    return key


@njit(cache=False)
def _is_attacked(board: np.ndarray, square: int, attacker: int) -> bool:
    if attacker == WHITE:
        origin = square - 15
        if _on_board(origin) and board[origin] == PAWN:
            return True
        origin = square - 17
        if _on_board(origin) and board[origin] == PAWN:
            return True
    else:
        origin = square + 15
        if _on_board(origin) and board[origin] == -PAWN:
            return True
        origin = square + 17
        if _on_board(origin) and board[origin] == -PAWN:
            return True

    for offset in KNIGHT_OFFSETS:
        origin = square + int(offset)
        if _on_board(origin) and board[origin] == attacker * KNIGHT:
            return True
    for offset in KING_OFFSETS:
        origin = square + int(offset)
        if _on_board(origin) and board[origin] == attacker * KING:
            return True
    for direction in BISHOP_DIRECTIONS:
        origin = square + int(direction)
        while _on_board(origin):
            piece = int(board[origin])
            if piece:
                if piece == attacker * BISHOP or piece == attacker * QUEEN:
                    return True
                break
            origin += int(direction)
    for direction in ROOK_DIRECTIONS:
        origin = square + int(direction)
        while _on_board(origin):
            piece = int(board[origin])
            if piece:
                if piece == attacker * ROOK or piece == attacker * QUEEN:
                    return True
                break
            origin += int(direction)
    return False


@njit(cache=False, inline="always")
def _add_move(row: np.ndarray, count: int, move: int) -> int:
    row[count] = move
    return count + 1


@njit(cache=False)
def _generate_pseudo(
    board: np.ndarray,
    side: int,
    castling: int,
    ep_square: int,
    king_square: int,
    row: np.ndarray,
) -> int:
    count = 0
    for square in range(128):
        if square & 0x88:
            continue
        piece = int(board[square])
        if piece * side <= 0:
            continue
        piece_type = abs(piece)

        if piece_type == PAWN:
            step = 16 * side
            target = square + step
            rank_index = square >> 4
            promotion_rank = 6 if side == WHITE else 1
            start_rank = 1 if side == WHITE else 6
            if _on_board(target) and board[target] == EMPTY:
                if rank_index == promotion_rank:
                    for promoted in (QUEEN, ROOK, BISHOP, KNIGHT):
                        count = _add_move(row, count, _encode_move(square, target, promoted, 0))
                else:
                    count = _add_move(row, count, _encode_move(square, target, 0, 0))
                    double_target = square + 2 * step
                    if rank_index == start_rank and board[double_target] == EMPTY:
                        count = _add_move(
                            row, count, _encode_move(square, double_target, 0, FLAG_DOUBLE)
                        )
            for delta in (15 * side, 17 * side):
                target = square + delta
                if not _on_board(target):
                    continue
                if board[target] * side < 0 or target == ep_square:
                    flag = FLAG_EP if target == ep_square and board[target] == EMPTY else 0
                    if rank_index == promotion_rank:
                        for promoted in (QUEEN, ROOK, BISHOP, KNIGHT):
                            count = _add_move(
                                row, count, _encode_move(square, target, promoted, flag)
                            )
                    else:
                        count = _add_move(row, count, _encode_move(square, target, 0, flag))

        elif piece_type == KNIGHT:
            for offset in KNIGHT_OFFSETS:
                target = square + int(offset)
                if _on_board(target) and board[target] * side <= 0:
                    count = _add_move(row, count, _encode_move(square, target, 0, 0))

        elif piece_type in (BISHOP, ROOK, QUEEN):
            if piece_type == BISHOP:
                direction_start, direction_end = 0, 4
            elif piece_type == ROOK:
                direction_start, direction_end = 4, 8
            else:
                direction_start, direction_end = 0, 8
            for direction_index in range(direction_start, direction_end):
                if direction_index < 4:
                    direction = int(BISHOP_DIRECTIONS[direction_index])
                else:
                    direction = int(ROOK_DIRECTIONS[direction_index - 4])
                target = square + direction
                while _on_board(target):
                    occupant = int(board[target])
                    if occupant * side > 0:
                        break
                    count = _add_move(row, count, _encode_move(square, target, 0, 0))
                    if occupant:
                        break
                    target += direction

        elif piece_type == KING:
            for offset in KING_OFFSETS:
                target = square + int(offset)
                if _on_board(target) and board[target] * side <= 0:
                    count = _add_move(row, count, _encode_move(square, target, 0, 0))

    enemy = -side
    if side == WHITE and king_square == 4:
        if (
            castling & CASTLE_WK
            and board[7] == ROOK
            and board[5] == EMPTY
            and board[6] == EMPTY
            and not _is_attacked(board, 4, enemy)
            and not _is_attacked(board, 5, enemy)
            and not _is_attacked(board, 6, enemy)
        ):
            count = _add_move(row, count, _encode_move(4, 6, 0, FLAG_CASTLE))
        if (
            castling & CASTLE_WQ
            and board[0] == ROOK
            and board[1] == EMPTY
            and board[2] == EMPTY
            and board[3] == EMPTY
            and not _is_attacked(board, 4, enemy)
            and not _is_attacked(board, 3, enemy)
            and not _is_attacked(board, 2, enemy)
        ):
            count = _add_move(row, count, _encode_move(4, 2, 0, FLAG_CASTLE))
    elif side == BLACK and king_square == 116:
        if (
            castling & CASTLE_BK
            and board[119] == -ROOK
            and board[117] == EMPTY
            and board[118] == EMPTY
            and not _is_attacked(board, 116, enemy)
            and not _is_attacked(board, 117, enemy)
            and not _is_attacked(board, 118, enemy)
        ):
            count = _add_move(row, count, _encode_move(116, 118, 0, FLAG_CASTLE))
        if (
            castling & CASTLE_BQ
            and board[112] == -ROOK
            and board[113] == EMPTY
            and board[114] == EMPTY
            and board[115] == EMPTY
            and not _is_attacked(board, 116, enemy)
            and not _is_attacked(board, 115, enemy)
            and not _is_attacked(board, 114, enemy)
        ):
            count = _add_move(row, count, _encode_move(116, 114, 0, FLAG_CASTLE))
    return count


@njit(cache=False)
def _make_move(
    board: np.ndarray,
    move: int,
    side: int,
    castling: int,
    halfmove: int,
    king_square: int,
) -> tuple[int, int, int, int, int]:
    origin = _from_square(move)
    target = _to_square(move)
    promoted = _promotion(move)
    piece = int(board[origin])
    captured = int(board[target])
    new_castling = castling

    if abs(piece) == KING:
        new_castling &= ~(CASTLE_WK | CASTLE_WQ) if side == WHITE else ~(
            CASTLE_BK | CASTLE_BQ
        )
    elif abs(piece) == ROOK:
        if origin == 0:
            new_castling &= ~CASTLE_WQ
        elif origin == 7:
            new_castling &= ~CASTLE_WK
        elif origin == 112:
            new_castling &= ~CASTLE_BQ
        elif origin == 119:
            new_castling &= ~CASTLE_BK
    if captured == ROOK:
        if target == 0:
            new_castling &= ~CASTLE_WQ
        elif target == 7:
            new_castling &= ~CASTLE_WK
    elif captured == -ROOK:
        if target == 112:
            new_castling &= ~CASTLE_BQ
        elif target == 119:
            new_castling &= ~CASTLE_BK

    if move & FLAG_EP:
        captured_square = target - 16 * side
        captured = int(board[captured_square])
        board[captured_square] = EMPTY

    board[origin] = EMPTY
    board[target] = side * promoted if promoted else piece
    if move & FLAG_CASTLE:
        if target > origin:
            board[origin + 1] = board[origin + 3]
            board[origin + 3] = EMPTY
        else:
            board[origin - 1] = board[origin - 4]
            board[origin - 4] = EMPTY

    new_ep = origin + 16 * side if move & FLAG_DOUBLE else -1
    new_halfmove = 0 if abs(piece) == PAWN or captured else halfmove + 1
    new_king = target if abs(piece) == KING else king_square
    return captured, new_castling, new_ep, new_halfmove, new_king


@njit(cache=False)
def _undo_move(board: np.ndarray, move: int, side: int, captured: int) -> None:
    origin = _from_square(move)
    target = _to_square(move)
    promoted = _promotion(move)
    moved_piece = int(board[target])
    if move & FLAG_CASTLE:
        if target > origin:
            board[origin + 3] = board[origin + 1]
            board[origin + 1] = EMPTY
        else:
            board[origin - 4] = board[origin - 1]
            board[origin - 1] = EMPTY
    board[origin] = side * PAWN if promoted else moved_piece
    if move & FLAG_EP:
        board[target] = EMPTY
        board[target - 16 * side] = captured
    else:
        board[target] = captured


@njit(cache=False)
def _generate_legal(
    board: np.ndarray,
    side: int,
    castling: int,
    ep_square: int,
    halfmove: int,
    king_square: int,
    ply: int,
    move_buffer: np.ndarray,
) -> int:
    row = move_buffer[ply]
    pseudo_count = _generate_pseudo(board, side, castling, ep_square, king_square, row)
    legal_count = 0
    for index in range(pseudo_count):
        move = int(row[index])
        captured, _, _, _, new_king = _make_move(
            board, move, side, castling, halfmove, king_square
        )
        legal = not _is_attacked(board, new_king, -side)
        _undo_move(board, move, side, captured)
        if legal:
            row[legal_count] = move
            legal_count += 1
    return legal_count


@njit(cache=False)
def _mobility(board: np.ndarray, square: int, side: int, piece_type: int) -> int:
    count = 0
    if piece_type == KNIGHT:
        for offset in KNIGHT_OFFSETS:
            target = square + int(offset)
            if _on_board(target) and board[target] * side <= 0:
                count += 1
    else:
        if piece_type == BISHOP:
            direction_start, direction_end = 0, 4
        elif piece_type == ROOK:
            direction_start, direction_end = 4, 8
        else:
            direction_start, direction_end = 0, 8
        for direction_index in range(direction_start, direction_end):
            if direction_index < 4:
                direction = int(BISHOP_DIRECTIONS[direction_index])
            else:
                direction = int(ROOK_DIRECTIONS[direction_index - 4])
            target = square + direction
            while _on_board(target):
                if board[target] * side > 0:
                    break
                count += 1
                if board[target]:
                    break
                target += direction
    return count


@njit(cache=False)
def _evaluate(board: np.ndarray, side: int, white_king: int, black_king: int) -> int:
    mg = 0
    eg = 0
    phase = 0
    white_bishops = 0
    black_bishops = 0

    for square in range(128):
        if square & 0x88:
            continue
        piece = int(board[square])
        if not piece:
            continue
        color = WHITE if piece > 0 else BLACK
        piece_type = abs(piece)
        rank_index = square >> 4
        file_index = square & 7
        relative = square if color == WHITE else (7 - rank_index) * 16 + file_index
        mg += color * (int(MG_VALUE[piece_type]) + int(PST_MG[piece_type, relative]))
        eg += color * (int(EG_VALUE[piece_type]) + int(PST_EG[piece_type, relative]))
        phase += int(PHASE_VALUE[piece_type])

        if piece_type == BISHOP:
            if color == WHITE:
                white_bishops += 1
            else:
                black_bishops += 1
        if piece_type in (KNIGHT, BISHOP, ROOK, QUEEN):
            mobility = _mobility(board, square, color, piece_type)
            mg += color * int(MOBILITY_MG[piece_type]) * mobility
            eg += color * int(MOBILITY_EG[piece_type]) * mobility

    if white_bishops >= 2:
        mg += 31
        eg += 42
    if black_bishops >= 2:
        mg -= 31
        eg -= 42

    for color in (WHITE, BLACK):
        for file_index in range(8):
            pawn_count = 0
            for rank_index in range(8):
                if board[rank_index * 16 + file_index] == color * PAWN:
                    pawn_count += 1
            if pawn_count > 1:
                mg -= color * 13 * (pawn_count - 1)
                eg -= color * 19 * (pawn_count - 1)

        for square in range(128):
            if square & 0x88 or board[square] != color * PAWN:
                continue
            file_index = square & 7
            rank_index = square >> 4
            relative_rank = rank_index if color == WHITE else 7 - rank_index
            isolated = True
            for nearby_file in range(max(0, file_index - 1), min(7, file_index + 1) + 1):
                if nearby_file == file_index:
                    continue
                for scan_rank in range(8):
                    if board[scan_rank * 16 + nearby_file] == color * PAWN:
                        isolated = False
                        break
                if not isolated:
                    break
            if isolated:
                mg -= color * 11
                eg -= color * 9

            passed = True
            first_rank = rank_index + 1 if color == WHITE else rank_index - 1
            last_rank = 8 if color == WHITE else -1
            rank_step = 1 if color == WHITE else -1
            scan_rank = first_rank
            while scan_rank != last_rank and passed:
                for nearby_file in range(max(0, file_index - 1), min(7, file_index + 1) + 1):
                    if board[scan_rank * 16 + nearby_file] == -color * PAWN:
                        passed = False
                        break
                scan_rank += rank_step
            if passed:
                mg += color * int(PASSED_MG[relative_rank])
                eg += color * int(PASSED_EG[relative_rank])

            support_rank = rank_index - 1 if color == WHITE else rank_index + 1
            if 0 <= support_rank < 8:
                supported = False
                if file_index > 0 and board[support_rank * 16 + file_index - 1] == color * PAWN:
                    supported = True
                if file_index < 7 and board[support_rank * 16 + file_index + 1] == color * PAWN:
                    supported = True
                if supported:
                    mg += color * 7
                    eg += color * 10

        for square in range(128):
            if square & 0x88 or board[square] != color * ROOK:
                continue
            file_index = square & 7
            friendly_pawn = False
            any_pawn = False
            for rank_index in range(8):
                occupant = int(board[rank_index * 16 + file_index])
                if abs(occupant) == PAWN:
                    any_pawn = True
                    if occupant == color * PAWN:
                        friendly_pawn = True
            if not any_pawn:
                mg += color * 20
                eg += color * 14
            elif not friendly_pawn:
                mg += color * 11
                eg += color * 7

        king_square = white_king if color == WHITE else black_king
        king_file = king_square & 7
        king_rank = king_square >> 4
        shield = 0
        for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
            for distance in (1, 2):
                shield_rank = king_rank + color * distance
                if 0 <= shield_rank < 8 and board[shield_rank * 16 + file_index] == color * PAWN:
                    shield += 12 if distance == 1 else 5
        mg += color * shield

        if phase >= 8:
            pressure = 0
            for offset in KING_OFFSETS:
                target = king_square + int(offset)
                if _on_board(target) and _is_attacked(board, target, -color):
                    pressure += 1
            mg -= color * pressure * 8

    phase = min(24, phase)
    white_score = (mg * phase + eg * (24 - phase)) // 24
    return (white_score if side == WHITE else -white_score) + 10


@njit(cache=False)
def _insufficient_material(board: np.ndarray) -> bool:
    minors = 0
    for square in range(128):
        if square & 0x88:
            continue
        piece_type = abs(int(board[square]))
        if piece_type in (PAWN, ROOK, QUEEN):
            return False
        if piece_type in (KNIGHT, BISHOP):
            minors += 1
            if minors > 1:
                return False
    return True


@njit(cache=False, inline="always")
def _capture_value(board: np.ndarray, move: int, side: int) -> int:
    if move & FLAG_EP:
        return int(MG_VALUE[PAWN])
    captured = abs(int(board[_to_square(move)]))
    return int(MG_VALUE[captured]) if captured else 0


@njit(cache=False)
def _order_moves(
    board: np.ndarray,
    side: int,
    count: int,
    tt_move: int,
    ply: int,
    move_buffer: np.ndarray,
    score_buffer: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
) -> None:
    moves = move_buffer[ply]
    scores = score_buffer[ply]
    side_index = 0 if side == WHITE else 1
    for index in range(count):
        move = int(moves[index])
        origin = _from_square(move)
        target = _to_square(move)
        promoted = _promotion(move)
        if move == tt_move:
            score = 30_000_000
        elif promoted:
            score = 20_000_000 + (int(MG_VALUE[promoted]) - int(MG_VALUE[PAWN])) * 1_000
        else:
            victim_value = _capture_value(board, move, side)
            if victim_value:
                attacker = abs(int(board[origin]))
                score = 18_000_000 + victim_value * 100 - int(MG_VALUE[attacker])
            elif move == killers[ply, 0]:
                score = 15_000_000
            elif move == killers[ply, 1]:
                score = 14_000_000
            else:
                score = int(history[side_index, origin, target])
        scores[index] = score

    # Move lists are short; insertion sort avoids allocating a temporary array per node.
    for index in range(1, count):
        move = int(moves[index])
        score = int(scores[index])
        cursor = index - 1
        while cursor >= 0 and scores[cursor] < score:
            moves[cursor + 1] = moves[cursor]
            scores[cursor + 1] = scores[cursor]
            cursor -= 1
        moves[cursor + 1] = move
        scores[cursor + 1] = score


@njit(cache=False, inline="always")
def _from_tt(score: int, ply: int) -> int:
    if score > MATE_BOUND:
        return score - ply
    if score < -MATE_BOUND:
        return score + ply
    return score


@njit(cache=False, inline="always")
def _to_tt(score: int, ply: int) -> int:
    if score > MATE_BOUND:
        return score + ply
    if score < -MATE_BOUND:
        return score - ply
    return score


@njit(cache=False)
def _is_repeated(hash_stack: np.ndarray, key: np.uint64, ply: int, halfmove: int) -> bool:
    if ply < 4 or halfmove < 4:
        return False
    earliest = max(0, ply - halfmove)
    index = ply - 2
    while index >= earliest:
        if hash_stack[index] == key:
            return True
        index -= 2
    return False


@njit(cache=False, inline="always")
def _out_of_nodes(nodes: np.ndarray) -> bool:
    nodes[0] += 1
    return nodes[0] >= nodes[1]


@njit(cache=False)
def _quiescence(
    board: np.ndarray,
    side: int,
    castling: int,
    ep_square: int,
    halfmove: int,
    king_square: int,
    opponent_king: int,
    alpha: int,
    beta: int,
    ply: int,
    generation: int,
    tt_keys: np.ndarray,
    tt_scores: np.ndarray,
    tt_moves: np.ndarray,
    tt_depths: np.ndarray,
    tt_bounds: np.ndarray,
    tt_ages: np.ndarray,
    move_buffer: np.ndarray,
    score_buffer: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    hash_stack: np.ndarray,
    nodes: np.ndarray,
) -> int:
    if _out_of_nodes(nodes):
        return ABORT
    if ply >= MAX_PLY - 1:
        white_king = king_square if side == WHITE else opponent_king
        black_king = opponent_king if side == WHITE else king_square
        return _evaluate(board, side, white_king, black_king)
    key = _position_hash(board, side, castling, ep_square)
    if halfmove >= 100 or _is_repeated(hash_stack, key, ply, halfmove):
        return 0
    hash_stack[ply] = key

    in_check = _is_attacked(board, king_square, -side)
    white_king = king_square if side == WHITE else opponent_king
    black_king = opponent_king if side == WHITE else king_square
    stand_pat = _evaluate(board, side, white_king, black_king)
    if not in_check:
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

    count = _generate_legal(
        board, side, castling, ep_square, halfmove, king_square, ply, move_buffer
    )
    if count == 0:
        return -MATE + ply if in_check else 0
    _order_moves(
        board, side, count, 0, ply, move_buffer, score_buffer, killers, history
    )

    for index in range(count):
        move = int(move_buffer[ply, index])
        promoted = _promotion(move)
        capture_value = _capture_value(board, move, side)
        if not in_check and not promoted and not capture_value:
            continue
        if not in_check and not promoted and stand_pat + capture_value + 180 < alpha:
            continue
        captured, new_castling, new_ep, new_halfmove, new_king = _make_move(
            board, move, side, castling, halfmove, king_square
        )
        child = _quiescence(
            board,
            -side,
            new_castling,
            new_ep,
            new_halfmove,
            opponent_king,
            new_king,
            -beta,
            -alpha,
            ply + 1,
            generation,
            tt_keys,
            tt_scores,
            tt_moves,
            tt_depths,
            tt_bounds,
            tt_ages,
            move_buffer,
            score_buffer,
            killers,
            history,
            hash_stack,
            nodes,
        )
        _undo_move(board, move, side, captured)
        if child == ABORT:
            return ABORT
        score = -child
        if score >= beta:
            return score
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def _search(
    board: np.ndarray,
    side: int,
    castling: int,
    ep_square: int,
    halfmove: int,
    king_square: int,
    opponent_king: int,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    generation: int,
    tt_keys: np.ndarray,
    tt_scores: np.ndarray,
    tt_moves: np.ndarray,
    tt_depths: np.ndarray,
    tt_bounds: np.ndarray,
    tt_ages: np.ndarray,
    move_buffer: np.ndarray,
    score_buffer: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    hash_stack: np.ndarray,
    nodes: np.ndarray,
) -> int:
    if _out_of_nodes(nodes):
        return ABORT
    if ply >= MAX_PLY - 1:
        white_king = king_square if side == WHITE else opponent_king
        black_king = opponent_king if side == WHITE else king_square
        return _evaluate(board, side, white_king, black_king)
    key = _position_hash(board, side, castling, ep_square)
    if halfmove >= 100 or _is_repeated(hash_stack, key, ply, halfmove):
        return 0
    hash_stack[ply] = key
    if depth <= 0:
        return _quiescence(
            board,
            side,
            castling,
            ep_square,
            halfmove,
            king_square,
            opponent_king,
            alpha,
            beta,
            ply,
            generation,
            tt_keys,
            tt_scores,
            tt_moves,
            tt_depths,
            tt_bounds,
            tt_ages,
            move_buffer,
            score_buffer,
            killers,
            history,
            hash_stack,
            nodes,
        )

    tt_index = int(key & np.uint64(TT_MASK))
    tt_move = 0
    if tt_ages[tt_index] and tt_keys[tt_index] == key:
        tt_move = int(tt_moves[tt_index])
        if int(tt_depths[tt_index]) >= depth:
            tt_score = _from_tt(int(tt_scores[tt_index]), ply)
            bound = int(tt_bounds[tt_index])
            if bound == EXACT:
                return tt_score
            if bound == LOWER:
                alpha = max(alpha, tt_score)
            else:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

    original_alpha = alpha
    in_check = _is_attacked(board, king_square, -side)
    count = _generate_legal(
        board, side, castling, ep_square, halfmove, king_square, ply, move_buffer
    )
    if count == 0:
        return -MATE + ply if in_check else 0
    if _insufficient_material(board):
        return 0

    white_king = king_square if side == WHITE else opponent_king
    black_king = opponent_king if side == WHITE else king_square
    static_score = _evaluate(board, side, white_king, black_king)
    if not in_check and depth <= 3 and static_score - 95 * depth >= beta:
        return static_score
    if not in_check and depth == 1 and static_score + 170 <= alpha:
        razor = _quiescence(
            board,
            side,
            castling,
            ep_square,
            halfmove,
            king_square,
            opponent_king,
            alpha,
            beta,
            ply,
            generation,
            tt_keys,
            tt_scores,
            tt_moves,
            tt_depths,
            tt_bounds,
            tt_ages,
            move_buffer,
            score_buffer,
            killers,
            history,
            hash_stack,
            nodes,
        )
        if razor == ABORT or razor <= alpha:
            return razor

    has_non_pawn = False
    if depth >= 3 and not in_check and static_score >= beta:
        for square in range(128):
            if square & 0x88:
                continue
            piece = int(board[square])
            if piece * side > 0 and abs(piece) in (KNIGHT, BISHOP, ROOK, QUEEN):
                has_non_pawn = True
                break
    if has_non_pawn:
        reduction = 2 + depth // 4
        null_child = _search(
            board,
            -side,
            castling,
            -1,
            0,
            opponent_king,
            king_square,
            depth - 1 - reduction,
            -beta,
            -beta + 1,
            ply + 1,
            generation,
            tt_keys,
            tt_scores,
            tt_moves,
            tt_depths,
            tt_bounds,
            tt_ages,
            move_buffer,
            score_buffer,
            killers,
            history,
            hash_stack,
            nodes,
        )
        if null_child == ABORT:
            return ABORT
        if -null_child >= beta:
            return -null_child

    _order_moves(
        board, side, count, tt_move, ply, move_buffer, score_buffer, killers, history
    )
    best_score = -INF
    best_move = 0
    side_index = 0 if side == WHITE else 1
    for move_index in range(count):
        move = int(move_buffer[ply, move_index])
        capture = _capture_value(board, move, side) > 0
        quiet = not capture and _promotion(move) == 0
        captured, new_castling, new_ep, new_halfmove, new_king = _make_move(
            board, move, side, castling, halfmove, king_square
        )
        gives_check = _is_attacked(board, opponent_king, side)
        extension = 1 if in_check and depth >= 2 and ply < 12 else 0
        next_depth = depth - 1 + extension

        if move_index == 0:
            child = _search(
                board,
                -side,
                new_castling,
                new_ep,
                new_halfmove,
                opponent_king,
                new_king,
                next_depth,
                -beta,
                -alpha,
                ply + 1,
                generation,
                tt_keys,
                tt_scores,
                tt_moves,
                tt_depths,
                tt_bounds,
                tt_ages,
                move_buffer,
                score_buffer,
                killers,
                history,
                hash_stack,
                nodes,
            )
            if child == ABORT:
                _undo_move(board, move, side, captured)
                return ABORT
            score = -child
        else:
            reduced_depth = next_depth
            if depth >= 3 and move_index >= 4 and quiet and not in_check and not gives_check:
                reduction = 1 + int(depth >= 6) + int(move_index >= 10)
                reduced_depth = max(0, next_depth - reduction)
            child = _search(
                board,
                -side,
                new_castling,
                new_ep,
                new_halfmove,
                opponent_king,
                new_king,
                reduced_depth,
                -alpha - 1,
                -alpha,
                ply + 1,
                generation,
                tt_keys,
                tt_scores,
                tt_moves,
                tt_depths,
                tt_bounds,
                tt_ages,
                move_buffer,
                score_buffer,
                killers,
                history,
                hash_stack,
                nodes,
            )
            if child == ABORT:
                _undo_move(board, move, side, captured)
                return ABORT
            score = -child
            if score > alpha and reduced_depth < next_depth:
                child = _search(
                    board,
                    -side,
                    new_castling,
                    new_ep,
                    new_halfmove,
                    opponent_king,
                    new_king,
                    next_depth,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    generation,
                    tt_keys,
                    tt_scores,
                    tt_moves,
                    tt_depths,
                    tt_bounds,
                    tt_ages,
                    move_buffer,
                    score_buffer,
                    killers,
                    history,
                    hash_stack,
                    nodes,
                )
                if child == ABORT:
                    _undo_move(board, move, side, captured)
                    return ABORT
                score = -child
            if score > alpha and score < beta:
                child = _search(
                    board,
                    -side,
                    new_castling,
                    new_ep,
                    new_halfmove,
                    opponent_king,
                    new_king,
                    next_depth,
                    -beta,
                    -alpha,
                    ply + 1,
                    generation,
                    tt_keys,
                    tt_scores,
                    tt_moves,
                    tt_depths,
                    tt_bounds,
                    tt_ages,
                    move_buffer,
                    score_buffer,
                    killers,
                    history,
                    hash_stack,
                    nodes,
                )
                if child == ABORT:
                    _undo_move(board, move, side, captured)
                    return ABORT
                score = -child
        _undo_move(board, move, side, captured)

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if quiet:
                if move != killers[ply, 0]:
                    killers[ply, 1] = killers[ply, 0]
                    killers[ply, 0] = move
                origin = _from_square(move)
                target = _to_square(move)
                history[side_index, origin, target] = min(
                    1_000_000, history[side_index, origin, target] + depth * depth
                )
            break

    bound = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
    if tt_ages[tt_index] != generation or depth >= int(tt_depths[tt_index]) - 1:
        tt_keys[tt_index] = key
        tt_scores[tt_index] = _to_tt(best_score, ply)
        tt_moves[tt_index] = best_move
        tt_depths[tt_index] = depth
        tt_bounds[tt_index] = bound
        tt_ages[tt_index] = generation
    return best_score


@njit(cache=False, nogil=True)
def _root_search(
    board: np.ndarray,
    side: int,
    castling: int,
    ep_square: int,
    halfmove: int,
    king_square: int,
    opponent_king: int,
    depth: int,
    alpha: int,
    beta: int,
    preferred: int,
    node_limit: int,
    generation: int,
    tt_keys: np.ndarray,
    tt_scores: np.ndarray,
    tt_moves: np.ndarray,
    tt_depths: np.ndarray,
    tt_bounds: np.ndarray,
    tt_ages: np.ndarray,
    move_buffer: np.ndarray,
    score_buffer: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    hash_stack: np.ndarray,
    nodes: np.ndarray,
) -> tuple[int, int, int, bool]:
    nodes[0] = 0
    nodes[1] = max(2, node_limit)
    if nodes[2] != 0:
        nodes[1] = 0
    key = _position_hash(board, side, castling, ep_square)
    hash_stack[0] = key
    count = _generate_legal(
        board, side, castling, ep_square, halfmove, king_square, 0, move_buffer
    )
    if count == 0:
        return 0, 0, int(nodes[0]), True
    tt_index = int(key & np.uint64(TT_MASK))
    tt_move = preferred
    if not tt_move and tt_ages[tt_index] and tt_keys[tt_index] == key:
        tt_move = int(tt_moves[tt_index])
    _order_moves(
        board, side, count, tt_move, 0, move_buffer, score_buffer, killers, history
    )

    original_alpha = alpha
    best_score = -INF
    best_move = int(move_buffer[0, 0])
    side_index = 0 if side == WHITE else 1
    for move_index in range(count):
        move = int(move_buffer[0, move_index])
        quiet = _capture_value(board, move, side) == 0 and _promotion(move) == 0
        captured, new_castling, new_ep, new_halfmove, new_king = _make_move(
            board, move, side, castling, halfmove, king_square
        )
        if move_index == 0:
            child = _search(
                board,
                -side,
                new_castling,
                new_ep,
                new_halfmove,
                opponent_king,
                new_king,
                depth - 1,
                -beta,
                -alpha,
                1,
                generation,
                tt_keys,
                tt_scores,
                tt_moves,
                tt_depths,
                tt_bounds,
                tt_ages,
                move_buffer,
                score_buffer,
                killers,
                history,
                hash_stack,
                nodes,
            )
        else:
            child = _search(
                board,
                -side,
                new_castling,
                new_ep,
                new_halfmove,
                opponent_king,
                new_king,
                depth - 1,
                -alpha - 1,
                -alpha,
                1,
                generation,
                tt_keys,
                tt_scores,
                tt_moves,
                tt_depths,
                tt_bounds,
                tt_ages,
                move_buffer,
                score_buffer,
                killers,
                history,
                hash_stack,
                nodes,
            )
            if child != ABORT:
                score = -child
                if score > alpha and score < beta:
                    child = _search(
                        board,
                        -side,
                        new_castling,
                        new_ep,
                        new_halfmove,
                        opponent_king,
                        new_king,
                        depth - 1,
                        -beta,
                        -alpha,
                        1,
                        generation,
                        tt_keys,
                        tt_scores,
                        tt_moves,
                        tt_depths,
                        tt_bounds,
                        tt_ages,
                        move_buffer,
                        score_buffer,
                        killers,
                        history,
                        hash_stack,
                        nodes,
                    )
        _undo_move(board, move, side, captured)
        if child == ABORT:
            return best_score, best_move, int(nodes[0]), False
        score = -child
        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if quiet:
                if move != killers[0, 0]:
                    killers[0, 1] = killers[0, 0]
                    killers[0, 0] = move
                origin = _from_square(move)
                target = _to_square(move)
                history[side_index, origin, target] = min(
                    1_000_000, history[side_index, origin, target] + depth * depth
                )
            break

    bound = UPPER if best_score <= original_alpha else LOWER if best_score >= beta else EXACT
    tt_keys[tt_index] = key
    tt_scores[tt_index] = _to_tt(best_score, 0)
    tt_moves[tt_index] = best_move
    tt_depths[tt_index] = depth
    tt_bounds[tt_index] = bound
    tt_ages[tt_index] = generation
    return best_score, best_move, int(nodes[0]), True


def _encode_position(board: chess.Board) -> tuple[np.ndarray, int, int, int, int, int, int]:
    encoded = np.zeros(128, dtype=np.int8)
    white_king = -1
    black_king = -1
    for square, piece in board.piece_map().items():
        encoded_square = chess.square_rank(square) * 16 + chess.square_file(square)
        color = WHITE if piece.color == chess.WHITE else BLACK
        encoded[encoded_square] = color * piece.piece_type
        if piece.piece_type == chess.KING:
            if color == WHITE:
                white_king = encoded_square
            else:
                black_king = encoded_square
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= CASTLE_WK
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= CASTLE_WQ
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= CASTLE_BK
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= CASTLE_BQ
    ep_square = -1
    if board.ep_square is not None:
        ep_square = chess.square_rank(board.ep_square) * 16 + chess.square_file(board.ep_square)
    side = WHITE if board.turn == chess.WHITE else BLACK
    king_square = white_king if side == WHITE else black_king
    opponent_king = black_king if side == WHITE else white_king
    return encoded, side, castling, ep_square, board.halfmove_clock, king_square, opponent_king


def _decode_move(move: int) -> chess.Move:
    origin_0x88 = move & 127
    target_0x88 = (move >> 7) & 127
    origin = (origin_0x88 >> 4) * 8 + (origin_0x88 & 7)
    target = (target_0x88 >> 4) * 8 + (target_0x88 & 7)
    promoted = (move >> 14) & 7
    return chess.Move(origin, target, promotion=promoted or None)


def legal_moves_for_test(fen: str) -> set[str]:
    """Expose the compiled generator for differential tests against python-chess."""
    board = chess.Board(fen)
    encoded, side, castling, ep_square, halfmove, king_square, _ = _encode_position(board)
    count = _generate_legal(
        encoded, side, castling, ep_square, halfmove, king_square, 0, _move_buffer
    )
    return {_decode_move(int(_move_buffer[0, index])).uci() for index in range(count)}


@njit(cache=False)
def _perft(
    board: np.ndarray,
    side: int,
    castling: int,
    ep_square: int,
    halfmove: int,
    king_square: int,
    opponent_king: int,
    depth: int,
    ply: int,
    move_buffer: np.ndarray,
) -> int:
    if depth == 0:
        return 1
    count = _generate_legal(
        board, side, castling, ep_square, halfmove, king_square, ply, move_buffer
    )
    if depth == 1:
        return count
    nodes = 0
    for index in range(count):
        move = int(move_buffer[ply, index])
        captured, new_castling, new_ep, new_halfmove, new_king = _make_move(
            board, move, side, castling, halfmove, king_square
        )
        nodes += _perft(
            board,
            -side,
            new_castling,
            new_ep,
            new_halfmove,
            opponent_king,
            new_king,
            depth - 1,
            ply + 1,
            move_buffer,
        )
        _undo_move(board, move, side, captured)
    return nodes


def perft_for_test(fen: str, depth: int) -> int:
    """Count legal leaf nodes to verify make/unmake as well as root generation."""
    board = chess.Board(fen)
    encoded, side, castling, ep_square, halfmove, king_square, opponent_king = _encode_position(
        board
    )
    return int(
        _perft(
            encoded,
            side,
            castling,
            ep_square,
            halfmove,
            king_square,
            opponent_king,
            depth,
            0,
            _move_buffer,
        )
    )


def _time_limits(time_left_ms: int) -> tuple[float, float]:
    remaining = max(0.001, time_left_ms / 1_000.0)
    if remaining < 1.0:
        soft = max(0.004, remaining * 0.05)
    elif remaining < 5.0:
        soft = 0.045 + remaining * 0.05
    else:
        soft = min(4.5, 0.10 + remaining / 34.0)
    hard = min(max(0.008, remaining - 0.07), soft * 1.65)
    return soft, max(soft, hard)


def _stop_pondering() -> None:
    """Stop the idle-time search before touching its shared search tables."""
    global _ponder_thread
    thread = _ponder_thread
    if thread is None:
        return
    _ponder_stop.set()
    _nodes[2] = 1
    _nodes[1] = 0
    thread.join()
    _ponder_thread = None
    _nodes[2] = 0


def _ponder_position(fen: str, generation: int) -> None:
    """Deepen the likely reply position while the opponent owns the clock."""
    try:
        board = chess.Board(fen)
        encoded, side, castling, ep_square, halfmove, king_square, opponent_king = (
            _encode_position(board)
        )
        preferred = 0
        for depth in range(1, 64):
            if _ponder_stop.is_set():
                break
            _, preferred, _, completed = _root_search(
                encoded,
                side,
                castling,
                ep_square,
                halfmove,
                king_square,
                opponent_king,
                depth,
                -INF,
                INF,
                preferred,
                2_000_000_000,
                generation,
                _tt_keys,
                _tt_scores,
                _tt_moves,
                _tt_depths,
                _tt_bounds,
                _tt_ages,
                _move_buffer,
                _score_buffer,
                _killers,
                _history,
                _hash_stack,
                _nodes,
            )
            if not completed:
                break
    except Exception:
        # Pondering is optional strength work; it must never endanger a legal reply.
        return


def _start_pondering(board: chess.Board, generation: int) -> None:
    global _ponder_thread
    if board.is_game_over(claim_draw=True):
        return
    _ponder_stop.clear()
    _nodes[2] = 0
    _ponder_thread = threading.Thread(
        target=_ponder_position,
        args=(board.fen(), generation),
        name="chess-ponder",
        daemon=True,
    )
    _ponder_thread.start()


def choose_move(fen: str, time_left_ms: int) -> str:
    """Search a FEN with iterative deepening and a conservative node/time budget."""
    global _generation
    _stop_pondering()
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        candidate = legal[0]
        board.push(candidate)
        _start_pondering(board, _generation)
        return candidate.uci()

    encoded, side, castling, ep_square, halfmove, king_square, opponent_king = _encode_position(
        board
    )
    _generation = (_generation + 1) & 0xFFFF
    if _generation == 0:
        _generation = 1
        _tt_ages.fill(0)
    generation = _generation

    start = time.perf_counter()
    soft_seconds, hard_seconds = _time_limits(time_left_ms)
    soft_deadline = start + soft_seconds
    hard_deadline = start + hard_seconds
    best_move = 0
    best_score = 0
    measured_nps = 250_000.0
    previous_elapsed = 0.0

    for depth in range(1, 64):
        now = time.perf_counter()
        remaining_hard = hard_deadline - now
        if remaining_hard <= 0.002:
            break
        if depth > 1 and previous_elapsed * 2.3 >= remaining_hard:
            break
        node_limit = max(2_000, int(measured_nps * remaining_hard * 0.70))
        if depth <= 2:
            node_limit = max(node_limit, 100_000)

        window = 55 if depth >= 3 else INF
        alpha = max(-INF, best_score - window)
        beta = min(INF, best_score + window)
        iteration_started = time.perf_counter()
        iteration_nodes = 0
        completed = False
        while True:
            score, move, node_count, completed = _root_search(
                encoded,
                side,
                castling,
                ep_square,
                halfmove,
                king_square,
                opponent_king,
                depth,
                alpha,
                beta,
                best_move,
                node_limit,
                generation,
                _tt_keys,
                _tt_scores,
                _tt_moves,
                _tt_depths,
                _tt_bounds,
                _tt_ages,
                _move_buffer,
                _score_buffer,
                _killers,
                _history,
                _hash_stack,
                _nodes,
            )
            iteration_nodes += node_count
            if not completed:
                break
            if score <= alpha and alpha > -INF:
                alpha = max(-INF, alpha - max(140, (beta - alpha) * 2))
            elif score >= beta and beta < INF:
                beta = min(INF, beta + max(140, (beta - alpha) * 2))
            else:
                best_score, best_move = score, move
                break
            remaining_hard = hard_deadline - time.perf_counter()
            if remaining_hard <= 0.002:
                completed = False
                break
            node_limit = max(2_000, int(measured_nps * remaining_hard * 0.65))

        previous_elapsed = time.perf_counter() - iteration_started
        if previous_elapsed > 0.001 and iteration_nodes > 0:
            iteration_nps = iteration_nodes / previous_elapsed
            measured_nps = min(5_000_000.0, max(30_000.0, iteration_nps * 0.85))
        if not completed or abs(best_score) > MATE_BOUND:
            break
        if time.perf_counter() >= soft_deadline:
            break

    candidate = _decode_move(best_move) if best_move else legal[0]
    if candidate not in board.legal_moves:
        candidate = legal[0]
    board.push(candidate)
    _start_pondering(board, generation)
    return candidate.uci()


def reset_for_test() -> None:
    """Reset game-persistent state when a benchmark reuses one Python process for many games."""
    global _generation
    _stop_pondering()
    _generation = 1
    _tt_ages.fill(0)
    _history.fill(0)
    _killers.fill(0)


# Compile the complete recursive path during the platform's import budget, not on move one.
_warm_board = chess.Board()
_warm_encoded, _warm_side, _warm_castle, _warm_ep, _warm_half, _warm_king, _warm_other = (
    _encode_position(_warm_board)
)
_root_search(
    _warm_encoded,
    _warm_side,
    _warm_castle,
    _warm_ep,
    _warm_half,
    _warm_king,
    _warm_other,
    2,
    -INF,
    INF,
    0,
    20_000,
    65_535,
    _tt_keys,
    _tt_scores,
    _tt_moves,
    _tt_depths,
    _tt_bounds,
    _tt_ages,
    _move_buffer,
    _score_buffer,
    _killers,
    _history,
    _hash_stack,
    _nodes,
)
