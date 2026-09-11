"""Probe the compiled engine at fixed depths on rated-game regression positions."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import fast_engine  # noqa: E402

POSITIONS = (
    (
        "r33-16-castle",
        "r2q1r1k/1p2b1pp/p3Qn2/3p4/Pn6/N1N2P2/1PP3PP/R1B1K2R w KQ - 3 16",
        "e1g1",
    ),
    (
        "r33-24-queen-raid",
        "4rr1k/1p4pp/p7/2Qp4/P4q2/N1N2P1n/1PP3PP/R4KR1 w - - 5 24",
        "c3d5",
    ),
    (
        "r34-15-attack",
        "r2qr1k1/1p1bbpp1/p1np1n1p/4pP2/P3P1P1/1NN1B3/1PP1BR1P/R2Q2K1 w - - 0 15",
        "h2h4",
    ),
    (
        "r34-16-attack",
        "2rqr1k1/1p1bbpp1/p1np1n1p/4pP2/P1B1P1P1/1NN1B3/1PP2R1P/R2Q2K1 w - - 2 16",
        "g4g5",
    ),
    (
        "r34-18-attack",
        "2rqr1k1/1p2bpp1/p1bp1n1p/4pP2/PnB1P1P1/2N1B2P/1PPN1R2/R2Q2K1 w - - 3 18",
        "g4g5",
    ),
    (
        "r37-36-passed-pawn",
        "8/2R4p/bB4p1/P2pk3/4p2P/1r6/5PP1/6K1 b - - 4 36",
        "d5d4",
    ),
    (
        "r38-41-mate-defense",
        "8/4qpk1/1p4p1/1N2p2p/P1Q1P1P1/2P2P1P/3r4/5K2 w - - 13 41",
        "c4d5",
    ),
    (
        "r41-16-rook-defense",
        "r4rk1/1pp2pbp/2n3p1/3Np3/p1P1P1n1/P3BN1q/1P2BP2/RQ1R2K1 w - - 2 16",
        "d1d3",
    ),
    (
        "r41-18-rook-defense",
        "r4rk1/1pp2pbp/2n3p1/3Np3/pBP1P1n1/P4N1q/1P2BP2/RQ1R2K1 w - - 6 18",
        "d1d2",
    ),
    (
        "r41-20-rook-defense",
        "r3r1k1/1p3p1p/2n3pb/3Np3/pBP1P1n1/P4N1q/1P2BP2/RQ1R2K1 w - - 3 20",
        "d1d3",
    ),
    (
        "r41-22-rook-defense",
        "3rr1k1/1p3p1p/6pb/2BNp3/p1PnP1n1/P4N1q/1P2BP2/RQ1R2K1 w - - 7 22",
        "d1d4",
    ),
    (
        "r41-24-mate-defense",
        "4r1k1/1p3p1p/6pb/3r4/p1PRP1n1/P4N1q/1P2BP2/RQ4K1 w - - 0 24",
        "b1f1",
    ),
    (
        "r43-11-king-safety",
        "r2q1rk1/pp2ppbp/2n1b1p1/2pn2N1/7P/2NP2P1/PP1BPPB1/1R1QK2R b K - 3 11",
        "d5c3",
    ),
    (
        "r43-17-counterplay",
        "r2q1rk1/p3p2p/1p2p1p1/2p1n2P/4B3/2PPP1P1/P4P2/1R1QK2R b K - 0 17",
        "c5c4",
    ),
    (
        "r44-24-missed-exchange",
        "1r4k1/5ppp/2b5/pN1p4/P1pPr3/5K1P/1P3PP1/R3R3 b - - 1 24",
        "e4e1",
    ),
    (
        "r45-44-missed-conversion",
        "4r3/1R4p1/4bk2/3pN2P/2pP4/2P2P2/2P5/6K1 w - - 3 44",
        "g1f2",
    ),
    (
        "r46-27-piece-activity",
        "4r1k1/1prbbppp/pN1pp3/P7/3PP2P/1P4P1/3B4/3R1RK1 w - - 1 27",
        "f1c1",
    ),
    (
        "r46-34-passed-pawn-blockade",
        "4r1k1/1pbr2pp/p3p3/P3P3/3p3P/1PB3P1/5R2/2R3K1 w - - 0 34",
        "b4d4",
    ),
    (
        "r47-18-queenside-counterplay",
        "1r1q1rk1/3b2bp/3p2p1/p1pPpp1n/P3P3/2N2NP1/1PQ1BP1P/3RR1K1 b - - 1 18",
        "b8b4",
    ),
    (
        "r47-22-king-space",
        "5rk1/3b2bp/1q1p1np1/p1pPp3/P3p3/2N3P1/1P1QBP1P/3RR1K1 b - - 1 22",
        "h7h5",
    ),
    (
        "r48-47-endgame-king",
        "8/5kp1/pR6/P2n3p/1n1P3P/5KP1/6P1/8 w - - 12 47",
        "b6b7",
    ),
    (
        "r49-63-missed-conversion",
        "8/8/8/5p2/r1p2P1k/4K3/p7/R7 b - - 7 63",
        "a4a5",
    ),
    (
        "r50-20-king-attack",
        "r2r2k1/4bp2/1q2p1p1/1p1b3p/p1pP1Bn1/P1P2B2/1PQNRPPP/4R1K1 b - - 1 20",
        "g8g7",
    ),
    (
        "r50-23-king-attack",
        "r2r2k1/4bp2/1q2R3/1p4p1/p1pP2np/P1P2NB1/1PQ2PPP/4R1K1 b - - 0 23",
        "h4g3",
    ),
    (
        "r51-24-queen-check",
        "2rq1rk1/1p4p1/7p/p2pQ1b1/P5P1/1NN2P2/1Pn4P/3RR1K1 w - - 0 24",
        "e1e2",
    ),
    (
        "r52-24-tactical-liquidation",
        "3r2k1/1p2b1pp/pN1qb3/P4r2/3R1p2/2Q2P2/1P3BPP/R5K1 b - - 0 24",
        "d6b8",
    ),
    (
        "r53-24-exchange-sacrifice",
        "r5k1/1b3pp1/4p2p/pq2N3/8/P3P1P1/1P2P2P/2RQ2K1 w - - 1 24",
        "e5f7",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-depth", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument(
        "--contains",
        default="",
        help="only run probe names containing any comma-separated fragment",
    )
    args = parser.parse_args()
    fragments = tuple(fragment.strip() for fragment in args.contains.split(",") if fragment.strip())

    for name, fen, reference in POSITIONS:
        if fragments and not any(fragment in name for fragment in fragments):
            continue
        board = chess.Board(fen)
        print(f"\n{name} (SF19 {reference})")
        preferred = 0
        for depth in range(args.min_depth, args.max_depth + 1):
            fast_engine.reset_for_test()
            encoded = fast_engine._encode_position(board)
            started = time.perf_counter()
            score, move, nodes, completed = fast_engine._root_search(
                *encoded,
                depth,
                -fast_engine.INF,
                fast_engine.INF,
                preferred,
                200_000_000,
                depth,
                fast_engine._tt_keys,
                fast_engine._tt_scores,
                fast_engine._tt_moves,
                fast_engine._tt_depths,
                fast_engine._tt_bounds,
                fast_engine._tt_ages,
                fast_engine._move_buffer,
                fast_engine._score_buffer,
                fast_engine._killers,
                fast_engine._history,
                fast_engine._hash_stack,
                fast_engine._nodes,
            )
            elapsed = time.perf_counter() - started
            selected = fast_engine._decode_move(move).uci()
            preferred = move
            marker = "*" if selected == reference else " "
            print(
                f"  d{depth}: {selected} {score:+5d} {nodes:>9,} nodes "
                f"{elapsed:>6.2f}s {marker} complete={completed}"
            )


if __name__ == "__main__":
    main()
