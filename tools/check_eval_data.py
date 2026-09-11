"""Offline checks for first-PV selection and quiet-position training filters."""

from pathlib import Path
from tempfile import TemporaryDirectory

import chess
import pyarrow as pa
import pyarrow.parquet as pq
from build_lichess_eval_cache import (
    SelectedEvaluation,
    _is_quiet_label,
    _iter_parquet_records,
    _select_evaluation,
)


def main() -> None:
    chosen = _select_evaluation(
        {
            "evals": [
                {"depth": 20, "knodes": 500, "pvs": [{"cp": 90, "line": "d2d4 d7d5"}]},
                {
                    "depth": 24,
                    "knodes": 900,
                    "pvs": [
                        {"cp": 35, "line": "e2e4 e7e5"},
                        {"cp": 29, "line": "d2d4 d7d5"},
                    ],
                },
            ]
        }
    )
    assert chosen == SelectedEvaluation(35.0, 24, 900, "e2e4")
    assert _is_quiet_label(chess.Board(), chosen)
    for fen, move in (
        (chess.STARTING_FEN, None),
        (chess.STARTING_FEN, "invalid"),
        (chess.STARTING_FEN, "e2e5"),
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
        ("7k/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
        ("4r1k1/8/8/8/8/8/8/4K3 w - - 0 1", "e1d1"),
        ("7k/8/8/8/8/8/8/3QK3 w - - 0 1", "d1d8"),
    ):
        assert not _is_quiet_label(chess.Board(fen), SelectedEvaluation(0, 24, 1, move))
    assert (
        _select_evaluation(
            {
                "evals": [
                    {"depth": 20, "pvs": [{"cp": 10, "line": "e2e4"}]},
                    {"depth": 24, "pvs": [{"mate": 3, "line": "d2d4"}]},
                ]
            }
        )
        is None
    )

    after = chess.Board()
    after.push_uci("e2e4")
    rows = [
        {
            "fen": chess.STARTING_FEN,
            "depth": 20,
            "knodes": 500,
            "cp": 90,
            "mate": None,
            "line": "d2d4 d7d5",
        },
        {
            "fen": chess.STARTING_FEN,
            "depth": 24,
            "knodes": 900,
            "cp": 35,
            "mate": None,
            "line": "e2e4 e7e5",
        },
        {
            "fen": chess.STARTING_FEN,
            "depth": 24,
            "knodes": 900,
            "cp": 29,
            "mate": None,
            "line": "d2d4 d7d5",
        },
        {"fen": after.fen(), "depth": 20, "knodes": 300, "cp": 15, "mate": None, "line": "e7e5"},
        {"fen": after.fen(), "depth": 24, "knodes": 800, "cp": None, "mate": 3, "line": "e7e5"},
    ]
    with TemporaryDirectory(prefix="chessathon-filter-check-") as temporary:
        path = Path(temporary) / "sample.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        selected = list(_iter_parquet_records(path, 2, True))
        assert len(selected) == 2
        assert selected[0].evaluation == chosen
        assert selected[1].evaluation is None
        without_pv = list(_iter_parquet_records(path, 2))
        assert without_pv[0].evaluation == SelectedEvaluation(35, 24, 900)
    print("Passed JSON/Parquet deepest-PV selection, batch boundaries and quiet filters.")


if __name__ == "__main__":
    main()
