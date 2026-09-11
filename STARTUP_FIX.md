# v3.1 startup fix

## Final v3.1 full-clock result

The full-clock batch is COMPLETE: **16 wins, 4 draws, 4 losses in 24 games (75%)**.
Opening starts: +14 =1 -1. Unbooked middlegames: +2 =3 -3. No startup failures,
flags, crashes or illegal moves. The aggregate improvement is driven by opening preparation;
no middlegame improvement is established by the small unbooked sample. The earlier live
checkpoints below are historical. Both lanes have finished; do not wait for a running job.
See `docs/releases/v3.1/README.md` and its JSON reports for the final evidence.


The parallel v3 test stopped when the new engine exceeded its 90-second initialization
budget in both lanes. Profiling exposed avoidable compilation work: root calls passed
literal `1` for the search ply, and null-move calls passed literal `-1` for en-passant and
halfmove state. Numba compiled three versions each of `_search` and `_quiescence`, including
separate literal-specialized recursive call trees.

Five call-site arguments now use `np.int64(...)`. They retain their numerical values while
using the same integer type as ordinary recursive calls. The resulting engine compiles one
search signature and one quiescence signature. All compilation still happens during import;
the fix does not postpone work until the game clock, rely on a disk cache, change the startup
limit, or suppress failures. The network, blend, search logic, book and tablebases are unchanged.

An alternative lower-optimization trial was rejected because it barely reduced startup time.
The selected fix keeps Numba's default compiler optimization. Compiler option documentation:
[Numba environment variables](https://numba.readthedocs.io/en/stable/reference/envvars.html).

## Validation

- Original cold profile: 65.33 seconds. Fixed cold profile: 47.79 seconds.
- Six depth-six unbooked searches gave exactly the same moves, scores and node counts.
  Measured speed was 194,321 vs 261,164 nodes/second. These are local samples on a shared
  laptop, not a reliable Elo estimate or a promised speedup on competition hardware.
- The fixed engine passed 25 regression tests with a 42.84-second cold import. A new
  regression test checks that recursive search has a single non-literal signature.
- Repeated two-core cold-start results are in
  `games/revamp/startup-unified-repeated/results.json`. The test performs three fresh
  starts per core, including the third-start case where the earlier run failed, and
  requires a legal unbooked move within a one-second clock. It rejects fallback execution.
  **All six passed**, with imports 36.84-43.15 seconds and first moves 88-131 milliseconds.
- Follow-up full-clock results are saved separately under
  `games/revamp/full-clock-v31-20260911/` when started. Inspect its state and lane logs;
  a planned match count is not a completed test result.

The exact source snapshot is `games/revamp/release-v3.1/`. The archive
`submission-v3.1.zip` has SHA256
`721223cfa632509810f145107908a5435acca7d3d630c9d3d1d7891e488c0a1a` and is 43,917,280
bytes uncompressed. `submission-v3.1.audit.json` contains file hashes and the archive audit.
The older `submission-v3.zip` is preserved. The root source uses the same fix with its
existing import-formatting convention.

## Continuing work

Retain the typed arguments when modifying root or null-move searches. Run
`tools/check_engine_candidate.py` on an isolated candidate and inspect recursive signatures
before paying for large matches. Use `tools/profile_startup.py` to compare cold startup and
fixed-depth behavior. `tools/check_parallel_startup.py` exercises repeated fresh starts on
the two physical performance cores of this laptop; its CPU masks are host-specific.

Run long timed matches only after other heavy jobs finish. Keep the original submitted-v2
and v3 snapshots immutable. The 90-second startup limit also applies to the old opponent;
if it fails, record that failure and distinguish it from a completed playing result.
Never treat a relaxed local timeout as platform validation. Platform upload validation
remains untested by this local work.
