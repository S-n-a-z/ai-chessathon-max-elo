# Chessathon model improvement handover

## Final v3.1 full-clock result

The full-clock batch is COMPLETE: **16 wins, 4 draws, 4 losses in 24 games (75%)**.
Opening starts: +14 =1 -1. Unbooked middlegames: +2 =3 -3. No startup failures,
flags, crashes or illegal moves. The aggregate improvement is driven by opening preparation;
no middlegame improvement is established by the small unbooked sample. The earlier live
checkpoints below are historical. Both lanes have finished; do not wait for a running job.
See `docs/releases/v3.1/README.md` and its JSON reports for the final evidence.


## Current release: v3.1 startup fix, September 11

Interim full-clock checkpoint at00:46UTC:14/24 games completed against submitted-v2,
**11 wins,2 draws,1 loss (85.7%)**. Unbooked subset:2 wins and2 draws. No failures or
flags so far; recorded v3.1 imports28.42-44.99s. Both lanes still running. This checkpoint
is provisional and must not be described as a completed24-game result.

Use `submission-v3.1.zip` or its identical `submission.zip` copy. This supersedes the v3
archive guidance below; `submission-v3.zip` remains available for rollback. Current SHA256:
`721223cfa632509810f145107908a5435acca7d3d630c9d3d1d7891e488c0a1a`.

Five integer casts eliminate duplicate Numba recursive compilation, with the trained model
and search choices unchanged. All 25 regression tests and all six parallel cold-start tests
passed. Repeated imports took 36.84-43.15 seconds, including both third attempts. Six fixed-depth
searches matched v3's moves, scores and nodes exactly. Read [STARTUP_FIX.md](STARTUP_FIX.md).
Root source includes the fix; exact archived source is `games/revamp/release-v3.1/`.

A NEW 24-game full-clock parallel batch was started under
`games/revamp/full-clock-v31-20260911/`, against unchanged submitted-v2, with fresh processes
and the 90-second startup gate. Inspect its `results.json` and lane logs before starting
other heavy work. Its results must not be confused with the failed older v3 batch below.
Do not upload on the basis of a planned game count. Nothing was uploaded by this task.

## Additional testing requested after this handover

**Outcome checked September 11:** the batch stopped after four completed games (+1 =3 -0,
62.5%). Both lanes hit the 90-second initialization timeout on their third attempted game;
the aggregate state is `failed`. No unbooked games completed. The four played games ended
normally, but the startup failures mean this is not a passed reliability gate. Including
the earlier two full-clock games gives +1 =4 -1 (50%) across six played games, excluding the
startup failures. No processes remained running. Diagnose startup and rerun the remaining
tests before claiming superiority at the full clock. The resource-pressure cause is unproven.

On September 10 the user requested more full-clock tests in parallel. The background job
`tools/run_full_clock_parallel.py` writes to `games/revamp/full-clock-parallel-20260910/`.
It schedules 24 new games: 16 from eight opening starts and eight from four unbooked starts,
with swapped colours, fresh processes, 120 s +0.5 s and the unchanged 90-second init limit.
Two lanes run on separate physical P-cores (affinity masks1 and4 on this laptop). The
combined `results.json` updates automatically and separates opening/unbooked scores; lane
directories hold PGNs and detailed checks. Consult its state before starting more work.
No submission file was changed. The original saved results below predate this new run.

Saved 10 September 2026. The user asked to save the strongest measured build and conserve
usage. The training and final matches have completed; no further experiment was started in
this save-and-handover step. Nothing has been uploaded by this task.

## Release to use

- `submission-v3.zip` is the selected archive; `submission.zip` is an identical copy.
- SHA256: `4cc755ead120bf22831d178400486778de55b634c1efd284f59eaace0cd4d56f`.
- 41,913,881 bytes compressed; 43,917,055 bytes uncompressed; 93 files.
- Root runtime and weights implement `games/revamp/release-v3/`; two root import blocks
  were sorted afterward for lint. The archive preserves the exact tested release bytes.
- `submission-v3.audit.json` records every archived file hash and packaging checks.
- Previous uploaded archive: `games/revamp/submission-v2.zip`; its extracted source is
  `games/revamp/submitted-v2/`. Previous working files: `games/revamp/pre-release-root/`.
- Earlier frozen pre-revamp source: `games/revamp/baseline/`. Keep this: training residuals
  were computed against its classical evaluator.

This is the strongest candidate in our larger fast match, but competition-clock superiority
is not established. Do not describe the results below as a guaranteed win or an absolute Elo.

## Evidence and limitations

Scores are from the candidate's perspective. Paths below are under `games/revamp/`.

| Comparison | Conditions | Wins / draws / losses | Evidence |
|---|---|---|---|
| Release v3 vs exact submitted v2 | 24 games, 10 s + 0.1 s | 18 / 5 / 1 (85.4%) | `release-v3-v2-timed/results.json` |
| Release v3 vs v2, beyond book eligibility | 16 games, 50,000 nodes per move | 6 / 7 / 3 (59.4%) | `release-v3-v2-unbooked/results.json` |
| Quiet network vs staged classical search | 16 games, 20,000 nodes per move | 10 / 3 / 3 (71.9%) | `neural-quiet-v-staged/results.json` |
| Release v3 vs v2 | Two fresh games, 120 s + 0.5 s | 0 / 1 / 1 | `release-v3-v2-full/results.json` |

The full-clock games finished by repetition and checkmate, with no flags, illegal moves or
crashes. Candidate imports took 60.19 and 65.38 seconds. This is a local reliability check,
not evidence of superiority at that clock or proof of platform acceptance. Both games use
the same opening with colours exchanged. The fast match reuses worker processes, whereas the
platform starts a fresh process per game. Prefer `--fresh` in future timed comparisons.

Published competition openings informed the new book and overlap the fast match's starts;
that score includes preparation benefits. The unbooked match starts at move 21, but is also
small and uses public games. Neither result establishes performance on unseen official starts.
Some diagnostic imports under heavy concurrent CPU load exceeded 90 seconds. Recheck cold
initialization on representative Linux hardware, with one CPU and fresh caches.

Completed checks include 24 engine tests, neural reference agreement on 500 positions and
750 move/unmove transitions (maximum discrepancy below 0.000092 cp), and 500 randomized rook
endgame checks. See `release-v3-check.json`, `neural-quiet-check.json`, and
`rook-tables-check.json`. Ruff and configured mypy checks passed; Numba internals have existing
type-check exclusions. The zip audit checked layout, size, allowed file types, CRC and bytes.

## What changed

`agent.py` exposes `get_move(fen, time_left_ms)`. `fast_engine.py` is the original Numba 0x88
search: iterative deepening, alpha-beta/PVS, transposition table, quiescence, pruning, move
ordering and clock control. Search work added incremental hashes, deferred legality checks,
wall-clock checks, explicit stalemate handling and prevention of consecutive null moves.

The selected evaluator combines the classical score with **50% of an original learned
residual correction**. `neural.py` implements cached integer accumulators; `neural_full.py`
provides the reference implementation and weight loading. The network is a shared symmetric
king-bucket sparse model with 24,576 input features and 32 hidden channels. Feature weights
are int8, accumulators int32 and output weights int16. No pretrained chess network or external
engine implementation is shipped.

`opening.py` serves 13,456 permitted opening positions and rejects every lookup after FEN
fullmove 20. The book combines the old 5,474 positions with 8,000 offline analyses, with
overlap. New preparation used Stockfish 18 at 50,000 nodes, two PV branches, and up to eight
additional plies from public starts. The teacher stays outside the submission.

`endgame.py` uses complete three/four-piece Syzygy data plus selected five-piece rook endings:
KRPvKR, KQRvKR, KRBvKR and KRNvKR. KRRvKR has WDL only as a promotion dependency; its root
positions fall back to search. Missing coverage must fall back safely. Preserve root DTZ,
all-alternative coverage and fifty-move handling when modifying this code.

## Training data and reproducibility

The winning neural experiment was `king32_quiet_1m`, trained from scratch on CC0 Lichess engine
evaluations. Its cache has 908,604 positions: 727,142 training, 90,852 validation and 90,610
sealed test positions. The deepest first PV at depth >=24 supplies a white-centric label.
The quiet filter excludes checked positions and positions whose teacher move captures,
promotes or gives check. This filter was more useful than simply scaling unfiltered data.

Training used seed 20260907, hidden 32, batch 1024, AdamW learning rate 0.002, weight decay
0.000001, cosine scheduling, and SmoothL1 with a 140 cp transition. Best epoch was 11 of 16
completed from 24 requested. Quantized validation teacher MAE was 168.57 cp versus 239.62 cp
for the classical evaluator on the same positions. The sealed test was not used for selection.

Exact positions have a stable canonical-FEN split and known competition positions are
excluded. The evaluation dump lacks game IDs, so related positions can cross splits. A
future PGN-derived dataset should split by source game before extracting positions.

Files needed to reproduce or continue:

- `games/revamp/king32_quiet_1m.npz`, `.json`, `.log`, `.best.pt`: selected export, training
  report, log and checkpoint. Do not assume the trainer can resume optimizer state; check its
  supported arguments first.
- `weights/NNUE.md`, `NNUE_TRAINING.json`, `NNUE_CACHE.json`: architecture and provenance.
- `C:/Users/galac/AppData/Local/Temp/blockshark-training/cache-revamp-quiet-1m`: training shards.
- `C:/Users/galac/AppData/Local/Temp/blockshark-training/lichess-evals-data_0000.parquet`:
  original 2,118,195,009-byte source; SHA256
  `c004ce90acd84e0db3c55ad8489bce42114b37558ab97400d9a20a31dd0f864a`.

The large dataset remains in a temporary directory. Before cleanup or moving machines, copy
the source and cache to persistent storage and verify their manifest hashes. The submission
contains trained weights and provenance, not those datasets.

## Next work, in priority order

1. **Establish strength at the actual clock.** Run fresh paired games against frozen v3 and
   v2 on a larger set of unseen openings. Include book-disabled or move-21 starts. Start with
   a small screening match, then use at least 50-100 opening pairs for promising changes if
   resources permit. Report paired uncertainty and failures, not only the point score.
2. **Diagnose the full-clock loss.** Inspect `release-v3-v2-full/game-002.pgn` and its logs;
   verify the filename in that directory. Use offline teacher analysis to locate the first
   substantial error and distinguish evaluation, pruning, repetition and time allocation.
   Convert reproducible failures into regression positions before changing search.
3. **Tune the selected model's blend and speed.** Try 0.25, 0.5 and 0.75 individually, keeping
   the book and tables identical. Measure nodes per second and full-clock results. Cached
   inference speed and stable initialization can matter more than a smaller teacher error.
4. **Expand the quiet dataset, then test 32 and 64 channels.** First try 2-5 million accepted
   quiet positions with the same filters and a fixed validation set. The existing source was
   exhausted: raising a limit alone is not a guarantee of obtaining more suitable positions.
   Use a larger source or additional independently deduplicated source data. Preserve phase
   balance, score conventions, source licensing and hashes. Keep the test set sealed.
5. **Improve target relevance.** An experimental alternative is to label quiet leaves reached
   by this engine's search, with a stronger offline teacher. Split independent games before
   generating labels. Compare against the current quiet-data recipe with identical search.
   This is a proposed experiment, not an implemented or proven gain.
6. **Tune search only after diagnosing losses.** Change one of move ordering, reductions,
   pruning margins or time allocation at a time. Require correctness checks and timed games;
   equal-node games alone hide the cost of slower evaluation. Do not repeat rejected
   experiments without a specific changed hypothesis.

Rejected experiments already include unfiltered 200k 32-channel (+4 =3 -9), unfiltered ~1m
32-channel (+6 =0 -10), and an endgame-only blend (+3 =4 -9), all in 16-game equal-node tests.
The unfiltered 64-channel model improved validation error but lacks positive playing evidence.
Do not replace the selected net just because a larger one has lower loss.

## Commands for the next session

Run from the project root in PowerShell. These are future commands, not jobs started by this
handover. Use unique output directories and run expensive jobs sequentially. Never benchmark
wall-clock strength while training or a teacher engine is consuming CPU.

Reproduce training on the existing cache, writing a new export:

```powershell
.venv312/Scripts/python.exe tools/train_king_nnue_cache.py --cache C:/Users/galac/AppData/Local/Temp/blockshark-training/cache-revamp-quiet-1m --baseline-evaluator games/revamp/baseline/fast_engine.py --hidden 32 --epochs 24 --batch-size 1024 --threads 1 --seed 20260907 --out games/revamp/king32_quiet_retrain.npz
```

The frozen baseline argument is essential: root `fast_engine.py` now includes the neural
correction. Residual targets must use the matching classical baseline, never an evaluator
that already adds the correction. The trainer verifies the baseline hash.

Build a larger cache from the documented official source (may require installing a training
decompressor locally; inspect `--help` for `--zstd`). Keep these training-only dependencies
outside the runtime archive:

```powershell
.venv312/Scripts/python.exe tools/build_lichess_eval_cache.py --source https://database.lichess.org/lichess_db_eval.jsonl.zst --out C:/Users/galac/AppData/Local/Temp/blockshark-training/cache-quiet-3m --baseline-evaluator games/revamp/baseline/fast_engine.py --quiet-only --min-depth 24 --max-accepted 3000000 --seed 20260907
```

To test a new network, copy `games/revamp/release-v3/` into a new candidate directory and
replace only its `weights/king_nnue.npz`, updating the provenance after validation. For blend
experiments change `NN_BLEND` in that isolated candidate. The older
`tools/build_neural_candidate.py` expects a **classical** source such as
`games/revamp/search-staged/`; do not run it on the already-neural root or release v3, because
its text transformations are not designed for reinserting a second wrapper. If using that
builder, copy the selected book, endgame module and table files afterward to control variables.

For a new isolated candidate `games/revamp/candidate-next`:

```powershell
.venv312/Scripts/python.exe tools/check_neural_candidate.py --candidate games/revamp/candidate-next --out games/revamp/next-neural-check.json
.venv312/Scripts/python.exe tools/check_engine_candidate.py --candidate games/revamp/candidate-next --out games/revamp/next-engine-check.json
.venv312/Scripts/python.exe tools/revamp_match.py --candidate games/revamp/candidate-next --baseline games/revamp/release-v3 --starts games/revamp/unbooked-starts.json --pairs 8 --nodes 50000 --out games/revamp/next-unbooked
.venv312/Scripts/python.exe tools/revamp_match.py --candidate games/revamp/candidate-next --baseline games/revamp/release-v3 --pairs 12 --fresh --base-ms 120000 --increment-ms 500 --init-seconds 90 --out games/revamp/next-full
```

Those default public starts are regression positions, not a held-out strength set. Supply a
separately prepared JSON `starts` list for the unseen-opening trial. Do not raise init limits
and describe the resulting run as compliant. Read every game termination and import time.

For an unchanged v3 rebuild:

```powershell
.venv312/Scripts/python.exe tools/package_release.py --candidate games/revamp/release-v3 --out submission-v3-rebuild.zip
```

The packager deliberately pins the v3 network hash. For a genuinely new selected model,
update that expected hash from the verified training artifact, preserve equivalent audit
checks, and use a new versioned filename. Never bypass a hash assertion just to package an
unidentified model. Preserve the previous source and archive before promotion.

## Rules and continuation state

Canonical rules were fetched on 10 September into `games/revamp/*2026-09-10.md`. Refresh
[the agent contract](https://aichessathon.com/docs/agent-contract.md) and
[the rules](https://aichessathon.com/docs/rules.md) before relying on limits. At this save:
Python source, original trained networks, opening lookups at fullmove <=20, tablebases with
<=7 pieces, <=50 MB uncompressed, 90-second initialization, and 120 s + 0.5 s with one CPU,
2 GB RAM and no runtime network. Do not ship third-party engines, pretrained networks or
middlegame lookup tables. Do not edit `harness/`. Upload validation is authoritative.

Data source documentation: [Lichess evaluations](https://database.lichess.org/#evals).
Book and table provenance is also shipped under `weights/`.

`games/revamp/STATUS.md` contains experiment history; its new top section supersedes old
RUNNING/QUEUED labels. `tools/finalize_trial.py` has finished. Do not rerun it on the frozen
release: it mutates the book/documentation and reuses old output paths. The previously created
hourly follow-up is named `improve-chessathon-engine`; check its current app status before
resuming scheduled work. Start no paid compute or uploads without the user's authorization.
