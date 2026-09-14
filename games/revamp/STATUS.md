# Chessathon improvement status

## Final v3.1 full-clock result

The full-clock batch is COMPLETE: **16 wins, 4 draws, 4 losses in 24 games (75%)**.
Opening starts: +14 =1 -1. Unbooked middlegames: +2 =3 -3. No startup failures,
flags, crashes or illegal moves. The aggregate improvement is driven by opening preparation;
no middlegame improvement is established by the small unbooked sample. The earlier live
checkpoints below are historical. Both lanes have finished; do not wait for a running job.
See `docs/releases/v3.1/README.md` and its JSON reports for the final evidence.


## Live full-clock checkpoint — September 11, 00:46 UTC

v3.1 versus unchanged submitted-v2:14/24 games completed, +11 =2 -1 (85.7% score).
Opening starts:+9 =0 -1 in10; unbooked middlegames:+2 =2 -0 in4.
All completed games ended normally:12 checkmates and2 repetitions. No startup failure,
flag, crash or illegal move reported. Completed v3.1 imports28.42-44.99s; old v2 imports
36.99-57.63s. Both lanes remain active. This is an interim result on public regression
starts, not an unseen-opening rating estimate. Do not duplicate the running test.

## v3.1 SAVED; new full-clock batch running — September 11

Startup fix selected and promoted: five np.int64 casts remove duplicate recursive Numba
specializations. Network, search semantics and knowledge data unchanged. All 25 engine tests
PASS; all six repeated parallel cold starts PASS (36.84-43.15s), with legal first moves
within 1 second. All six fixed-depth benchmark moves/scores/node counts match original v3.

- Current archive `submission-v3.1.zip`, identical to `submission.zip`, SHA256
  `721223cfa632509810f145107908a5435acca7d3d630c9d3d1d7891e488c0a1a`.
  Size 43,917,280 bytes uncompressed. Audit in submission-v3.1.audit.json.
- Exact archived source: release-v3.1/. Root fast_engine.py has the same fix with sorted
  imports. Previous root fast_engine.py saved as root-before-startup-fix.py in this folder.
  Previous v3 archive remains at project root, and previous v2 backup is unchanged.
- RUNNING 24 new full-clock games vs unchanged submitted-v2, two physical P-core lanes,
  fresh processes and unchanged 90-second init gate: `full-clock-v31-20260911/`.
  Its manifest records process IDs and results.json updates automatically. No other heavy
  experiment should run concurrently. Do not duplicate the old failed batch.
- Startup check details: startup-unified-repeated/results.json. Explanation and continuation:
  project-root STARTUP_FIX.md and MODEL_IMPROVEMENT_HANDOVER.md.
- No upload performed. This section supersedes earlier statuses below.
- Final save checks: root Ruff and configured mypy passed; archived files match the
  tested candidate byte-for-byte. Root implementation matches the archive apart from
  import ordering. The first fixed-engine imports in the new full-clock lanes passed.

## Startup fix in verification — September 11

User authorized an efficient startup fix and new tests. Isolated candidate `startup-unified/`
normalizes five constant arguments to np.int64 at root/null search calls. This reduces
recursive `_search` and `_quiescence` signatures from three each to one each, retaining
default full LLVM optimization and identical trained weights, book and tables.

- Fresh profile: v3 import65.33s /194,321nps; fixed import47.79s /261,164nps on six
  depth-six unbooked searches. All moves, scores and node counts exactly match; timings
  are local samples, not an Elo estimate. Profile JSONs: startup-v3/startup-unified-profile.
- Candidate engine suite:25 tests PASS, import42.84s. New regression prevents duplicate
  literal specializations. Report startup-unified-engine-check.json.
- An OPT1 trial barely improved startup (62.21s); it is not selected.
- RUNNING six fresh candidate imports on two physical cores, three consecutive games per
  core, preserving90s init limit and checking a legal first unbooked move within1s:
  startup-unified-repeated/; unified exec session7777. Do not duplicate or start heavy jobs.
- Next after this check: package a preserved v3.1 archive, update root, and rerun full-clock
  tests against unchanged submitted-v2. Current saved archives still unchanged.

## Parallel test outcome — checked September 11

Both lanes STOPPED with a 90-second initialization timeout on their third attempted game.
The requested 24-game batch did not complete; only four games finished: v3 +1 =3 -0,
62.5%, all opening starts. No unbooked games finished. Completed terminations were three
repetitions and one checkmate, with no in-game flags or illegal moves.
The aggregate state is `failed`; do not present the four completed games as a passed gate.
Completed-game v3 imports ranged52.95-85.01s; v2 imports41.32-87.18s. Parallel resource
pressure may contribute, but the cause is not established. Combined with the earlier two
full-clock games, played results are +1 =4 -1 (50%) in six games, excluding init failures.
No test processes remained when checked. Startup reliability and a larger full-clock sample
remain unresolved. Saved archives were not changed. See full-clock-parallel-20260910 logs.

## Additional full-clock tests — September 10, 22:10 UTC

User explicitly requested more full-clock tests in parallel after saving the release.
`tools/run_full_clock_parallel.py` is running as background supervisor PID25992.
Do not duplicate it or start CPU-heavy training while timed games run.

- Output directory: `games/revamp/full-clock-parallel-20260910/`.
  Combined `results.json` updates automatically; `manifest.json` records lane PIDs.
  Per-lane logs, PGNs, import times and per-game results are retained.
- Initial startup check at22:14UTC: both new and old engine processes cleared the
  90-second gate in both lanes; both first games are actively playing. CPU affinity
  inheritance was verified on the actual engine processes. No completed score yet.
- Planned 24 NEW games: eight opening pairs and four unbooked middlegame pairs.
  Opening pair zero from the old full-clock check is excluded; FENs are deduplicated.
- Two concurrent match lanes, CPU affinity masks1 and4, on distinct physical P-cores of
  this i3-1215U. One logical CPU per lane, inherited by both agents; numerical threads=1.
- Exact frozen release-v3 versus submitted-v2; fresh processes for every game,
  120 seconds +0.5 seconds per move, unchanged harness, 90-second startup limit.
- Local Windows hardware, shared thermal budget and low available RAM mean results are
  local evidence. Startup failures must remain visible; never extend the init limit to hide them.
- Opening and unbooked summaries are separate. These public starts are regression tests,
  not unseen official positions. Saved submission archives remain untouched.
- When complete, inspect all terminations/init times and update the handover with the
  combined score. If state is failed, inspect lane logs before starting another run.

## SAVED RELEASE — September 10, 2026

The user requested immediate saving to conserve usage, then a Markdown continuation guide.
The release has been packaged and the root runtime promoted. Nothing has been uploaded.
**This section supersedes every historical running/queued/provisional label below.**

- `submission-v3.zip` and `submission.zip` are identical: SHA256
  `4cc755ead120bf22831d178400486778de55b634c1efd284f59eaace0cd4d56f`.
- Size 41,913,881 bytes compressed / 43,917,055 uncompressed; 93 files.
  Exact file hashes and checks: `submission-v3.audit.json` at project root.
- Frozen selected source: `games/revamp/release-v3/`; root runtime and weights implement it.
  Root fast_engine.py/neural.py imports were sorted afterward for lint; archived source stays
  byte-for-byte identical to the tested release. Root Ruff and configured mypy now pass.
- Previous archive saved in `games/revamp/submission-v2.zip`; previous root runtime,
  README and weights backed up in `games/revamp/pre-release-root/`.
- Completed timed trial vs exact v2: +18 =5 -1 in 24 games at 10 s + 0.1 s (85.4%).
- Completed unbooked trial: +6 =7 -3 in 16 games at 50k nodes (59.4%).
- Completed full-clock trial: +0 =1 -1 in two fresh games at 120 s + 0.5 s.
  Both cleanly finished, no flags/crashes/illegal moves; candidate init 60.19/65.38 seconds.
  This small full-clock result does NOT establish a strength improvement at the live clock.
- Book preparation, training and finalize_trial.py ALL COMPLETE. No new experiments started
  during this save. Do not rerun finalize_trial.py against the frozen release.
- Continuation instructions: `MODEL_IMPROVEMENT_HANDOVER.md` at project root. Prioritize
  broader fresh-process full-clock tests, loss diagnosis, then controlled blend/data trials.
- The hourly follow-up was previously created; inspect its current app status before scheduling
  more work. Preserve user usage constraints. No paid compute or automatic upload authorized.

## Historical experiment log


Updated 2026-09-10, work in progress. Do not upload an untested experiment.

## Current priority: finish the release (September 10, 16:45 UTC)

User explicitly asked to finish ASAP and make it substantially better. Stop broad training
experiments and select/package the strongest measured build today. This section supersedes
the older running/queued labels further below. Canonical rules refreshed into *2026-09-10.md.

- ALL September 8 jobs completed. Staged classical vs root: +10 =14 -8 in 32 timed games,
  score 53.125%, pair SE 5.04 percentage points. No decisive gain. Startup 114.35/94.94s
  under diagnostic allowance, despite the separate 50.3s local check. Reliability still a gate.
- Quiet cache COMPLETE: 908,604 positions, 727,142 train / 90,852 validation / 90,610 sealed
  test. king32_quiet_1m.npz trained from scratch; best epoch11 of16, validation teacher error
  239.62 -> 168.57 cp. Its game strength was not yet tested before this continuation.
- RUNNING neural-quiet-staged (50% cached blend) vs search-staged, 16 games at 20k nodes,
  .venv312, local init allowance180s: neural-quiet-v-staged.log / directory, session31057.
  COMPLETED: +10 =3 -3 (71.875%, pair SE8.76 points). First positive neural playing result.
  Inference check passed500 positions +750 transitions, max0.000092cp discrepancy; import59.78s.
  Full24-test check passed with import60.56s: neural-quiet-engine-check.json.
- RUNNING permitted opening preparation: tools/build_competition_book.py with Stockfish18
  OFFLINE ONLY, 50k nodes, branches2, max8000 analyses, extension8plies, absolute move<=20.
  Source1777 opening positions from public team games through round100. Reads games/round-*.pgn
  and latest-public/round-*.pgn. Output competition-book-v3.json / .audit.json / .log,
  session78866. Partial checkpoints every250; wait for metadata complete before packaging.
  Do not run timed strength matches while this teacher process is consuming CPU.
- opening.py now refuses all lookups after move20, including repeated early arrangements.
  Same guard copied into both experimental candidates before their new match. submitted-v2
  remains byte-for-byte untouched. Added a rule-boundary regression test (24 tests total now).
- Next: use the neural result to choose classical or neural candidate; add completed opening
  book to a NEW release directory, then compare timed games against exact submitted-v2.
  Include held-out starts if possible; disclose that published opening starts informed the book.
  Run all tests, full-clock fresh games, startup/zip checks. Package a reviewable archive even
  if the measurable gain is smaller than requested; report the evidence honestly. No upload yet.

### Release assembly and live jobs (September 10, 16:55 UTC)

Update17:04UTC: opening book COMPLETE and audited: 13,456 total positions, 8,000 new analyses.
Copied by tools/finalize_trial.py into release-v3. Rook table documentation also updated there.
Unbooked50k-node match against exact v2 COMPLETE: +6 =7 -3, 59.375%, pair SE10.5 points.
FINAL TIMED TRIAL RUNNING: release-v3-v2-timed,24 games10s+0.1s. Both workers passed90s
startup under normal load (candidate44.62s, v234.72s). No other heavy jobs are running.
tools/finalize_trial.py (session13132) automatically starts2 fresh full-clock games afterward
as release-v3-v2-full. Do not duplicate either test. Logs finalize-trial.log and each match.log.
Rook and NN checks have passed. Wait for final results, then archive/promote the measured build.

- Selected provisional original quiet NN (50% blend) plus staged search. Directory release-v3.
  It still has the old book until competition-book-v3 metadata says complete.
- Added38,530,256 bytes of public Lichess Syzygy data for KRPvKR, KQRvKR, KRBvKR, KRNvKR,
  plus KRRvKR WDL promotion dependency. KRRvKR DTZ intentionally omitted to respect50MB cap;
  that root material class searches. All WDL capture/promotion dependencies checked. Download
  headers/sizes verified and SHA256 recorded in rook-tables/ROOK_TABLES.json.
- release-v3/endgame.py requires root DTZ coverage and coverage of all alternatives, handles
  zeroing distance and near-fifty-move wins, and returns an immediate mate first. Random500
  rook endings preserve WDL; missing DTZ fallback checked. rook-tables-check.json.
- release-v3 passed24 tests, but its latest import under other CPU load took100.84s. It is NOT
  ready to promote until fresh startup gates pass after other work ends. Prior NN import60.56s.
  Ruff passes runtime; strict mypy passes its3 entry modules. Network/runtime docs added under
  release-v3/weights. Current unzipped weights ~43.6MB, leaving room for the new opening book.
- Selected8 neutral public middlegames at move21, teacher50k nodes and |score|<=80cp, from
  latest-public games. These cannot use opening books. unbooked-starts.json.
- RUNNING release-v3 vs exact submitted-v2 on these8 paired middlegames at50k nodes:
  release-v3-v2-unbooked.log / results directory, session25017. Concurrent teacher book build
  is acceptable for fixed-node tests, but stop/wait for it before timed tests.
- Opening preparation still running (session78866); wait for8000 analyses / complete metadata.
  Then copy completed book to release-v3, update provenance, run timed comparison vs v2 and
  fresh full-clock games, audit/package submission-v3.zip, and promote root only with evidence.

## Authorized work

User wants maximum legal playing strength, original neural training from online data, larger
datasets if needed, and continuing updates. An hourly thread heartbeat named
`improve-chessathon-engine` is active. No paid compute or upload has been requested.

## Frozen baselines

- `submitted-v2/` is the exact active dashboard submission extracted from submission.zip.
  Zip SHA256: e2337551b90197a4faecaf6f8f13151f48a66d9c5230aacb765bdcc6cdf4a8a7.
  Dashboard on September 7: Stockfish's Nightmare, rank 175/377, rating 1594, +11 =9 -17.
- `baseline/` is the user's working runtime at start of this task, including uncommitted changes.
  fast_engine.py SHA256: b00ce99588ea3c38822b56e81bd875cfb337dc3912e6e2c946c7efa26a188d12.

## Completed

- Canonical rules fetched. Own-trained networks and engine-labelled training data allowed;
  third-party engine source, pretrained networks and runtime label lookup forbidden.
- Original 32-hidden king-conditioned residual trained from scratch for 18 epochs on cached
  200,000 Lichess positions (depth >=24): 159,952 training, 20,079 validation, 19,969 sealed test.
  Validation teacher MAE 294.09 -> 202.53 cp (31.1% lower); this is not Elo evidence.
  Weights `king32_200k.npz`, full provenance/training report `king32_200k.json`.
- Search changes in root fast_engine.py: tactical-only quiescence, stalemate checks,
  incremental hashes, consecutive-null prevention, real clock checks, urgent-clock fallback,
  removed timed-path static move override. Fixed workers honor USE_ROOT_POSTPROCESS flag.
- New tests tests/test_search_core.py cover hashes, tactical generator, stalemate, deadline.
  The latest run passed all 23 tests (final-tests.log), including 500 random generator positions,
  perft, incremental hash changes, urgent clocks and deadline unwind. A Python/JIT hash test needed
  an explicit uint64 argument because Python otherwise boxes the returned key as signed int.
  Mypy passed all 9 configured source files; ruff passed runtime, tests and tools.
- tools/revamp_match.py saves paired game PGNs, clocks, fingerprints, genuine outcomes and
  600-ply draws. Supports fixed nodes and fresh platform runner processes. harness/ untouched.
- tools/build_neural_candidate.py + tools/neural_runtime.py build isolated neural candidates.

## Results and active work (September 8, 15:56 UTC)

- Larger cache COMPLETE: 983,811 positions in 48 shards. Split: 787,212 training, 98,263
  validation, 98,336 sealed test. Location `%TEMP%/blockshark-training/cache-revamp-1m`.
  Source exhausted after 54 million flattened rows; log cache-1m.log. Source Parquet SHA256
  c004ce90acd84e0db3c55ad8489bce42114b37558ab97400d9a20a31dd0f864a, CC0 Lichess evals.
- Larger 32-unit training COMPLETE: king32_1m.npz / .json / .best.pt; best epoch 10 of 15 run.
  On the SAME 98,263 validation rows, HCE MAE 289.17, old 200k net 255.89, new net 204.59 cp.
  Full report validation-slices.json. Near-equal early/middle positions still worsen slightly,
  while near-equal endgame error improves from HCE 191.55 to 155.32 cp.
- Neural 200k half blend LOST equal-node match: +4 =3 -9, 16 games at 20k nodes.
  neural-v-search-20k-relaxed/results.json. REJECTED for release.
- Neural 983,811-position half blend with cached accumulators also LOST: +6 =0 -10 at 20k
  nodes. neural1m-v-search-20k/results.json. REJECTED for full-game release.
- Larger-capacity 64-unit network COMPLETE: king64_1m.npz / .json / .best.pt. From scratch,
  best epoch 10 of 16 completed, 1,230.55 seconds. Quantized teacher MAE 197.33 cp on the same
  validation rows. This is only 3.6% better than the larger 32-unit net; game strength untested.
- Endgame-only 32-unit network (75% blend, phase <=6) LOST +3 =4 -9 against root at 20k
  nodes, 16 games: neural1m-endgame-match/results.json. REJECTED for release.
- Current classical search vs frozen pre-revamp working engine: +6 =4 -6 at 20k nodes;
  first timed test +1 =4 -3. Restored iteration-cost predictor, retest +3 =3 -2 at 10s+0.1s.
  Results in search-v-baseline-retimed. This small result does NOT establish a strength gain.
  Root remains an UNPROMOTED development candidate. search-original-inlining preserves an
  earlier variant. Current code avoids forced inlining of _hash_after_move to reduce JIT cost.
- Isolated search-staged variant lazily checks legal moves when searched, retaining legal-only
  move indices for PVS/LMR and terminal checks before pruning. tools/build_staged_search.py
  constructs it from this project's original source. Depth-6 results match all 8 moves, scores,
  node counts and completion flags EXACTLY: 560,948 total nodes. Current 3.619s, staged 2.873s
  (155k vs 195k NPS), but load was not controlled. Files search-{current,staged}-depth6.json.
  Correctness tests subsequently passed (below); paired timed evidence still needed before
  promotion. It has not been copied to root.
- Neural inference tools verified full recomputation on 500 independent positions and cached
  accumulators on 500 positions plus 750 move/unmove transitions (max difference 0.000172 cp).
  Reports neural-check-opt1.json and neural-cached-check.json. New phase gating and later
  weights still need their own check. Accumulator scratch storage follows the 128 board cells.
- Created .venv312 with Python 3.12.14, numpy 2.5.2, numba 0.67.0 and chess 1.11.2.
  search-staged PASSED all 23 tests and imported in 50.325 seconds with compiled search active.
  Output search-staged-python312-check.json / .log. This passes the local 90-second check;
  Linux/EPYC platform validation remains necessary. Existing .venv and bundled libraries unchanged.
- First 32-game match attempt hit the 90-second worker init watchdog before playing any game,
  despite the 50.3s standalone test. Log search-staged-timed312.log. Startup is NOT consistently
  within budget on this laptop. The old shell ended without starting its dependent cache builder.
- RUNNING diagnostic retry, 32-game timed search-staged vs root under .venv312, 10s+0.1s,
  relaxed 360s LOCAL init allowance: search-staged-timed312-relaxed.log / results directory.
  Unified shell session 48219; original failed session 99436 has ended.
  That SAME shell automatically starts the quiet cache build after successful match completion.
  Do not start a duplicate builder merely because its log has not appeared yet.
- Added --quiet-only to tools/build_lichess_eval_cache.py: retain unchanged source FEN/score
  only when not in check and deepest PV1 begins with a legal non-capture/non-promotion/non-check.
  Provenance records the filter; stable split is unchanged. tools/check_eval_data.py passed
  JSON/Parquet selection, multi-PV, mate labels, batch boundaries and filter checks.
  Quiet cache QUEUED behind the match: cache-revamp-quiet-1m under the same TEMP training
  directory, target 1m, min depth24, shard25k, parquet batch8192, frozen baseline. Log
  cache-quiet-1m.log. A small stdlib supervisor is ALREADY WAITING (session 81448):
  tools/train_after_cache.py starts king32_quiet_1m.npz training after cache completion, with
  24 epochs requested, 32 hidden units, two threads and >=200k accepted rows. Log
  king32_quiet_1m.log. The 6-hour supervisor deadline reports an error if preparation stalls.
  Primary rationale: https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-(train.py)
  describes rejecting best-move captures; no external training/engine code was copied.

## Next decisions

1. Finish timed search-staged match; inspect 64-unit validation slices on the same rows after
   timed matches finish. Do not run heavy training or imports alongside timed strength tests.
2. If staged timed results support promotion, compare against exact submitted-v2 and run fresh
   full-clock games. Keep it isolated until evidence supports replacing root.
3. Endgame neural lost, so quiet-position training is the next experiment. If that fails too,
   investigate simpler shared features or tuned classical weights. Never promote on MAE alone.
4. Strongest candidate still needs comparison against exact submitted-v2, fresh-process games
   at full time control, target Python startup, and package inspection. Current submission.zip
   remains unchanged. Do not upload without a user request.
5. Update this status before ending work so the hourly heartbeat can continue without duplicates.

## Practical constraints and caveats

Laptop: i3-1215U, 6 cores/8 threads, 8 GB RAM, CPU-only torch. RAM is very constrained with
other apps open; avoid simultaneous heavy imports/training. Existing .venv uses Python 3.13.5,
not platform Python 3.12. The separate .venv312 matches target Python/packages; Linux platform
verification remains necessary.
No absolute Elo or winning guarantee is justified yet. Existing ab_match.py uses material
adjudication and must not be used as final strength evidence. Stable FEN-hash split prevents exact
position leakage, not game-level separation (the public eval dump has no game IDs).
Observed local imports ranged 75–248s in current trials, and a heavily loaded prior baseline
import took 294s. These are local Windows resource-contention measurements; they do not pass
the platform's 90s startup requirement. The isolated staged build subsequently passed a local
Python 3.12 check in 50.3s. No new upload or new submission.zip has been produced.
agent.py now reports JIT initialization failures to stderr, and the benchmark worker refuses
to silently benchmark the pure-Python fallback when the compiled engine fails to initialize.
