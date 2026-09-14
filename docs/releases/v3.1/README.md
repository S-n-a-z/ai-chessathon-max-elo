# v3.1 release evidence

The full-clock comparison against the exact previous submitted-v2 finished all 24 games:
**16 wins, 4 draws, 4 losses (75% score)** at 120 seconds plus 0.5 seconds per move.

| Starts | Wins | Draws | Losses | Score |
|---|---:|---:|---:|---:|
| Opening positions | 14 | 1 | 1 | 90.625% |
| Unbooked middlegames | 2 | 3 | 3 | 43.75% |

Both lanes exited successfully. Every game ended by checkmate or repetition; there were no
startup failures, flags, crashes or illegal moves. Candidate import times ranged from
28.42 to 44.99 seconds. Each game used fresh processes on one logical CPU
per match lane, with two lanes on separate physical performance cores of the local Windows
laptop. The hardware differs from the competition environment.

The book was prepared using published competition openings. These are public regression
starts, not a held-out official opening set. The strong aggregate result is driven by opening
preparation; the small unbooked sample does not demonstrate a middlegame strength improvement.
This is not an absolute Elo estimate or a guarantee of winning the competition.

All 25 engine regression tests passed. Six additional cold-start tests, two at a time,
passed in 36.84-43.15 seconds. Six fixed-depth benchmark positions retained identical moves,
scores and node counts after the startup fix.

The accompanying JSON files retain detailed scores and tested source hashes. Per-game PGNs
are in the lane directories. Historical absolute paths in reports identify the original
local runs; they are not prerequisites for using the archive.

Upload the repository-root `submission-v3.1.zip` as-is. Its SHA256 is
`721223cfa632509810f145107908a5435acca7d3d630c9d3d1d7891e488c0a1a`.
The local `submission.zip` is identical. Platform upload validation has not been checked here.
