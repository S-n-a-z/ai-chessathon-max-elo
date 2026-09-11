# BlockShark — AI Chessathon v3.1

An original Numba chess engine with a neural evaluator trained from scratch, opening
preparation and Syzygy endgame coverage. The v3.1 update removes duplicate recursive
compilation while preserving search decisions and the trained network.

Upload [submission-v3.1.zip](https://github.com/S-n-a-z/ai-chessathon-max-elo/releases/download/v3.1/submission-v3.1.zip) as-is. The archive is 43,917,280 bytes
uncompressed. Its checksum and file audit are in [submission-v3.1.audit.json](submission-v3.1.audit.json).

The completed 24-game full-clock comparison against the previous submission scored
**16 wins, 4 draws and 4 losses (75%)**, with no startup failures, flags, crashes or illegal
moves. Opening starts scored +14 =1 -1; unbooked middlegames scored +2 =3 -3. The aggregate
improvement is driven by opening preparation, and the small unbooked sample does not establish
a middlegame improvement. These local public-opening tests do not guarantee a competition win.

All 25 regression tests and six repeated parallel cold starts passed. Read:

- [Release results and PGNs](docs/releases/v3.1/README.md)
- [Startup fix](STARTUP_FIX.md)
- [Model improvement handover](MODEL_IMPROVEMENT_HANDOVER.md)
- [Training provenance](weights/NNUE.md)

Large training caches and experimental snapshots referenced by the handover remain local.
To recreate the frozen current runtime on another machine, extract the versioned submission
zip into a new directory. The original classical evaluator used for residual labels is retained
at `games/revamp/baseline/fast_engine.py`.

## Quick start

```
git clone https://github.com/S-n-a-z/ai-chessathon-max-elo
cd ai-chessathon-max-elo
make setup
make play
```

That plays the engine against a baseline using the live 120 s + 0.5 s clock and prints the result.
Run `make zip` to create `submission.zip` with the Python runtime modules at its root and the
permitted knowledge data under `weights/`.

## Writing an agent

`agent.py` exposes the required interface. One function:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    return "e2e4"
```

The platform calls this function once per move and keeps the process alive for the full game.

```
make play                                          # one game, real time control
make arena                                         # 20 fast games, prints a score
make play FEN="<fen>"                              # start from a given position
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
uv run python -m harness.arena --opponent ../my-old-version --games 200
```

Anything your agent writes to stdout or stderr shows up under the result, so `print` debugging
works. The platform discards it during rated games and shows it in your validation log.

## The ladder

Measured with `harness/arena.py`. Beating greedy is a search. Beating minimax is a search plus an
evaluation worth searching with.

| Matchup | Games | Time control | Score |
|---|---|---|---|
| random vs greedy | 20 | 10 s + 0.1 s | 10.0% (+1 =2 -17) |
| greedy vs minimax | 6 | 120 s + 0.5 s | 0.0% (+0 =0 -6) |
| numba vs minimax | 6 | 10 s + 0.5 s | 66.7% (+2 =4 -0) |

- `baselines/random` plays a uniformly random legal move. It is what `agent.py` starts as.
- `baselines/greedy` searches one ply on material.
- `baselines/minimax` searches two plies on material and mobility, with no time management.
- `baselines/numba` is `minimax` with the evaluation jitted. It is barely stronger, which is
  the point: jitting a shallow search buys headroom, not depth. Read it for the warm-up call
  at the bottom, which is how you keep compilation off your clock.

## What's here

```
agent.py             required entry point and pure-Python safety fallback
fast_engine.py       original compiled move generator, evaluation, and search
opening.py           legal lookup for the shallow opening book
endgame.py           WDL/DTZ-optimal play in covered Syzygy endings
baselines/           random, greedy, minimax, numba; each is a directory with an agent.py
harness/runner.py    the process the platform runs your agent in
harness/referee.py   local referee (its legacy ply-cap adjudication differs from the live rules)
harness/rules.py     the event constants the harness enforces
harness/sandbox.py   the one process, spoken to as the platform speaks to a container
harness/play.py      one game between two agent directories
harness/arena.py     many games, with a score
harness/package.py   builds submission.zip with agent.py at the root
docs/IDEAS.md        where the strength actually comes from
tools/               reproducible benchmarking and evaluator-training utilities
tests/               legality, perft, clock, mate, and protocol regression tests
weights/             trained neural evaluator, opening book and Syzygy tables
```

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here for local regression testing, not to pre-validate an upload. It matches the
agent API and clock, but its legacy 300-ply material adjudication currently differs from the live
600-ply draw cap. Acceptance and match behaviour on the platform are authoritative.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
