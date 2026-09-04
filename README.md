# AI Chessathon: maximum-Elo engine

An original, self-contained chess engine built for the
[AI Chessathon](https://aichessathon.com). The current engine combines:

- iterative deepening with aspiration windows;
- principal-variation alpha-beta search;
- quiescence search for tactically stable leaf positions;
- transposition-table, killer-move, history, and MVV-LVA move ordering;
- null-move pruning, razoring, reverse futility pruning, and late-move reductions;
- a tapered middlegame/endgame evaluation covering material, mobility, pawn structure,
  passed pawns, bishop pair, rook files, king safety, and tempo; and
- clock-aware hard deadlines with a legal fallback move; and
- cancellable opponent-time pondering, which the competition rules explicitly permit.

`fast_engine.py` contains an original 0x88 board, legal move generator, evaluation, and recursive
search compiled by Numba. `agent.py` provides the required entry point and retains a readable
pure-Python fallback. The submission does not bundle, invoke, port, or translate Stockfish or
another third-party engine. The target is the strongest legal entry we can iteratively measure,
not a claim of Stockfish-equivalent playing strength.

The compiled search visits roughly fifteen times as many nodes per move as the initial
python-chess search in local fixed-time measurements. In a four-game A/B check at 5 s + 0.1 s,
the final pondering build scored +3 =1 -0 against the otherwise identical non-pondering build.
Earlier smoke tests scored +2 =0 -0 against the supplied minimax baseline and +2 =0 -0 against
the previous pure-Python engine. A separate four-game spot check against Stockfish's 1800-Elo
limited mode scored +2 =0 -2. These are regression checks, not statistically meaningful
rating estimates or evidence of Stockfish-equivalent strength.

A small evaluator trained from 30,000 locally generated Stockfish-labelled positions was also
tested. It lost both direct A/B games against the handcrafted evaluator, so its weights were
removed from the submission. Only measured improvements are kept.

## Quick start

```
git clone https://github.com/S-n-a-z/ai-chessathon-max-elo
cd ai-chessathon-max-elo
make setup
make play
```

That plays the engine against a baseline over a full 120 s + 0.5 s game and prints the result.
Run `make zip` to create `submission.zip` with `agent.py` and `fast_engine.py` at its root.

## Writing an agent

`agent.py` is the whole submission. One function:

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
fast_engine.py       original compiled move generator, evaluation, search, and pondering
baselines/           random, greedy, minimax, numba; each is a directory with an agent.py
harness/runner.py    the process the platform runs your agent in
harness/referee.py   the clock, legality, draw and adjudication rules
harness/rules.py     the event constants the harness enforces
harness/sandbox.py   the one process, spoken to as the platform speaks to a container
harness/play.py      one game between two agent directories
harness/arena.py     many games, with a score
harness/package.py   builds submission.zip with agent.py at the root
docs/IDEAS.md        where the strength actually comes from
tools/               reproducible benchmarking and evaluator-training utilities
tests/               legality, perft, clock, mate, and protocol regression tests
```

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here so your games are honest, not so you can pre-validate an upload. Acceptance
happens on the platform, and the validation log on your dashboard is the authority on it.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
