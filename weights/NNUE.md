# Original trained evaluator

This network was initialized and trained from scratch by this project. No published chess
network, pretrained weights, or third-party engine implementation is included.

The source is the CC0 Lichess evaluation database, using the deepest available first principal
variation at depth 24 or greater. Static-evaluation training excludes checked positions and
positions whose selected teacher move captures, promotes, or gives check. Source FENs and
white-centric scores remain unchanged. Exact positions have a stable canonical-FEN hash split;
published competition positions are excluded. Related positions can still cross splits because
the dump does not identify source games.

The cache contains 908,604 positions: 727,142 training, 90,852 validation and 90,610 sealed test.
The test split was not used to choose this candidate. The 32-channel symmetric sparse model
shares weights between both colour perspectives. Features combine a horizontally canonicalized
king bucket, piece colour/type and square. Clipped-ReLU accumulators feed an antisymmetric
linear output and a learned tempo term. It predicts a correction to our classical evaluation.
The runtime blends half of that correction with the classical score.

Training used seed 20260907, AdamW, batch 1024, learning rate 0.002, weight decay 0.000001,
cosine scheduling and SmoothL1 loss with a 140-centipawn transition. Best validation epoch was
11 of 16 completed. Quantized validation teacher MAE was 168.57 cp versus 239.62 cp for the
classical evaluator on the same positions. Evaluation loss is not an Elo estimate.

`NNUE_TRAINING.json` records training history and source/code hashes. `NNUE_CACHE.json` records
filter settings, data splits and shard hashes. The readable training code remains in the project:
`tools/build_lichess_eval_cache.py`, `tools/train_king_nnue.py`, and
`tools/train_king_nnue_cache.py`. `neural.py` and `neural_full.py` implement original inference.
Feature weights are int8, accumulators int32, output weights int16, with explicit scales.
King-indexed caches retain accumulator state across local move/unmove operations.

Training source documentation: https://database.lichess.org/#evals
