# Syzygy three- and four-piece tables

This directory contains the complete WDL and DTZ Syzygy tables for standard-chess positions
with three or four pieces. They were downloaded from the public Lichess tablebase mirror:

- https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/
- https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/

The AI Chessathon rules explicitly permit endgame tablebases as shipped data. No network access
is used at runtime. `endgame.py` probes these files through the preinstalled `chess.syzygy` API.

Additional five-piece WDL/DTZ coverage: rook-and-pawn versus rook, queen-and-rook versus rook, bishop-and-rook versus rook, knight-and-rook versus rook. KRRvKR has WDL only as a promotion dependency; that root class uses search. All necessary WDL capture/promotion dependencies are present. Sizes, source URLs and SHA256 hashes are recorded in ROOK_TABLES.json.
