# Opening preparation

13,456 unique opening positions. The original 5,474-position CC0 Lichess opening book is supplemented by 8,000 offline analyses with Stockfish18 at 50,000 nodes and two principal variations per position. Preparation starts from public Chessathon games through round100 and explores up to eight more plies, always at move20 or earlier. The runtime rejects book lookups after move20.

The teacher executable, teacher network, analysis scores and source PGNs are not shipped. Only permitted opening moves are stored. File metadata records the teacher identity, public source hashes and preparation settings.

Sources: https://github.com/lichess-org/chess-openings and public team game PGNs from https://aichessathon.com/
