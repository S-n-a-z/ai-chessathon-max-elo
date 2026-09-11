"""Download public AI Chessathon PGNs for reproducible local analysis.

The team and game pages are public.  Private validation/rated logs are deliberately not
requested: those remain available only through the team's authenticated dashboard.
"""

from __future__ import annotations

import argparse
import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import cast

TEAM_URL = "https://aichessathon.com/team/c3d8b5c4-e521-4341-8130-34d7e73234e5"
ROW_RE = re.compile(
    r'<tr data-href="/game/(?P<id>[0-9a-f-]+)[^"]*">.*?'
    r">Rated (?P<round>\d+).*?"
    r'match-colour".*?(?P<colour>White|Black)</span>.*?'
    r'match-verdict"[^>]*>(?P<result>Win|Draw|Loss|Void)</td>.*?'
    r'match-opening">(?P<opening>.*?)</td></tr>',
    re.DOTALL,
)
PGN_RE = re.compile(
    r'href="data:application/x-chess-pgn;charset=utf-8,(?P<data>[^"]+)"'
)


@dataclass(frozen=True)
class Match:
    round_number: int
    game_id: str
    colour: str
    result: str
    opening: str


def _read_url(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "BlockShark-public-game-sync/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return cast(bytes, response.read()).decode("utf-8")


def discover_matches(team_url: str) -> list[Match]:
    page = _read_url(team_url)
    matches = [
        Match(
            round_number=int(match.group("round")),
            game_id=match.group("id"),
            colour=match.group("colour").lower(),
            result=match.group("result").lower(),
            opening=html.unescape(match.group("opening")),
        )
        for match in ROW_RE.finditer(page)
    ]
    if not matches:
        raise RuntimeError("no public games found on the team page")
    return matches


def download_pgn(game_id: str) -> str:
    page = _read_url(f"https://aichessathon.com/game/{game_id}")
    match = PGN_RE.search(page)
    if match is None:
        raise RuntimeError(f"no PGN download found for game {game_id}")
    encoded = html.unescape(match.group("data"))
    return urllib.parse.unquote(encoded).replace("\r\n", "\n").rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-url", default=TEAM_URL)
    parser.add_argument("--games", type=Path, default=Path("games"))
    parser.add_argument("--from-round", type=int, default=0)
    parser.add_argument(
        "--include-wins",
        action="store_true",
        help="download wins too; by default only draws, losses, and voids are retained",
    )
    arguments = parser.parse_args()

    selected = [
        match
        for match in discover_matches(arguments.team_url)
        if match.round_number >= arguments.from_round
        and (arguments.include_wins or match.result != "win")
    ]
    arguments.games.mkdir(parents=True, exist_ok=True)
    for match in sorted(selected, key=lambda item: item.round_number):
        destination = arguments.games / f"round-{match.round_number}-{match.result}.pgn"
        destination.write_text(download_pgn(match.game_id), encoding="utf-8")
        print(
            f"round {match.round_number}: {match.result}, {match.colour}, "
            f"{match.opening} -> {destination}",
            flush=True,
        )


if __name__ == "__main__":
    main()
