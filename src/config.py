"""Shared paths, season splits and feature names.

Seasons are labelled by the calendar year in which they END
(e.g. 2016 = the 2015/16 season), following the ESPN / hoopR convention.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

FIRST_SEASON = 2001          # 2000/01
LAST_SEASON = 2027           # 2026/27 (in progress -> new, never-seen data)

SPLITS = {
    "train":      list(range(2001, 2017)),   # 2000/01-2015/16
    "val":        list(range(2017, 2020)),   # 2016/17-2018/19
    "excluded":   [2020, 2021],              # bubble / limited fans
    "retrain":    [2022],                    # only used when retraining before the test run
    "test":       list(range(2023, 2027)),   # 2022/23-2025/26
    "new":        [2027],                    # 2026/27, collected during the project
}

def split_of(season: int) -> str:
    for name, seasons in SPLITS.items():
        if season in seasons:
            return name
    raise ValueError(season)

# ESPN team ids 1..30 are the 30 NBA franchises; larger ids are All-Star / Rising Stars teams.
NBA_TEAM_IDS = set(range(1, 31))

# Per team-game measures (one RNN time step for one team).
# OFF, DEF and pace are stored RELATIVE to the league average of that season (see build_games.py).
OFF = ["ortg", "efg", "tov_pct", "orb_pct", "ftr", "par3"]          # own offense
DEF = [f"opp_{c}" for c in OFF]                                     # same quantities allowed
CONTEXT = ["pace", "rest", "was_home", "opp_strength"]              # tempo, fatigue, venue, schedule
TEAM_STEP = OFF + DEF + CONTEXT                                     # 16 values
STRENGTH_STEP = ["ortg", "opp_ortg", "rest", "was_home", "opp_strength"]  # strength-only baseline

# League averages early in a season are shrunk towards the previous season's average,
# weighted as if the previous season contributed this many team-games.
LEAGUE_PRIOR_GAMES = 200

M_MAX = 10   # largest sequence length tested; games before each team's 11th game are skipped
REST_CAP = 3