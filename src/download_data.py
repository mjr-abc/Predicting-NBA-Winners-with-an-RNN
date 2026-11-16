"""Download the raw data into data/raw/.

1. ESPN team box scores (one row per team and game), 2000/01-2026/27,
   mirrored by the sportsdataverse / hoopR project.
2. stats.nba.com play-by-play and pbpstats possessions for 2007/08 from the
   nba_data collection (shufinskiy). ESPN's rebound counts for 2007/08 are
   broken, so they are rebuilt from the play-by-play (see build_games.py).

Usage:  python src/download_data.py                    # download everything + overview
        python src/download_data.py --seasons 2027     # refresh the current season only
        python src/download_data.py --overview-only    # only print the data overview
"""
import argparse
import tarfile
import urllib.request

from config import RAW, FIRST_SEASON, LAST_SEASON

ESPN_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
            "espn_nba_team_boxscores/team_box_{season}.parquet")
NBA_DATA_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets/{name}.tar.xz"


def fetch(url, dest, force=False):
    if dest.exists() and not force:
        print(f"  exists  {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get     {url}")
    urllib.request.urlretrieve(url, dest)


def season_label(s):
    """ESPN season 2008 -> '2007/08'."""
    return f"{s - 1}/{str(s)[2:]}"


def print_overview():
    """Print a per-season summary of the raw ESPN data and flag suspicious values."""
    import pandas as pd   # imported here so the download itself needs no extra packages

    files = sorted((RAW / "espn").glob("team_box_*.parquet"))
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    print("\n" + "=" * 78)
    print("DATA OVERVIEW (raw ESPN team box scores)")
    print("=" * 78)
    print(f"{len(raw):,} rows in {len(files)} files, {raw.shape[1]} columns")
    print("Rows per season type (1 = preseason, 2 = regular season, 3 = playoffs, 5 = play-in):")
    print("  " + raw.season_type.value_counts().sort_index().to_string().replace("\n", "\n  "))

    reg = raw[raw.season_type == 2]
    nba = reg[reg.team_id.between(1, 30) & reg.opponent_team_id.between(1, 30)]
    other = reg[~reg.index.isin(nba.index)]
    print(f"\nRegular-season rows not between two NBA teams (All-Star etc.): {len(other)}")
    print("  teams: " + ", ".join(sorted(other.team_abbreviation.unique())))

    # per-season averages per team and game
    home = nba[nba.team_home_away == "home"]
    tab = nba.groupby("season").agg(
        games=("game_id", "nunique"),
        pts=("team_score", "mean"),
        fga=("field_goals_attempted", "mean"),
        fg3a=("three_point_field_goals_attempted", "mean"),
        orb=("offensive_rebounds", "mean"),
        drb=("defensive_rebounds", "mean"),
        tov=("turnovers", "mean"),
        tot_tov=("total_turnovers", "mean"),
    )
    tab["home_win"] = home.groupby("season").team_winner.mean()
    tab.index = [season_label(s) for s in tab.index]

    # flag values far outside the normal range
    flags = []
    for s, r in tab.iterrows():
        f = []
        if r.drb < 25 or r.orb < 9:
            f.append("REBOUNDS?")
        if r.tot_tov > 1.5 * r.tov:
            f.append("total_tov double counted")
        flags.append(", ".join(f))
    tab["check"] = flags

    print("\nAverages per team and game (regular season, NBA teams only):")
    print(tab.round({"pts": 1, "fga": 1, "fg3a": 1, "orb": 1, "drb": 1,
                     "tov": 1, "tot_tov": 1, "home_win": 3}).to_string())

    # zoom in on the broken season
    s08 = nba[nba.season == 2008]
    bad = (s08.defensive_rebounds < 20).mean()
    print(f"\n2007/08: {bad:.0%} of team-games have fewer than 20 defensive rebounds "
          f"(a normal team gets 30-35). Examples:")
    cols = ["game_date", "team_abbreviation", "opponent_team_abbreviation", "team_score",
            "field_goals_attempted", "offensive_rebounds", "defensive_rebounds"]
    print(s08[cols].head(6).to_string(index=False))
    print("\n2008/09 for comparison (correct):")
    print(nba[nba.season == 2009][cols].head(6).to_string(index=False))
    print("\nThe 2007/08 rebounds are rebuilt from play-by-play in build_games.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*",
                    default=list(range(FIRST_SEASON, LAST_SEASON + 1)))
    ap.add_argument("--overview-only", action="store_true",
                    help="skip the download and only print the data overview")
    args = ap.parse_args()

    if args.overview_only:
        print_overview()
        return

    print("ESPN team box scores")
    for s in args.seasons:
        # the current season changes daily, so always re-download it
        fetch(ESPN_URL.format(season=s), RAW / "espn" / f"team_box_{s}.parquet",
              force=(s == LAST_SEASON))

    print("nba_data 2007/08 (rebound repair)")
    for name in ["nbastats_2007", "pbpstats_2007"]:
        csv = RAW / "nba_data" / f"{name}.csv"
        if csv.exists():
            print(f"  exists  {csv.name}")
            continue
        arc = RAW / "nba_data" / f"{name}.tar.xz"
        fetch(NBA_DATA_URL.format(name=name), arc)
        with tarfile.open(arc) as t:
            try:
                t.extractall(arc.parent, filter="data")
            except TypeError:                   # Python < 3.12 has no filter argument
                t.extractall(arc.parent)
        arc.unlink()

    print_overview()


if __name__ == "__main__":
    main()