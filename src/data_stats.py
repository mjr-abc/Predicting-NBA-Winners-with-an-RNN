"""Statistics for the progress report: games and home-win rate per split,
per-season feature means, a drift check, and one cleaned training example."""
import numpy as np
import pandas as pd

from config import PROCESSED, SPLITS, TEAM_STEP, M_MAX, ROOT

ORDER = ["train", "val", "excluded", "retrain", "test", "new"]


def season_label(s):
    return f"{s-1}/{str(s)[2:]}"


def main():
    games = pd.read_csv(PROCESSED / "games.csv")
    meta = pd.read_csv(PROCESSED / f"sequences_m{M_MAX}_meta.csv")
    tg = pd.read_csv(PROCESSED / "team_games.csv")

    rows = []
    for s in ORDER:
        g, m = games[games.split == s], meta[meta.split == s]
        rows.append({"split": s,
                     "seasons": f"{season_label(min(SPLITS[s]))}-{season_label(max(SPLITS[s]))}",
                     "games (all)": len(g), f"games (m>={M_MAX})": len(m),
                     "home-win rate": round(m.home_win.mean(), 3)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    per_season = tg.groupby("season")[TEAM_STEP].mean().round(3)
    per_season["home_win"] = games.groupby("season").home_win.mean().round(3)
    per_season["games"] = games.groupby("season").size()

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "split_stats.csv", index=False)
    per_season.to_csv(out / "per_season_means.csv")
    print(per_season[["games", "home_win", "ortg", "efg", "orb_pct", "par3", "pace"]].to_string())

    # drift check: mean of each feature in val/test, in units of the training std.
    # Values near 0 mean the training standardization also fits the later seasons.
    train = tg[tg.split == "train"]
    drift = pd.DataFrame({s: (tg[tg.split == s][TEAM_STEP].mean() - train[TEAM_STEP].mean())
                             / train[TEAM_STEP].std() for s in ["val", "test"]}).round(2)
    drift.to_csv(out / "drift_check.csv")
    print("\nDrift check (val/test mean in training standard deviations):")
    print(drift.T.to_string())

    # one cleaned training example (unstandardized, for the report)
    ex = meta[meta.split == "train"].iloc[1000]
    home = tg[(tg.team == ex.home) & (tg.season == ex.season) & (tg.date < ex.date)].tail(M_MAX)
    print(f"\nExample: {ex.date} {ex.away} @ {ex.home}, home_win={ex.home_win}")
    print(home[["date", "opp_id", "home_away"] + TEAM_STEP].round(3).to_string(index=False))


if __name__ == "__main__":
    main()