"""Clean the raw box scores and compute per team-game measures.

Output
  data/processed/team_games.csv  one row per team and game (16 step features + ids)
  data/processed/games.csv       one row per game (home/away ids, label, split)

Steps
  1. Keep regular-season games (ESPN season_type == 2) between two of the 30 NBA
     franchises (drops All-Star, Rising Stars and other exhibition games).
  2. Drop duplicate game ids and keep only games with exactly one home and one away row.
  3. Replace ESPN's broken 2007/08 rebound counts with counts rebuilt from the
     stats.nba.com play-by-play (each player's final 'Off:x Def:y' counter).
  4. Compute possessions and the per-game measures (formulas below).
  5. Add rest days, venue (was_home) and opponent strength (opponent's net rating
     before that game).
  6. Express the six measures (offense and defense) and pace relative to the league
     average of that season up to the day before the game, so that 2001 and 2026
     teams are on the same scale.
"""
import glob

import numpy as np
import pandas as pd

from config import (RAW, PROCESSED, NBA_TEAM_IDS, REST_CAP, OFF, TEAM_STEP,
                    LEAGUE_PRIOR_GAMES, split_of)

KEEP = {
    "game_id": "game_id", "season": "season", "game_date": "date",
    "team_id": "team_id", "team_abbreviation": "team", "team_home_away": "home_away",
    "opponent_team_id": "opp_id", "team_score": "pts",
    "field_goals_made": "fgm", "field_goals_attempted": "fga",
    "three_point_field_goals_made": "fg3m", "three_point_field_goals_attempted": "fg3a",
    "free_throws_made": "ftm", "free_throws_attempted": "fta",
    "offensive_rebounds": "orb", "defensive_rebounds": "drb",
    # 'turnovers' is consistent across seasons; 'total_turnovers' double counts in 2001/02-2011/12
    "turnovers": "tov",
}


def load_espn():
    files = sorted(glob.glob(str(RAW / "espn" / "team_box_*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n0 = len(df)
    df = df[df.season_type == 2]
    df = df[df.team_id.isin(NBA_TEAM_IDS) & df.opponent_team_id.isin(NBA_TEAM_IDS)]
    df = df[list(KEEP)].rename(columns=KEEP)
    df = df.drop_duplicates(["game_id", "team_id"])
    ok = df.groupby("game_id").home_away.agg(lambda s: sorted(s) == ["away", "home"])
    df = df[df.game_id.isin(ok[ok].index)]
    df["date"] = pd.to_datetime(df.date)
    print(f"ESPN rows: {n0} raw -> {len(df)} regular-season NBA team-games")
    return df


def rebounds_from_pbp(pbp_csv, dates_csv):
    """Team offensive/defensive rebounds per game from stats.nba.com play-by-play."""
    p = pd.read_csv(pbp_csv, usecols=["GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "PERIOD",
                                      "PLAYER1_ID", "PLAYER1_TEAM_ID", "HOMEDESCRIPTION",
                                      "VISITORDESCRIPTION", "SCORE"])
    side = np.where(p.HOMEDESCRIPTION.notna(), "home",
                    np.where(p.VISITORDESCRIPTION.notna(), "away", None))
    p["side"] = side
    rb = p[(p.EVENTMSGTYPE == 4) & p.PLAYER1_TEAM_ID.notna()].copy()   # player rebounds only
    text = rb.HOMEDESCRIPTION.where(rb.side == "home", rb.VISITORDESCRIPTION).fillna("")
    cnt = text.str.extract(r"Off:(\d+) Def:(\d+)").astype(float)
    rb["off"], rb["dfn"] = cnt[0], cnt[1]
    per_player = rb.dropna(subset=["off"]).groupby(["GAME_ID", "PLAYER1_ID"]).agg(
        off=("off", "max"), dfn=("dfn", "max"), side=("side", "first"))
    t = per_player.groupby(["GAME_ID", "side"])[["off", "dfn"]].sum().unstack()
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    score = (p.dropna(subset=["SCORE"]).sort_values(["GAME_ID", "PERIOD", "EVENTNUM"])
             .groupby("GAME_ID").SCORE.last().str.split(" - ", expand=True).astype(int))
    t["away_pts"], t["home_pts"] = score[0], score[1]
    dates = (pd.read_csv(dates_csv, usecols=["GAMEID", "GAMEDATE"]).drop_duplicates()
             .set_index("GAMEID").GAMEDATE)
    t["date"] = pd.to_datetime(dates.reindex(t.index).values)
    return t.reset_index()


def repair_2008_rebounds(df):
    pbp = rebounds_from_pbp(RAW / "nba_data" / "nbastats_2007.csv",
                            RAW / "nba_data" / "pbpstats_2007.csv")
    s = df[df.season == 2008]
    home = s[s.home_away == "home"][["game_id", "date", "pts"]]
    away = s[s.home_away == "away"][["game_id", "pts"]].rename(columns={"pts": "away_pts"})
    g = home.merge(away, on="game_id").rename(columns={"pts": "home_pts"})
    m = g.merge(pbp, on=["date", "home_pts", "away_pts"], how="left")
    m = m[~m.game_id.duplicated(keep=False)]           # drop ambiguous date/score matches
    m = m.dropna(subset=["off_home"]).set_index("game_id")
    fixed = 0
    for side in ["home", "away"]:
        idx = (df.season == 2008) & (df.home_away == side) & df.game_id.isin(m.index)
        df.loc[idx, "orb"] = m.loc[df.loc[idx, "game_id"], f"off_{side}"].values
        df.loc[idx, "drb"] = m.loc[df.loc[idx, "game_id"], f"dfn_{side}"].values
        fixed += idx.sum()
    unmatched = df[(df.season == 2008) & ~df.game_id.isin(m.index)].game_id.unique()
    df = df[~df.game_id.isin(unmatched)]                # drop the few games we cannot repair
    print(f"2007/08 rebounds rebuilt for {fixed} team-games; dropped {len(unmatched)} unmatched games")
    return df


def add_measures(df):
    opp = df[["game_id", "team_id"] + [c for c in df.columns if c in
              ("pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "orb", "drb", "tov")]]
    opp = opp.rename(columns=lambda c: c if c == "game_id" else f"o_{c}")
    df = df.merge(opp, on="game_id")
    df = df[df.team_id != df.o_team_id].drop(columns="o_team_id")

    # Possessions (Kubatko et al., 2007): FGA - ORB + TOV + 0.44 FTA, averaged over both teams
    poss_t = df.fga - df.orb + df.tov + 0.44 * df.fta
    poss_o = df.o_fga - df.o_orb + df.o_tov + 0.44 * df.o_fta
    df["poss"] = (poss_t + poss_o) / 2

    def four(prefix=""):
        g = lambda c: df[prefix + c]
        o = "o_" if prefix == "" else ""          # the other team's columns
        return {
            "ortg":    100 * g("pts") / df.poss,
            "efg":     (g("fgm") + 0.5 * g("fg3m")) / g("fga"),
            "tov_pct": g("tov") / (g("fga") + 0.44 * g("fta") + g("tov")),
            "orb_pct": g("orb") / (g("orb") + df[o + "drb"]),
            "ftr":     g("fta") / g("fga"),
            "par3":    g("fg3a") / g("fga"),
        }
    for k, v in four("").items():
        df[k] = v
    for k, v in four("o_").items():
        df[f"opp_{k}"] = v
    df["pace"] = df.poss                        # possessions per game (not adjusted for overtime)
    df["net_rtg"] = df.ortg - df.opp_ortg
    df["win"] = (df.pts > df.o_pts).astype(int)
    return df


def add_rest(df):
    df = df.sort_values(["team_id", "season", "date"])
    prev = df.groupby(["team_id", "season"]).date.shift()
    rest = (df.date - prev).dt.days - 1
    df["rest"] = rest.fillna(REST_CAP).clip(0, REST_CAP)
    df["game_no"] = df.groupby(["team_id", "season"]).cumcount() + 1   # 1 = season opener
    return df


def add_context(df):
    """Venue and opponent strength for every team-game (only pre-game information)."""
    df = df.sort_values(["team_id", "season", "date"])
    df["was_home"] = (df.home_away == "home").astype(int)

    # season-to-date net rating BEFORE this game (0 = league average if no games yet)
    g = df.groupby(["team_id", "season"]).net_rtg
    n_before = g.cumcount()
    df["pre_net"] = ((g.cumsum() - df.net_rtg) / n_before.replace(0, np.nan)).fillna(0.0)

    # opponent strength = the opponent's pre_net in the same game
    opp = df[["game_id", "team_id", "pre_net"]].rename(
        columns={"team_id": "opp_id", "pre_net": "opp_strength"})
    return df.merge(opp, on=["game_id", "opp_id"])


def add_league_relative(df):
    """Subtract the league average of the same season, using only games before that date.

    Early in a season few games are available, so the average is shrunk towards the
    previous season's average (weighted like LEAGUE_PRIOR_GAMES team-games). For the very
    first season (2000/01, training data) the prior is that season's own average.
    The league average of an 'allowed' measure (opp_efg, ...) equals the league average
    of the measure itself, so offense and defense use the same reference.
    """
    base = OFF + ["pace"]
    day = df.groupby(["season", "date"])[base].sum()
    n_day = df.groupby(["season", "date"]).size()
    sum_before = day.groupby(level="season").cumsum() - day        # strictly before that date
    n_before = n_day.groupby(level="season").cumsum() - n_day

    season_mean = df.groupby("season")[base].mean()
    prior = season_mean.shift(1).fillna(season_mean)                # previous season's average
    prior = prior.reindex(sum_before.index.get_level_values("season")).to_numpy()

    k = LEAGUE_PRIOR_GAMES
    league = (sum_before.to_numpy() + k * prior) / (n_before.to_numpy()[:, None] + k)
    league = pd.DataFrame(league, index=sum_before.index, columns=[f"lg_{c}" for c in base])
    df = df.merge(league.reset_index(), on=["season", "date"])

    for c in OFF:
        df[c] = df[c] - df[f"lg_{c}"]
        df[f"opp_{c}"] = df[f"opp_{c}"] - df[f"lg_{c}"]
    df["pace"] = df["pace"] - df["lg_pace"]
    return df.drop(columns=[f"lg_{c}" for c in base])


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_espn()
    df = repair_2008_rebounds(df)
    df = add_measures(df)
    df = add_rest(df)
    df = add_context(df)
    df = add_league_relative(df)
    df["split"] = df.season.map(split_of)

    ids = ["game_id", "season", "split", "date", "team_id", "team", "opp_id", "home_away",
           "game_no", "pts", "o_pts", "win", "net_rtg", "pre_net"]
    tg = df[ids + TEAM_STEP].sort_values(["date", "game_id", "home_away"])
    assert tg[TEAM_STEP + ["pre_net"]].notna().all().all(), "missing feature values"
    tg.to_csv(PROCESSED / "team_games.csv", index=False)

    h = tg[tg.home_away == "home"].set_index("game_id")
    a = tg[tg.home_away == "away"].set_index("game_id")
    games = pd.DataFrame({
        "season": h.season, "split": h.split, "date": h.date,
        "home_id": h.team_id, "home": h.team, "away_id": a.team_id.reindex(h.index),
        "away": a.team.reindex(h.index), "home_pts": h.pts, "away_pts": h.o_pts,
        "home_game_no": h.game_no, "away_game_no": a.game_no.reindex(h.index),
        "home_win": h.win,
    }).reset_index().sort_values(["date", "game_id"])
    games.to_csv(PROCESSED / "games.csv", index=False)
    print(f"wrote {len(tg)} team-games and {len(games)} games to {PROCESSED}")


if __name__ == "__main__":
    main()