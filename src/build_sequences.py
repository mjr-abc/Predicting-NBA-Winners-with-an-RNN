"""Turn team_games.csv into model-ready tensors.

For every game in which both teams have already played at least M_MAX games
that season, we store
  home_seq  (M_MAX, 16)  the home team's last M_MAX games, oldest first
  away_seq  (M_MAX, 16)  the away team's last M_MAX games
  match_seq (M_MAX, 20)  precomputed matchup vector per step (for the baselines)
  x         (3,)         pre-game season-to-date net rating difference, home rest, away rest
  y                      1 if the home team won
Shorter sequences (m < M_MAX) are obtained by slicing the last m steps, so all
models see exactly the same games.

All features are standardized with means/stds from TRAINING team-games only.
(The measures are already relative to the league average of their season, so the
training statistics also fit the later seasons.)

Matchup step (20): for each of the six measures k,
  home offense k + away defense-allowed k   (both z-scores; high = home expected to do well in k)
  away offense k + home defense-allowed k
  plus the four context values (pace, rest, was_home, opp_strength) of both teams.
The net rating difference is not added: it equals the difference of the two ortg sums.
"""
import json

import numpy as np
import pandas as pd

from config import PROCESSED, OFF, DEF, CONTEXT, TEAM_STEP, M_MAX


def main(m=M_MAX):
    tg = pd.read_csv(PROCESSED / "team_games.csv", parse_dates=["date"])
    tg = tg.sort_values(["team_id", "season", "date"]).reset_index(drop=True)

    # standardize with training statistics only
    train = tg[tg.split == "train"]
    mu, sd = train[TEAM_STEP].mean(), train[TEAM_STEP].std()
    net_mu, net_sd = train.pre_net.mean(), train.pre_net.std()
    Z = ((tg[TEAM_STEP] - mu) / sd).to_numpy(np.float32)
    pre_net_z = ((tg.pre_net - net_mu) / net_sd).to_numpy(np.float32)
    rest_z = Z[:, TEAM_STEP.index("rest")]

    # sequence of the previous m games for every team-game row
    seq = {}
    for _, idx in tg.groupby(["team_id", "season"]).indices.items():
        idx = np.sort(idx)
        for j in range(m, len(idx)):
            seq[idx[j]] = Z[idx[j - m:j]]

    key = {(g, t): i for i, (g, t) in enumerate(zip(tg.game_id, tg.team_id))}
    games = pd.read_csv(PROCESSED / "games.csv", parse_dates=["date"])
    rows = []
    for g in games.itertuples():
        ih, ia = key[(g.game_id, g.home_id)], key[(g.game_id, g.away_id)]
        if ih in seq and ia in seq:
            rows.append((g, ih, ia))
    print(f"{len(rows)} of {len(games)} games have >= {m} prior games for both teams")

    H = np.stack([seq[ih] for _, ih, _ in rows])
    A = np.stack([seq[ia] for _, _, ia in rows])
    o = [TEAM_STEP.index(c) for c in OFF]
    d = [TEAM_STEP.index(c) for c in DEF]
    c = [TEAM_STEP.index(k) for k in CONTEXT]
    match = np.concatenate([
        H[:, :, o] + A[:, :, d],                     # home offense vs away defense
        A[:, :, o] + H[:, :, d],                     # away offense vs home defense
        H[:, :, c], A[:, :, c],                      # context of both teams
    ], axis=2).astype(np.float32)
    x = np.array([[pre_net_z[ih] - pre_net_z[ia], rest_z[ih], rest_z[ia]]
                  for _, ih, ia in rows], dtype=np.float32)

    meta = pd.DataFrame({
        "game_id": [g.game_id for g, _, _ in rows], "season": [g.season for g, _, _ in rows],
        "split": [g.split for g, _, _ in rows], "date": [g.date for g, _, _ in rows],
        "home": [g.home for g, _, _ in rows], "away": [g.away for g, _, _ in rows],
        "home_win": [g.home_win for g, _, _ in rows],
    })
    np.savez_compressed(PROCESSED / f"sequences_m{m}.npz", home_seq=H, away_seq=A,
                        match_seq=match, x=x, y=meta.home_win.to_numpy(np.float32),
                        game_id=meta.game_id.to_numpy(), season=meta.season.to_numpy(),
                        split=meta.split.to_numpy().astype(str))
    meta.to_csv(PROCESSED / f"sequences_m{m}_meta.csv", index=False)
    json.dump({"features": TEAM_STEP, "mean": mu.tolist(), "std": sd.tolist(),
               "pre_net_mean": float(net_mu), "pre_net_std": float(net_sd)},
              open(PROCESSED / "scaler.json", "w"), indent=2)
    print(f"saved home/away seq {H.shape}, match seq {match.shape}, x {x.shape}")


if __name__ == "__main__":
    main()