# Predicting NBA Winners with an RNN

APS360 (University of Toronto, Fall 2026) course project by Malte Jessen-Richardsen.

The goal is to predict the winner of NBA regular-season games before tip-off. Each team is
represented by a sequence of its last *m* games: its offense, the defense it played and its
rest days. A shared-weight LSTM encodes both teams' sequences, and a bilinear layer compares
the home team's offense with the away team's defense, and the reverse.

## Quick start

```bash
pip install -r requirements.txt
python src/download_data.py      # ~200 MB into data/raw/
python src/build_games.py        # cleaning + per team-game measures
python src/build_sequences.py    # model-ready tensors (m = 10)
python src/data_stats.py         # split statistics -> reports/
```

To refresh the season currently being played (2026/27, our never-seen test data), run
`python src/download_data.py --seasons 2027` and rebuild.

## Data

| Source | Content | Used for |
|---|---|---|
| ESPN team box scores, mirrored by [sportsdataverse / hoopR](https://github.com/sportsdataverse/sportsdataverse-data) | one row per team and game, 2000/01–2026/27 | all games, scores, home/away, box-score totals |
| [`nba_data`](https://github.com/shufinskiy/nba_data): stats.nba.com play-by-play, 2007/08 | event-level play-by-play | rebuilding ESPN's broken 2007/08 rebounds |
| [`nba_data`](https://github.com/shufinskiy/nba_data): pbpstats possessions, 2007/08 | possession-level data | checking the possession estimate |

### Cleaning (`src/build_games.py`)

1. Keep regular-season games between two of the 30 NBA franchises. This removes All-Star,
   Rising Stars and other exhibition games, which ESPN also lists as regular season.
2. Drop duplicate game ids, and keep only games with exactly one home row and one away row.
3. **2007/08 rebounds:** in about 80% of ESPN's 2007/08 box scores the rebound counts are
   wrong (8.4 offensive / 11.2 defensive per game instead of about 11 / 30). They are rebuilt
   from the stats.nba.com play-by-play: each player's last `Off:x Def:y` counter, summed per
   team. On 2008/09, where ESPN is correct, this method matches ESPN in 99.5% of team-games.
   The 11 games that cannot be matched are dropped.
4. Use ESPN's `turnovers` column. `total_turnovers` double counts in 2001/02–2011/12.

### Per team-game measures (one time step = 16 values)

| Group | Measures |
|---|---|
| Offense (6) | offensive rating `100·PTS/poss`, eFG% `(FGM+0.5·3PM)/FGA`, TOV% `TOV/(FGA+0.44·FTA+TOV)`, ORB% `ORB/(ORB+opp DRB)`, FT rate `FTA/FGA`, 3PA rate `3PA/FGA` |
| Defense (6) | the same six measures for the opponent (allowed) |
| Context (4) | pace (possessions per game), rest days before the game (capped at 3), `was_home` (1/0), `opp_strength` (the opponent's season-to-date net rating before that game) |

Possessions are estimated as `FGA − ORB + TOV + 0.44·FTA`, averaged over both teams. Against
pbpstats' exact possession counts for 2007/08: r = 0.975, about 2.4 possessions too high on average.

**League-relative measures.** The league changed a lot between 2001 and 2026 (3PA rate 0.17 → 0.43).
The six measures and pace are therefore stored relative to the league average of the same season,
computed only from games before that date and shrunk towards the previous season's average early
in the season (`LEAGUE_PRIOR_GAMES = 200`). After this, the validation and test means of every
feature lie within 0.1 training standard deviations of the training mean (`reports/drift_check.csv`);
without it, the test 3PA rate would be 2.4 standard deviations above the training mean.

Net rating is not a step feature: it equals ORtg − opp_ORtg, which are already in the step.
The season-to-date net rating is used in `x` (see below).

### Sequences (`src/build_sequences.py`)

- A team's sequence holds its last *m* games of the same season (default `M_MAX = 10`),
  oldest first. Games where either team has played fewer than `M_MAX` games are skipped, so
  every model and every *m* sees the same games.
- `x` = pre-game season-to-date net rating difference, home rest days and away rest days.
- `match_seq` (20 per step, for the baselines) = home offense + away defense-allowed and away
  offense + home defense-allowed, each for the six measures (in z-scores, so a high value
  means the attacking team should do well), plus the four context values of both teams.
- All features are standardized with the mean and std of the **training** team-games only.