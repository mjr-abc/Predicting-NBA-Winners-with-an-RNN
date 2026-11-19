"""Baseline models, trained on the training seasons and evaluated on validation.

  Home team (reference)  always predicts the training home-win rate (chance level)
  Net ratings and rest   logistic regression on x: the pre-game season-to-date
                         net rating difference and both teams' rest days
  Static averages        logistic regression on the 20 matchup features averaged
                         over the last m games + x (net rating difference, both rest days)

The test seasons are not touched. The only tuned setting is the L2 strength C of the
logistic regressions, chosen by validation log-loss.

Outputs (in reports/)
  baseline_results.csv        log-loss (primary), Brier, accuracy on train and validation
  baseline_bootstrap.csv      95% bootstrap intervals for log-loss differences
  baseline_confidence.csv     accuracy by predicted confidence (qualitative)
  baseline_coefficients.csv   static-averages coefficients, largest first (qualitative)
  baseline_val_predictions.csv  validation predictions of all baselines
  figures/calibration_baselines.png   calibration plot (qualitative)

  baseline_m_sweep.csv              static averages for several m (only with --sweep)
  figures/baseline_m_sweep.png      validation log-loss vs. m (only with --sweep)

Usage:  python src/baselines.py                       # m = 10
        python src/baselines.py --m 5                 # average over the last 5 games instead
        python src/baselines.py --sweep 1 3 5 7 10    # compare several m, then stop

All values of m are evaluated on the SAME games (those where both teams have played at
least M_MAX games), so the results are directly comparable. To test m > M_MAX, raise
M_MAX in config.py and rerun build_sequences.py; this removes more early-season games,
so results for different M_MAX values must not be compared with each other.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import PROCESSED, ROOT, OFF, CONTEXT, M_MAX

C_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
N_BOOT = 1000
SEED = 0

MATCH_NAMES = ([f"home off + away def: {k}" for k in OFF]
               + [f"away off + home def: {k}" for k in OFF]
               + [f"home {c}" for c in CONTEXT] + [f"away {c}" for c in CONTEXT])
X_NAMES = ["net rating diff (pre-game)", "home rest", "away rest"]


def static_features(d, m):
    """20 matchup features averaged over the last m games + x (3 values) = 23 inputs."""
    static = d["match_seq"][:, -m:, :].mean(axis=1)
    return np.concatenate([static, d["x"]], axis=1)


def load(m):
    d = np.load(PROCESSED / f"sequences_m{M_MAX}.npz")
    feats = {
        "net": d["x"],                                        # x: net rating difference + both rest days
        "static": static_features(d, m),                      # 20 averaged + 3 = 23 inputs
    }
    meta = pd.read_csv(PROCESSED / f"sequences_m{M_MAX}_meta.csv")
    return feats, d["y"], d["split"], meta, d


def scores(y, p):
    return {"log_loss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p),
            "accuracy": accuracy_score(y, p > 0.5)}


def fit_lr(X, y, tr, va, verbose=True):
    """Fit a standardized logistic regression for every C; keep the best on validation."""
    best = None
    for c in C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        model.fit(X[tr], y[tr])
        ll = log_loss(y[va], model.predict_proba(X[va])[:, 1])
        if verbose:
            print(f"    C={c:<6} val log-loss {ll:.4f}")
        if best is None or ll < best[0]:
            best = (ll, c, model)
    if verbose:
        print(f"    -> chosen C={best[1]}")
    return best[2], best[1]


def bootstrap_diff(y, p_a, p_b, n=N_BOOT, seed=SEED):
    """95% interval of mean log-loss(a) - log-loss(b) over resampled validation games."""
    eps = 1e-15
    ll = lambda p: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
    diff = ll(p_a) - ll(p_b)                                   # per-game difference
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    boot = diff[idx].mean(axis=1)
    return diff.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def confidence_table(y, p):
    """Accuracy by how confident the prediction is.

    confidence = predicted probability of the team the model picks (home or away).
    For a well-calibrated model, accuracy in each group matches the mean confidence.
    """
    conf = np.maximum(p, 1 - p)
    bins = [0.5, 0.55, 0.65, 0.75, 1.0]
    labels = ["toss-up (0.50-0.55)", "lean (0.55-0.65)", "favourite (0.65-0.75)", "strong (>0.75)"]
    cut = pd.cut(conf, bins, labels=labels, include_lowest=True)
    df = pd.DataFrame({"bin": cut, "correct": (p > 0.5) == (y == 1), "conf": conf,
                       "picks_home": p > 0.5})
    return df.groupby("bin", observed=False).agg(
        games=("correct", "size"), mean_confidence=("conf", "mean"),
        accuracy=("correct", "mean"), share_home_picks=("picks_home", "mean")).round(3)


def calibration_plot(y, preds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Net ratings and rest": "#2a78d6", "Static averages": "#eb6834"}
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=200)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#9a9a96", label="perfect calibration")
    for name, p in preds.items():
        b = np.clip(np.digitize(p, edges) - 1, 0, 9)
        rows = [(p[b == i].mean(), y[b == i].mean(), (b == i).sum())
                for i in range(10) if (b == i).sum() >= 20]       # skip near-empty bins
        px, py, n = map(np.array, zip(*rows))
        ax.plot(px, py, lw=2, marker="o", ms=5, color=colors[name], label=name)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Observed home-win rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(color="#e6e5e0", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Calibration on validation seasons", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def sweep_m(d, y, tr, va, ms, p_net):
    """Static-averages baseline for several m on the same games, compared with net ratings and rest."""
    rows = []
    for m in ms:
        X = static_features(d, m)
        model, c = fit_lr(X, y, tr, va, verbose=False)
        p = model.predict_proba(X[va])[:, 1]
        diff, lo, hi = bootstrap_diff(y[va], p, p_net)
        r = {"m": m, "C": c, **{f"val_{k}": v for k, v in scores(y[va], p).items()},
             "diff_vs_net_ratings_rest": diff, "ci_low": lo, "ci_high": hi,
             "significant": hi < 0 or lo > 0}
        rows.append(r)
        print(f"  m={m:<3} C={c:<6} val log-loss {r['val_log_loss']:.4f}  "
              f"vs net ratings and rest {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    return pd.DataFrame(rows).round(4)


def sweep_plot(table, net_ll, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=200)
    ax.axhline(net_ll, ls="--", lw=1, color="#9a9a96")
    ax.text(table.m.max(), net_ll, " net ratings and rest", va="bottom", ha="right",
            fontsize=8, color="#5f5e5a")
    ax.plot(table.m, table.val_log_loss, lw=2, marker="o", ms=5, color="#eb6834")
    best = table.loc[table.val_log_loss.idxmin()]
    ax.annotate(f"best: m = {int(best.m)}", (best.m, best.val_log_loss),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Games averaged (m)")
    ax.set_ylabel("Validation log-loss")
    ax.set_xticks(table.m)
    ax.grid(color="#e6e5e0", lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Static averages: sequence length", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=M_MAX, help="number of past games averaged (<= M_MAX)")
    ap.add_argument("--sweep", type=int, nargs="+",
                    help="compare the static-averages baseline for these values of m and stop")
    args = ap.parse_args()
    for m in [args.m] + (args.sweep or []):
        assert 1 <= m <= M_MAX, f"m must be between 1 and M_MAX={M_MAX}"

    feats, y, split, meta, d = load(args.m)
    tr, va = split == "train", split == "val"
    print(f"train games: {tr.sum()}, validation games: {va.sum()}, m = {args.m}")

    if args.sweep:
        net_model, _ = fit_lr(feats["net"], y, tr, va, verbose=False)
        p_net = net_model.predict_proba(feats["net"][va])[:, 1]
        net_ll = log_loss(y[va], p_net)
        print(f"Net ratings and rest: val log-loss {net_ll:.4f}")
        print("Static averages (difference to net ratings and rest, 95% bootstrap interval):")
        table = sweep_m(d, y, tr, va, sorted(args.sweep), p_net)
        out = ROOT / "reports"
        out.mkdir(exist_ok=True)
        table.to_csv(out / "baseline_m_sweep.csv", index=False)
        sweep_plot(table, net_ll, out / "figures" / "baseline_m_sweep.png")
        best = table.loc[table.val_log_loss.idxmin()]
        print(f"\nBest m = {int(best.m)} (val log-loss {best.val_log_loss:.4f}). "
              f"Saved reports/baseline_m_sweep.csv and figures/baseline_m_sweep.png")
        return

    preds_tr, preds_va, chosen_c = {}, {}, {}

    # 1) home-team reference: training home-win rate for every game
    rate = y[tr].mean()
    preds_tr["Home team (reference)"] = np.full(tr.sum(), rate)
    preds_va["Home team (reference)"] = np.full(va.sum(), rate)
    print(f"Home team: training home-win rate = {rate:.3f}")

    # 2) and 3) logistic regressions
    models = {}
    for name, key in [("Net ratings and rest", "net"), ("Static averages", "static")]:
        print(f"{name} ({feats[key].shape[1]} inputs)")
        model, c = fit_lr(feats[key], y, tr, va)
        models[name], chosen_c[name] = model, c
        preds_tr[name] = model.predict_proba(feats[key][tr])[:, 1]
        preds_va[name] = model.predict_proba(feats[key][va])[:, 1]

    # quantitative results
    rows = []
    for name in preds_va:
        r = {"model": name, "C": chosen_c.get(name, "")}
        r.update({f"train_{k}": v for k, v in scores(y[tr], preds_tr[name]).items()})
        r.update({f"val_{k}": v for k, v in scores(y[va], preds_va[name]).items()})
        rows.append(r)
    results = pd.DataFrame(rows).round(4)

    yv = y[va]
    boot = []
    for a, b in [("Net ratings and rest", "Home team (reference)"),
                 ("Static averages", "Home team (reference)"),
                 ("Static averages", "Net ratings and rest")]:
        d, lo, hi = bootstrap_diff(yv, preds_va[a], preds_va[b])
        boot.append({"comparison": f"{a} - {b}", "log_loss_diff": d, "ci_low": lo, "ci_high": hi,
                     "significant": hi < 0 or lo > 0})
    boot = pd.DataFrame(boot).round(4)

    # qualitative results
    conf = confidence_table(yv, preds_va["Static averages"])
    static_lr = models["Static averages"][-1]
    coef = pd.DataFrame({"feature": MATCH_NAMES + X_NAMES, "coefficient": static_lr.coef_[0]})
    coef = coef.reindex(coef.coefficient.abs().sort_values(ascending=False).index).round(3)

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    results.to_csv(out / "baseline_results.csv", index=False)
    boot.to_csv(out / "baseline_bootstrap.csv", index=False)
    conf.to_csv(out / "baseline_confidence.csv")
    coef.to_csv(out / "baseline_coefficients.csv", index=False)
    pv = meta[va].reset_index(drop=True)[["game_id", "season", "date", "home", "away", "home_win"]]
    for name, p in preds_va.items():
        pv[name] = p.round(4)
    pv.to_csv(out / "baseline_val_predictions.csv", index=False)
    calibration_plot(yv, {k: preds_va[k] for k in ["Net ratings and rest", "Static averages"]},
                     out / "figures" / "calibration_baselines.png")

    pd.set_option("display.width", 200)
    print("\n=== Results (log-loss is the primary metric) ===")
    print(results[["model", "C", "val_log_loss", "val_brier", "val_accuracy",
                   "train_log_loss", "train_accuracy"]].to_string(index=False))
    print("\n=== Bootstrap: difference in validation log-loss (negative = first model better) ===")
    print(boot.to_string(index=False))
    print("\n=== Static averages: accuracy by confidence (validation) ===")
    print(conf.to_string())
    print("\n=== Static averages: largest coefficients (standardized inputs) ===")
    print(coef.head(10).to_string(index=False))
    print(f"\nSaved tables and figure to {out}")


if __name__ == "__main__":
    main()