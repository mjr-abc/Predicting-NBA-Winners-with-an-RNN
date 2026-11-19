"""Neural models (3) and (4) for the progress report (later optimized for final submission)

  (3) FormLSTM: shared LSTM -> one form score per team
  (4) MatchupLSTM: shared LSTM -> offense state o and defense state d -> bilinear matchup

Both models also get x (net rating difference, home rest, away rest) through one linear layer.
Trained on the training seasons, evaluated on the validation seasons.

Usage:  
(3) FormLSTM: python src/train_lstm.py form
(4) MatchupLSTM: python src/train_lstm.py matchup
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines import bootstrap_diff, scores
from config import PROCESSED, ROOT, M_MAX, TEAM_STEP

HIDDEN_SIZE = 16
STATE_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 30


class FormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FormLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_form = nn.Linear(hidden_size, 1)
        self.fc_x = nn.Linear(3, 1)

    def encode(self, seq):
        h0 = torch.zeros(1, seq.size(0), self.hidden_size)
        c0 = torch.zeros(1, seq.size(0), self.hidden_size)
        out, __ = self.rnn(seq, (h0, c0))
        return out[:, -1, :]

    def forward(self, home_seq, away_seq, x):
        f_home = self.fc_form(self.encode(home_seq))
        f_away = self.fc_form(self.encode(away_seq))
        return (f_home - f_away + self.fc_x(x)).squeeze(1)


class MatchupLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, state_size):
        super(MatchupLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_off = nn.Linear(hidden_size, state_size)
        self.fc_def = nn.Linear(hidden_size, state_size)
        self.bilinear = nn.Bilinear(state_size, state_size, 1)
        self.fc_x = nn.Linear(3, 1)

    def encode(self, seq):
        h0 = torch.zeros(1, seq.size(0), self.hidden_size)
        c0 = torch.zeros(1, seq.size(0), self.hidden_size)
        out, __ = self.rnn(seq, (h0, c0))
        h = out[:, -1, :]
        return self.fc_off(h), self.fc_def(h)

    def forward(self, home_seq, away_seq, x):
        o_home, d_home = self.encode(home_seq)
        o_away, d_away = self.encode(away_seq)
        matchup = self.bilinear(o_home, d_away) - self.bilinear(o_away, d_home)
        return (matchup + self.fc_x(x)).squeeze(1)


def get_loader(d, name, shuffle):
    idx = d["split"] == name
    ds = TensorDataset(torch.tensor(d["home_seq"][idx]), torch.tensor(d["away_seq"][idx]),
                       torch.tensor(d["x"][idx]), torch.tensor(d["y"][idx]))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def get_loss(model, loader, criterion):
    model.eval()
    total, n, probs = 0.0, 0, []
    with torch.no_grad():
        for home_seq, away_seq, x, y in loader:
            pred = model(home_seq, away_seq, x)
            total += criterion(pred, y).item() * len(y)
            n += len(y)
            probs.append(torch.sigmoid(pred))
    return total / n, torch.cat(probs).numpy()


def train(model, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_losses, val_losses = [], []
    best_loss, best_epoch, best_state = float("inf"), 0, None

    for epoch in range(NUM_EPOCHS):
        model.train()
        for home_seq, away_seq, x, y in train_loader:
            optimizer.zero_grad()
            pred = model(home_seq, away_seq, x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        train_loss, __ = get_loss(model, train_loader, criterion)
        val_loss, __ = get_loss(model, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        # weights of the best epoch
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return train_losses, val_losses, best_epoch


def plot_curves(train_losses, val_losses, best_epoch, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(4.2, 3.2), dpi=200)
    plt.plot(epochs, train_losses, label="training")
    plt.plot(epochs, val_losses, label="validation")
    plt.axvline(best_epoch, ls="--", color="gray", lw=1)
    plt.xlabel("Epoch")
    plt.ylabel("Log-loss")
    plt.title(title, fontsize=10)
    plt.legend(frameon=False)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "matchup"
    torch.manual_seed(0)

    d = np.load(PROCESSED / f"sequences_m{M_MAX}.npz")
    train_loader = get_loader(d, "train", shuffle=True)
    val_loader = get_loader(d, "val", shuffle=False)

    if model_name == "form":
        model = FormLSTM(len(TEAM_STEP), HIDDEN_SIZE)
        title = "(3) Shared LSTM"
    else:
        model = MatchupLSTM(len(TEAM_STEP), HIDDEN_SIZE, STATE_SIZE)
        title = "(4) Shared LSTM + bilinear"
    print(f"{title}: {sum(p.numel() for p in model.parameters())} parameters")

    train_losses, val_losses, best_epoch = train(model, train_loader, val_loader)

    # Validation results of the best epoch
    __, p_val = get_loss(model, val_loader, nn.BCEWithLogitsLoss())
    y_val = d["y"][d["split"] == "val"]
    res = scores(y_val, p_val)
    print(f"\n=== {title}, best epoch {best_epoch} ===")
    print(f"log-loss {res['log_loss']:.4f}, Brier {res['brier']:.4f}, accuracy {res['accuracy']:.3f}")

    # Compare with the baselines on the same validation games
    out = ROOT / "reports"
    base = pd.read_csv(out / "baseline_val_predictions.csv")
    for ref in ["Net ratings and rest", "Static averages"]:
        diff, lo, hi = bootstrap_diff(y_val, p_val, base[ref].to_numpy())
        sig = "significant" if (hi < 0 or lo > 0) else "not significant"
        print(f"vs {ref}: {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] ({sig})")

    plot_curves(train_losses, val_losses, best_epoch, title,
                out / "figures" / f"learning_curves_{model_name}.png")


if __name__ == "__main__":
    main()