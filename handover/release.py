import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from handover import evaluate as ev
from handover.dataset import SPLITS, load, split
from handover.models import DEVICE, policy_features

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 300
LR = 3e-3
WEIGHT_DECAY = 1e-3
HIDDEN = 32
EARLY_S = 0.1


class HoldModel(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(8, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, last):
        out, _ = self.gru(x)
        return nn.functional.softplus(self.head(out[torch.arange(len(last)), last])[:, 0])


def approach(h):
    return policy_features(h["giver_raw"], h["receiver_raw"], h["fps"])[:h["contact"] + 1]


def hold_time(h):
    return (h["release"] - h["contact"]) / h["fps"]


def batch(handovers):
    T = max(h["contact"] + 1 for h in handovers)
    X = np.zeros((len(handovers), T, 8), np.float32)
    for i, h in enumerate(handovers):
        X[i, :h["contact"] + 1] = approach(h)
    last = torch.tensor([h["contact"] for h in handovers])
    return torch.from_numpy(X), last


def train(train_set, seed):
    torch.manual_seed(seed)
    model = HoldModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    X, last = (a.to(DEVICE) for a in batch(train_set))
    y = torch.tensor([hold_time(h) for h in train_set], dtype=torch.float32, device=DEVICE)
    for _ in range(EPOCHS):
        loss = ((model(X, last) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.cpu().eval()


def run_split(handovers, name):
    train_set, test_set = split(handovers, name)
    true = np.array([hold_time(h) for h in test_set])
    preds = {"release at contact": np.zeros_like(true),
             "mean training hold": np.full_like(true, np.mean([hold_time(h) for h in train_set]))}
    X, last = batch(test_set)
    for seed in SEEDS:
        model = train(train_set, seed)
        with torch.no_grad():
            preds[f"gru_seed{seed}"] = model(X, last).numpy()
    rows = []
    for method, p in preds.items():
        for h, t, q in zip(test_set, true, p):
            seed = int(method[8:]) if method.startswith("gru_seed") else ""
            rows.append(dict(split=name, take=h["take"], card=h["card"], method="gru" if seed != "" else method,
                             seed=seed, true_hold_s=t, predicted_hold_s=q, error_s=q - t))
    return rows, true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.set_num_threads(4)
    handovers = load(args.data, args.sheet, direction="robot_to_human")
    all_rows = []
    for name in SPLITS:
        rows, true = run_split(handovers, name)
        print(f"== split: {name} (test {', '.join(SPLITS[name][1])}); robot-role giver, {len(true)} test handovers; "
              f"true hold {true.mean():.2f} s (range {true.min():.2f}-{true.max():.2f})")
        print(f"{'method':22s}{'mean abs error s':>18s}{'early by >0.1 s':>17s}{'late by >0.1 s':>16s}")
        for method in dict.fromkeys(r["method"] for r in rows):
            err = np.array([r["error_s"] for r in rows if r["method"] == method])
            print(f"{method:22s}{np.abs(err).mean():18.2f}{np.mean(err < -EARLY_S):17.0%}{np.mean(err > EARLY_S):16.0%}")
        all_rows += rows
    Path(args.out).mkdir(exist_ok=True)
    ev.write_csv(all_rows, Path(args.out) / "release.csv")


if __name__ == "__main__":
    main()
