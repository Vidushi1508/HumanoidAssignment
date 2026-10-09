import argparse
from pathlib import Path

import numpy as np
import torch

from handover import evaluate as ev
from handover.dataset import SPLITS, load, split
from handover.models import DEVICE, WorldModel, features

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 400
LR = 3e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 64


def targets(h, horizon):
    n, raw, smooth = len(h["giver"]), h["giver_raw"], h["giver"]
    future = np.stack([smooth[np.minimum(np.arange(t + 1, t + 1 + horizon), n - 1)] - raw[t] for t in range(n)])
    point = smooth[h["contact"]] - raw
    ttc = np.maximum(h["contact"] - np.arange(n), 0) / h["fps"]
    mask = np.zeros(n, bool)
    mask[h["onset"]:h["contact"] + 1] = True
    return future, point, ttc, mask


def batch(handovers, horizon):
    T = max(len(h["giver"]) for h in handovers)
    X = np.zeros((len(handovers), T, 4), np.float32)
    Y_traj = np.zeros((len(handovers), T, horizon, 2), np.float32)
    Y_point = np.zeros((len(handovers), T, 2), np.float32)
    Y_ttc = np.zeros((len(handovers), T), np.float32)
    M = np.zeros((len(handovers), T), bool)
    for i, h in enumerate(handovers):
        n = len(h["giver"])
        X[i, :n] = features(h, n - 1)
        Y_traj[i, :n], Y_point[i, :n], Y_ttc[i, :n], M[i, :n] = targets(h, horizon)
    return [torch.from_numpy(a) for a in (X, Y_traj, Y_point, Y_ttc, M)]


def train(handovers, seed, horizon):
    torch.manual_seed(seed)
    model = WorldModel(HIDDEN, horizon).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    X, Y_traj, Y_point, Y_ttc, M = (a.to(DEVICE) for a in batch(handovers, horizon))
    for _ in range(EPOCHS):
        traj, point, ttc = model(X)
        loss = (((traj - Y_traj) ** 2).sum(-1).mean(-1)[M].mean()
                + ((point - Y_point) ** 2).sum(-1)[M].mean()
                + ((ttc - Y_ttc) ** 2)[M].mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.cpu().eval()


def predictor(model):
    def predict(h, t):
        with torch.no_grad():
            traj, point, ttc = model(torch.from_numpy(features(h, t))[None])
        here = h["giver_raw"][t]
        return here + traj[0, -1].numpy(), here + point[0, -1].numpy(), float(ttc[0, -1])
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.set_num_threads(4)
    handovers = load(args.data, args.sheet)
    horizon = round(ev.HORIZON_S * handovers[0]["fps"])
    Path(args.out).mkdir(exist_ok=True)
    all_points, all_traj = [], []
    for name in SPLITS:
        train_set, _ = split(handovers, name)
        learned = {f"gru_seed{s}": predictor(train(train_set, s, horizon)) for s in SEEDS}
        point_rows, traj_rows = ev.evaluate(handovers, name, learned)
        for r in point_rows + traj_rows:
            if r["method"].startswith("gru_seed"):
                r["seed"], r["method"] = int(r["method"][8:]), "gru"
        print(f"== split: {name} (test {', '.join(SPLITS[name][1])}); gru = mean over {len(SEEDS)} seeds")
        print(ev.summarise(point_rows, traj_rows))
        seed_means = [np.mean([r["point_err_cm"] for r in point_rows if r.get("seed") == s and r["fraction"] == 0.5])
                      for s in SEEDS]
        print(f"  gru handover point error at 50% seen, per seed: {' '.join(f'{v:.1f}' for v in seed_means)} cm")
        all_points += point_rows
        all_traj += traj_rows
    ev.write_csv(all_points, Path(args.out) / "world_model_point.csv")
    ev.write_csv(all_traj, Path(args.out) / "world_model_trajectory.csv")


if __name__ == "__main__":
    main()
