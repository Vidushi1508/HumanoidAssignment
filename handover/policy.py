import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

from handover import evaluate as ev
from handover import train as world
from handover.dataset import SPLITS, load, split
from handover.models import Policy, policy_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [0, 1, 2, 3, 4]
EPOCHS = 400
LR = 3e-3
WEIGHT_DECAY = 1e-4
HIDDEN = 64
MAX_SPEED = 1.5
MOVE_START = 0.05
GRASP_DIST = 0.05
ARRIVE_DIST = 0.05
GAINS = [2, 4, 8, 12, 20, 30]
NOISE_STEP = 0.003
CORRECTION_GAIN = 5.0


def grasp_offset(train_set):
    return np.mean([h["receiver"][h["contact"]] - h["giver"][h["contact"]] for h in train_set], axis=0)


def train_policy(train_set, seed, noise=False):
    torch.manual_seed(seed)
    T = max(len(h["giver"]) for h in train_set)
    X = np.zeros((len(train_set), T, 8), np.float32)
    V = np.zeros((len(train_set), T, 2), np.float32)
    G = np.zeros((len(train_set), T), np.float32)
    MV = np.zeros((len(train_set), T), bool)
    MG = np.zeros((len(train_set), T), bool)
    for i, h in enumerate(train_set):
        n = len(h["giver"])
        X[i, :n] = policy_features(h["giver_raw"], h["receiver_raw"], h["fps"])
        V[i, :n - 1] = np.diff(h["receiver"], axis=0) * h["fps"]
        G[i, h["contact"]:n] = 1
        MV[i, :h["contact"]] = True
        MG[i, :n] = True
    X, V, G, MV, MG = (torch.from_numpy(a) for a in (X, V, G, MV, MG))

    model = Policy(HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(EPOCHS):
        x, v = X, V
        if noise:
            drift = torch.cumsum(torch.randn(V.shape) * NOISE_STEP, dim=1)
            x = X.clone()
            x[..., 4:6] += drift
            x[..., 6:8] += torch.diff(drift, dim=1, prepend=torch.zeros_like(drift[:, :1])) * train_set[0]["fps"]
            v = V - CORRECTION_GAIN * drift
        vel, logit, _ = model(x)
        loss = (((vel - v) ** 2).sum(-1)[MV].mean()
                + torch.nn.functional.binary_cross_entropy_with_logits(logit[MG], G[MG]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def limit(v):
    speed = np.linalg.norm(v)
    return v * min(1.0, MAX_SPEED / speed) if speed > 0 else v


class Chase:
    def __init__(self, target, offset, gain):
        self.target, self.offset, self.gain = target, offset, gain

    def reset(self, h):
        self.moving = False

    def step(self, h, t, pos):
        giver = h["giver_raw"]
        self.moving = self.moving or np.linalg.norm(giver[t] - giver[0]) > MOVE_START
        grasp = np.linalg.norm(pos - (giver[t] + self.offset)) < GRASP_DIST
        if not self.moving:
            return np.zeros(2), grasp
        return limit(self.gain * (self.target(h, t) + self.offset - pos)), grasp


class Learned:
    def __init__(self, model):
        self.model = model

    def reset(self, h):
        self.state, self.prev = None, None

    def step(self, h, t, pos):
        fps, giver = h["fps"], h["giver_raw"]
        prev = pos if self.prev is None else self.prev
        x = np.r_[giver[t], (giver[t] - giver[max(t - 1, 0)]) * fps, pos, (pos - prev) * fps].astype(np.float32)
        with torch.no_grad():
            vel, logit, self.state = self.model(torch.from_numpy(x)[None, None], self.state)
        self.prev = pos.copy()
        return limit(vel[0, 0].numpy()), float(torch.sigmoid(logit[0, 0])) > 0.5


def rollout(controller, h):
    controller.reset(h)
    pos = h["receiver_raw"][0].copy()
    path, grasp = [], []
    for t in range(len(h["giver_raw"])):
        vel, g = controller.step(h, t, pos)
        path.append(pos.copy())
        grasp.append(g)
        pos = pos + vel / h["fps"]
    return np.array(path), np.array(grasp)


def metrics(h, path, grasp):
    fps, c, o, true = h["fps"], h["contact"], h["onset"], h["receiver"]
    near = np.flatnonzero(np.linalg.norm(path - true[c], axis=1) < ARRIVE_DIST)
    grasps = np.flatnonzero(grasp)
    return dict(
        contact_err_cm=100 * np.linalg.norm(path[c] - true[c]),
        path_err_cm=100 * np.linalg.norm(path[o:c + 1] - true[o:c + 1], axis=1).mean(),
        arrival_s=(near[0] - c) / fps if len(near) else np.nan,
        grasp_err_s=(grasps[0] - c) / fps if len(grasps) else np.nan,
    )


def reactive(h, t):
    return h["giver_raw"][t]


def best_gain(train_set, target, offset):
    err = {k: np.mean([metrics(h, *rollout(Chase(target, offset, k), h))["contact_err_cm"] for h in train_set])
           for k in GAINS}
    return min(err, key=err.get)


def build_controllers(train_set, seed, horizon):
    offset = grasp_offset(train_set)
    predict = world.predictor(world.train(train_set, seed, horizon))

    def predicted_point(h, t):
        return predict(h, t)[1]

    return {
        "reactive": Chase(reactive, offset, best_gain(train_set, reactive, offset)),
        "world_model_target": Chase(predicted_point, offset, best_gain(train_set, predicted_point, offset)),
        "learned_policy": Learned(train_policy(train_set, seed)),
        "learned_policy_noise": Learned(train_policy(train_set, seed, noise=True)),
    }


def run_split(handovers, name, horizon):
    train_set, test_set = split(handovers, name)
    rows, paths = [], {}

    def add(method, seed, h, path, grasp):
        rows.append(dict(split=name, take=h["take"], card=h["card"], method=method, seed=seed, **metrics(h, path, grasp)))
        if seed in ("", 0):
            paths[(h["take"], h["card"], method)] = path

    for h in test_set:
        add("human", "", h, h["receiver"], np.arange(len(h["giver"])) >= h["contact"])
    for seed in SEEDS:
        controllers = build_controllers(train_set, seed, horizon)
        if seed == SEEDS[0]:
            print(f"  chosen gains (training takes): reactive {controllers['reactive'].gain}, "
                  f"world model target {controllers['world_model_target'].gain} (seed {seed})")
        for name, controller in controllers.items():
            if name == "reactive" and seed != SEEDS[0]:
                continue
            for h in test_set:
                add(name, "" if name == "reactive" else seed, h, *rollout(controller, h))
    return rows, paths


def summarise(rows):
    keys = ["contact_err_cm", "path_err_cm", "arrival_s", "grasp_err_s"]
    lines = [f"{'method':26s}{'contact err cm':>16s}{'path err cm':>13s}{'arrival s':>11s}{'arrived':>9s}"
             f"{'grasp err s':>13s}{'grasped':>9s}"]
    for m in dict.fromkeys(r["method"] for r in rows):
        sel = [r for r in rows if r["method"] == m]
        v = {k: np.array([r[k] for r in sel], float) for k in keys}
        lines.append(f"{m:26s}{v['contact_err_cm'].mean():16.1f}{v['path_err_cm'].mean():13.1f}"
                     f"{np.nanmean(v['arrival_s']):+11.2f}{np.mean(~np.isnan(v['arrival_s'])):9.0%}"
                     f"{np.nanmean(np.abs(v['grasp_err_s'])):13.2f}{np.mean(~np.isnan(v['grasp_err_s'])):9.0%}")
    return "\n".join(lines)


def plot(test_set, paths, out):
    examples = [next(h for h in test_set if h["hand_height"] == height and h["take"] == take)
                for take, height in [("A3", "low"), ("A3", "high"), ("B3", "mid"), ("B3", "high")]]
    styles = [("human", "k", "human receiver"), ("reactive", "C0", "reactive"),
              ("world_model_target", "C2", "world model + controller"), ("learned_policy", "C3", "learned policy"),
              ("learned_policy_noise", "C1", "learned policy, noise-trained")]
    fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 7))
    for col, h in enumerate(examples):
        t = (np.arange(len(h["giver"])) - h["contact"]) / h["fps"]
        top, bottom = axes[0, col], axes[1, col]
        top.plot(*h["giver"].T, color="0.6", lw=3, label="giver hand")
        bottom.plot(t, h["giver"][:, 0], color="0.6", lw=3)
        for key, colour, label in styles:
            path = paths[(h["take"], h["card"], key)]
            top.plot(*path.T, color=colour, label=label)
            top.plot(*path[h["contact"]], "o", color=colour)
            bottom.plot(t, path[:, 0], color=colour)
        top.plot(0, 0, "k^", ms=8)
        top.set_aspect("equal")
        top.set_title(f"{h['take']} card {h['card']} ({h['human_chair']}, {h['hand_height']})", fontsize=10)
        bottom.axvline(0, color="k", ls=":")
        bottom.set_xlabel("time from contact (s)")
        top.grid()
        bottom.grid()
    axes[0, 0].set_ylabel("y up (m)")
    axes[1, 0].set_ylabel("x toward human (m)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("receiving: paths in the robot-shoulder frame (top, dots at contact) and forward position over time (bottom)")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/mnt/d/Humanoid/work")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot", help="png with example rollouts from the main split (seed 0)")
    args = ap.parse_args()

    torch.set_num_threads(4)
    handovers = load(args.work, args.sheet)
    horizon = round(ev.HORIZON_S * handovers[0]["fps"])
    Path(args.out).mkdir(exist_ok=True)
    all_rows = []
    for name in SPLITS:
        rows, paths = run_split(handovers, name, horizon)
        if name == "main" and args.plot:
            plot(split(handovers, name)[1], paths, args.plot)
        print(f"== split: {name} (test {', '.join(SPLITS[name][1])}); learned methods = mean over {len(SEEDS)} seeds")
        print(summarise(rows))
        all_rows += rows
    ev.write_csv(all_rows, Path(args.out) / "policy.csv")


if __name__ == "__main__":
    main()
