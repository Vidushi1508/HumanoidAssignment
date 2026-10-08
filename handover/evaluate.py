import argparse
import csv
from pathlib import Path

import numpy as np

from handover import models
from handover.dataset import SPLITS, load, split

FRACTIONS = [0.25, 0.5, 0.75]
HORIZON_S = 0.5
REST_FRAMES = 10


def observed_until(h, fraction):
    return h["onset"] + round(fraction * (h["contact"] - h["onset"]))


def train_priors(train):
    point = np.mean([h["giver"][h["contact"]] for h in train], axis=0)
    remaining = {f: np.mean([(h["contact"] - observed_until(h, f)) / h["fps"] for h in train]) for f in FRACTIONS}
    return point, remaining


def point_predictions(h, f, prior_point, prior_remaining, learned):
    t = observed_until(h, f)
    hist, fps = h["giver_raw"][:t + 1], h["fps"]
    steps = max(1, round(prior_remaining[f] * fps))
    out = {
        "stay": (hist[-1], np.nan),
        "mean": (prior_point, prior_remaining[f]),
        "const_velocity": (models.const_velocity(hist, fps, steps)[-1], prior_remaining[f]),
        "kalman": (models.kalman(hist, fps, steps)[-1], prior_remaining[f]),
        "min_jerk": models.min_jerk_point(hist, fps, min(REST_FRAMES, h["onset"])),
    }
    for name, predict in learned.items():
        _, point, ttc = predict(h, t)
        out[name] = (point, ttc)
    return out, h["giver"][h["contact"]], (h["contact"] - t) / fps


def trajectory_errors(h, horizon, learned):
    fps = h["fps"]
    errs = {m: [] for m in ["stay", "const_velocity", "kalman", "min_jerk", *learned]}
    for t in range(h["onset"], h["contact"] + 1):
        if t + horizon >= len(h["giver"]):
            break
        hist, truth = h["giver_raw"][:t + 1], h["giver"][t + 1:t + 1 + horizon]
        preds = {
            "stay": models.stay(hist, fps, horizon),
            "const_velocity": models.const_velocity(hist, fps, horizon),
            "kalman": models.kalman(hist, fps, horizon),
            "min_jerk": models.min_jerk(hist, fps, horizon, min(REST_FRAMES, h["onset"])),
        }
        for name, predict in learned.items():
            preds[name] = predict(h, t)[0]
        for m, p in preds.items():
            errs[m].append(np.linalg.norm(p - truth, axis=1).mean())
    return errs


def evaluate(handovers, split_name, learned={}):
    train, test = split(handovers, split_name)
    prior_point, prior_remaining = train_priors(train)
    point_rows, traj_rows = [], []
    for h in test:
        for f in FRACTIONS:
            preds, true_point, true_ttc = point_predictions(h, f, prior_point, prior_remaining, learned)
            for method, (point, ttc) in preds.items():
                point_rows.append(dict(split=split_name, take=h["take"], card=h["card"], fraction=f, method=method,
                                       point_err_cm=100 * np.linalg.norm(point - true_point),
                                       ttc_err_s=abs(ttc - true_ttc)))
        horizon = round(HORIZON_S * h["fps"])
        for method, errs in trajectory_errors(h, horizon, learned).items():
            traj_rows.append(dict(split=split_name, take=h["take"], card=h["card"], method=method,
                                  ade_cm=100 * np.mean(errs)))
    return point_rows, traj_rows


def summarise(point_rows, traj_rows):
    lines = []
    methods = list(dict.fromkeys(r["method"] for r in point_rows))
    lines.append(f"{'handover point error (cm), mean':34s}" + "".join(f"{f'{f:.0%} seen':>12s}" for f in FRACTIONS))
    for m in methods:
        vals = [np.mean([r["point_err_cm"] for r in point_rows if r["method"] == m and r["fraction"] == f]) for f in FRACTIONS]
        lines.append(f"  {m:32s}" + "".join(f"{v:12.1f}" for v in vals))
    lines.append(f"{'time-to-contact error (s), mean':34s}" + "".join(f"{f'{f:.0%} seen':>12s}" for f in FRACTIONS))
    for m in methods:
        if all(np.isnan(r["ttc_err_s"]) for r in point_rows if r["method"] == m):
            continue
        vals = [np.mean([r["ttc_err_s"] for r in point_rows if r["method"] == m and r["fraction"] == f]) for f in FRACTIONS]
        lines.append(f"  {m:32s}" + "".join(f"{v:12.2f}" for v in vals))
    lines.append(f"next {HORIZON_S} s of the giver's hand, mean error (cm) during the reach")
    for m in dict.fromkeys(r["method"] for r in traj_rows):
        lines.append(f"  {m:32s}{np.mean([r['ade_cm'] for r in traj_rows if r['method'] == m]):12.1f}")
    return "\n".join(lines)


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/mnt/d/Humanoid/work")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    handovers = load(args.work, args.sheet)
    Path(args.out).mkdir(exist_ok=True)
    all_points, all_traj = [], []
    for name in SPLITS:
        point_rows, traj_rows = evaluate(handovers, name)
        print(f"== split: {name} (test {', '.join(SPLITS[name][1])})")
        print(summarise(point_rows, traj_rows))
        all_points += point_rows
        all_traj += traj_rows
    write_csv(all_points, Path(args.out) / "baselines_point.csv")
    write_csv(all_traj, Path(args.out) / "baselines_trajectory.csv")


if __name__ == "__main__":
    main()
