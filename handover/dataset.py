import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from handover.calibrate import robot_origin, to_metres
from handover.segment import MIN_VIS, WRIST, wrist_track
from handover.track import HUMAN, ROBOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAKES = ["A1_MUG", "A2_BOTTLE", "A3_BOX", "B1_MUG", "B2_BOTTLE", "B3_BOX"]
SPLITS = {
    "main": (["A1", "A2", "B1", "B2"], ["A3", "B3"]),
    "cross_person": (["A1", "A2", "A3"], ["B1", "B2", "B3"]),
}


def causal_track(pose, person, start, end):
    xy = pose[start:end, person, WRIST[person], :2].astype(float)
    ok = pose[start:end, person, WRIST[person], 3] >= MIN_VIS
    last = np.maximum.accumulate(np.where(ok, np.arange(len(xy)), -1))
    last[last < 0] = np.argmax(ok)
    return xy[last]


def hesitated(row):
    planned = row["hesitation"] == "yes" and "not performed" not in row["notes"]
    return planned or "hesitation performed" in row["notes"] or "extra hesitation" in row["notes"]


def load(work_dir, sheet_path, direction="human_to_robot", tail_s=0.0):
    work_dir = Path(work_dir)
    sheet = {(r["take"], r["card"]): r for r in csv.DictReader(open(sheet_path))}
    handovers = []
    for name in TAKES:
        take = name.split("_")[0]
        data = np.load(work_dir / f"{name}_pose.npz")
        pose, fps = data["pose"], float(data["fps"])
        calib = json.load(open(work_dir / "calib" / f"{name}.json"))
        for ev in csv.DictReader(open(work_dir / "segments" / f"{name}.csv")):
            if ev["direction"] != direction:
                continue
            onset, contact, release = int(ev["giver_onset"]), int(ev["contact"]), int(ev["release"])
            start = int(ev["start"])
            end = min(int(ev["end"]), release + 1 + round(tail_s * fps))
            giver, receiver = (HUMAN, ROBOT) if direction == "human_to_robot" else (ROBOT, HUMAN)
            origin = robot_origin(pose, onset, fps)

            def metres(xy):
                return to_metres(xy, origin, calib)

            row = sheet[(take, ev["card"])]
            handovers.append(dict(
                take=take, card=int(ev["card"]), direction=direction, human=row["human"], robot=row["robot"],
                object=row["object"], human_chair=row["human_chair"], hand_height=row["hand_height"],
                speed=row["speed"], hesitation=hesitated(row), fps=fps,
                giver_raw=metres(causal_track(pose, giver, start, end)),
                giver=metres(wrist_track(pose, giver, start, end)),
                receiver_raw=metres(causal_track(pose, receiver, start, end)),
                receiver=metres(wrist_track(pose, receiver, start, end)),
                onset=onset - start, receiver_onset=int(ev["receiver_onset"]) - start,
                contact=contact - start, release=release - start,
            ))
    return handovers


def split(handovers, name):
    train, test = SPLITS[name]
    return [h for h in handovers if h["take"] in train], [h for h in handovers if h["take"] in test]


def plot(data_dir, sheet_path, out):
    colours = {"low": "C0", "mid": "C1", "high": "C3"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for col, direction in enumerate(["human_to_robot", "robot_to_human"]):
        for h in load(data_dir, sheet_path, direction):
            ax = axes[0 if h["take"].startswith("A") else 1, col]
            reach = h["giver"][h["onset"]:h["contact"] + 1]
            ax.plot(*reach.T, color=colours[h["hand_height"]], lw=1)
            ax.plot(*reach[-1], "ko", ms=3)
    titles = ["session A: human p1 gives", "session A: robot role p2 gives",
              "session B: human p2 gives", "session B: robot role p1 gives"]
    for ax, title in zip(axes.flat, titles):
        ax.plot(0, 0, "k^", ms=10)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.grid()
    for ax in axes[1]:
        ax.set_xlabel("x toward human (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y up (m)")
    for height, colour in colours.items():
        axes[0, 0].plot([], [], color=colour, label=f"{height} hand height")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("all 120 reaches, giver's wrist from reach onset to contact (dot); origin = robot-role shoulder")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--plot", default="results/reaches.png")
    args = ap.parse_args()
    plot(args.data, args.sheet, args.plot)


if __name__ == "__main__":
    main()
