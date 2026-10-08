import csv
import json
from pathlib import Path

import numpy as np

from handover.calibrate import robot_origin, to_metres
from handover.segment import MIN_VIS, WRIST, wrist_track
from handover.track import HUMAN, ROBOT

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


def load(work_dir, sheet_path, direction="human_to_robot"):
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
            end = release + 1
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
