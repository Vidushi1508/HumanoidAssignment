import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import median_filter

from handover.track import HUMAN, ROBOT, TORSO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WRIST = {HUMAN: 16, ROBOT: 15}
MIN_VIS = 0.5
MAX_GAP_S = 0.5
MIN_HANDOVER_S = 3.0
EDGE_TRIM_S = 0.5
MEET_FRAC = 0.6
ONSET_FRAC = 0.1
CONTACT_FRAC = 0.1
RECEIVER_REST_S = 0.5
EVENTS = ["giver_onset", "receiver_onset", "contact", "release"]


def visible_stretches(pose, fps, width):
    centre = pose[:, :, TORSO, 0].mean(axis=2)
    wrists_seen = (pose[:, HUMAN, WRIST[HUMAN], 3] >= MIN_VIS) & (pose[:, ROBOT, WRIST[ROBOT], 3] >= MIN_VIS)
    both = (centre[:, HUMAN] < width / 2) & (centre[:, ROBOT] > width / 2) & wrists_seen
    edges = np.diff(np.concatenate([[0], both.astype(int), [0]]))
    runs = [[a, b] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < MAX_GAP_S * fps:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    trim = round(EDGE_TRIM_S * fps)
    return [(a + trim, b - trim) for a, b in merged if b - a >= MIN_HANDOVER_S * fps]


def wrist_track(pose, person, start, end):
    xy = pose[start:end, person, WRIST[person], :2].astype(float)
    xy[~(pose[start:end, person, WRIST[person], 3] >= MIN_VIS)] = np.nan
    t = np.arange(len(xy))
    ok = ~np.isnan(xy[:, 0])
    if not ok.any():
        return xy
    xy = np.stack([np.interp(t, t[ok], xy[ok, k]) for k in range(2)], axis=1)
    xy = median_filter(xy, size=(5, 1), mode="nearest")
    kernel = np.ones(5) / 5
    return np.stack([np.convolve(np.pad(xy[:, k], 2, mode="edge"), kernel, "valid") for k in range(2)], axis=1)


def hands_meet(pose, start, end):
    dist = np.linalg.norm(wrist_track(pose, HUMAN, start, end) - wrist_track(pose, ROBOT, start, end), axis=1)
    return dist.min() < MEET_FRAC * dist[0]


def find_events(giver, receiver, fps):
    dist = np.linalg.norm(giver - receiver, axis=1)
    near = dist <= dist.min() + CONTACT_FRAC * (dist[0] - dist.min())
    contact = int(np.argmax(near))
    release = contact + int(np.argmin(near[contact:])) - 1 if not near[contact:].all() else len(dist) - 1

    def onset(xy, rest):
        moved = np.linalg.norm(xy[rest:contact + 1] - xy[rest], axis=1)
        below = np.flatnonzero(moved <= ONSET_FRAC * moved[-1])
        return rest + int(below[-1]) + 1

    giver_onset = onset(giver, 0)
    receiver_onset = onset(receiver, max(0, giver_onset - round(RECEIVER_REST_S * fps)))
    return dict(giver_onset=giver_onset, receiver_onset=receiver_onset, contact=contact, release=release), dist


def segment(pose, fps, width, sheet_rows):
    stretches = [s for s in visible_stretches(pose, fps, width)[1:] if hands_meet(pose, *s)]
    if sheet_rows and len(stretches) != len(sheet_rows):
        found = ", ".join(f"{a / fps:.0f}-{b / fps:.0f}s" for a, b in stretches)
        raise SystemExit(f"found {len(stretches)} handovers, take sheet has {len(sheet_rows)}: {found}")

    rows, signals = [], []
    for card, (start, end) in enumerate(stretches, 1):
        human_gives = card % 2 == 1
        giver, receiver = (HUMAN, ROBOT) if human_gives else (ROBOT, HUMAN)
        g, r = wrist_track(pose, giver, start, end), wrist_track(pose, receiver, start, end)
        events, dist = find_events(g, r, fps)
        row = dict(sheet_rows[card - 1]) if sheet_rows else {}
        row.update(card=card, direction="human_to_robot" if human_gives else "robot_to_human",
                   start=start, end=end, **{k: start + v for k, v in events.items()})
        rows.append(row)
        signals.append((g, r, dist, events))
    return rows, signals


def plot(rows, signals, fps, out):
    n = len(rows)
    fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(16, 2.6 * ((n + 3) // 4)), squeeze=False)
    for ax, row, (g, r, dist, events) in zip(axes.flat, rows, signals):
        t = np.arange(len(dist)) / fps
        ax.plot(t, np.linalg.norm(np.gradient(g, axis=0), axis=1) * fps, label="giver speed")
        ax.plot(t, np.linalg.norm(np.gradient(r, axis=0), axis=1) * fps, label="receiver speed")
        ax.plot(t, dist, "k", label="wrist distance")
        for name, colour in zip(EVENTS, ["C0", "C1", "k", "C3"]):
            ax.axvline(events[name] / fps, color=colour, ls="--", lw=1)
        ax.set_title(f"card {row['card']} {row['direction']}", fontsize=9)
    axes.flat[0].legend(fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.supxlabel("time in handover (s)")
    fig.supylabel("px/s and px")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("out_csv")
    ap.add_argument("--sheet", help="take_sheet.csv; rows for this take are joined by card number")
    ap.add_argument("--plot", help="png with the signals and events of every handover")
    args = ap.parse_args()

    data = np.load(args.track)
    pose, fps = data["pose"], float(data["fps"])
    take = Path(args.track).name.split("_")[0]
    sheet_rows = []
    if args.sheet:
        sheet_rows = [r for r in csv.DictReader(open(args.sheet)) if r["take"] == take]

    rows, signals = segment(pose, fps, int(data["size"][0]), sheet_rows)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if args.plot:
        plot(rows, signals, fps, args.plot)

    giver_first = sum(r["giver_onset"] <= r["receiver_onset"] for r in rows)
    print(f"{take}: {len(rows)} handovers; giver moved first in {giver_first}/{len(rows)}")
    for r in rows:
        print(f"  card {r['card']:2d} {r['direction']:15s} reach {(r['contact'] - r['giver_onset']) / fps:4.2f}s  "
              f"receiver lag {(r['receiver_onset'] - r['giver_onset']) / fps:+5.2f}s  "
              f"hold {(r['release'] - r['contact']) / fps:4.2f}s")


if __name__ == "__main__":
    main()
