import argparse
import json

import cv2
import numpy as np

from handover.segment import WRIST, visible_stretches
from handover.track import HUMAN, ROBOT

BAND_SPACING_M = 1.0
MAX_BAND_GAP_PX = 40
MAX_END_GAP_PX = 60
LEVEL_DEG = 15
HANGING_DEG = 75
STEP = 3
SHOULDER = {HUMAN: 12, ROBOT: 11}
ORIGIN_WINDOW_S = 0.5


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (((h < 8) | (h > 172)) & (s > 100) & (v > 80)).astype(np.uint8)


def stick_line(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < 2000:
        return None
    pts = np.column_stack(np.nonzero(labels == biggest)[::-1]).astype(np.float32)
    dx, dy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([dx, dy])


def band_centres(mask, point, direction):
    ys, xs = np.nonzero(mask)
    rel = np.column_stack([xs, ys]) - point
    along = rel @ direction
    across = rel @ np.array([-direction[1], direction[0]])
    along = np.round(along[np.abs(across) < 15]).astype(int)
    lo = along.min()
    red = np.zeros(along.max() - lo + 1, bool)
    red[along - lo] = True

    edges = np.diff(np.concatenate([[0], red.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    main = int(np.argmax(ends - starts))
    if main == 0 or main == len(starts) - 1:
        return None
    left_gap = starts[main] - ends[main - 1]
    right_gap = starts[main + 1] - ends[main]
    if not (0 < left_gap <= MAX_BAND_GAP_PX and 0 < right_gap <= MAX_BAND_GAP_PX):
        return None
    if ends[main - 1] - starts[main - 1] > MAX_END_GAP_PX or ends[main + 1] - starts[main + 1] > MAX_END_GAP_PX:
        return None
    left = (ends[main - 1] + starts[main]) / 2 + lo
    right = (ends[main] + starts[main + 1]) / 2 + lo
    return point + left * direction, point + right * direction


def calibrate(video_path, pose, fps, width):
    start, end = visible_stretches(pose, fps, width)[0]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    spacings, downs = [], []
    for i in range(start, end):
        ok, bgr = cap.read()
        if not ok:
            break
        if (i - start) % STEP:
            continue
        mask = red_mask(bgr)
        line = stick_line(mask)
        if line is None:
            continue
        point, direction = line
        angle = np.degrees(np.arctan2(abs(direction[1]), abs(direction[0])))
        if angle < LEVEL_DEG:
            bands = band_centres(mask, point, direction)
            if bands is not None:
                spacings.append(np.linalg.norm(bands[1] - bands[0]))
        elif angle > HANGING_DEG:
            downs.append(direction if direction[1] > 0 else -direction)
    cap.release()

    down = np.median(downs, axis=0)
    down /= np.linalg.norm(down)
    return dict(
        px_per_m=float(np.median(spacings) / BAND_SPACING_M),
        px_per_m_iqr=[float(np.percentile(spacings, 25)), float(np.percentile(spacings, 75))],
        level_frames=len(spacings),
        down=[float(down[0]), float(down[1])],
        hanging_frames=len(downs),
        tilt_deg=float(np.degrees(np.arctan2(down[0], down[1]))),
    )


def robot_origin(pose, giver_onset, fps):
    window = pose[max(0, giver_onset - round(ORIGIN_WINDOW_S * fps)):giver_onset, ROBOT, SHOULDER[ROBOT], :2]
    return np.nanmedian(window, axis=0)


def to_metres(xy, origin, calib):
    up = -np.asarray(calib["down"])
    toward_human = np.array([up[1], -up[0]])
    rel = (np.asarray(xy) - origin) / calib["px_per_m"]
    return np.stack([rel @ toward_human, rel @ up], axis=-1)


def arm_length(pose, person, calib):
    vec = pose[:, person, WRIST[person], :2] - pose[:, person, SHOULDER[person], :2]
    vis = np.minimum(pose[:, person, WRIST[person], 3], pose[:, person, SHOULDER[person], 3])
    return float(np.nanpercentile(np.linalg.norm(vec[vis > 0.5], axis=1), 99) / calib["px_per_m"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("track")
    ap.add_argument("out_json")
    args = ap.parse_args()

    data = np.load(args.track)
    pose, fps = data["pose"], float(data["fps"])
    calib = calibrate(args.video, pose, fps, int(data["size"][0]))
    with open(args.out_json, "w") as f:
        json.dump(calib, f, indent=2)

    lo, hi = calib["px_per_m_iqr"]
    print(f"scale {calib['px_per_m']:.1f} px/m (IQR {lo:.1f}-{hi:.1f}, {calib['level_frames']} level frames); "
          f"tilt {calib['tilt_deg']:+.1f} deg ({calib['hanging_frames']} hanging frames)")
    print(f"longest shoulder-wrist distance (99th pct): human {arm_length(pose, HUMAN, calib):.2f} m, "
          f"robot role {arm_length(pose, ROBOT, calib):.2f} m")


if __name__ == "__main__":
    main()
