import argparse

import cv2
import numpy as np

COLOURS = [(0, 140, 255), (255, 120, 0)]
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (15, 19), (16, 20), (0, 11), (0, 12)]
SCALE = 0.5


def draw(img, people):
    for person, colour in zip(people, COLOURS):
        if np.isnan(person[0, 0]):
            continue
        p = (person[:, :2] * SCALE).astype(int)
        for a, b in BONES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), colour, 2)
        for j, label in [(15, "L"), (16, "R")]:
            cv2.circle(img, tuple(p[j]), 6, colour, -1)
            cv2.putText(img, label, tuple(p[j] + [8, -8]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("track")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds")
    args = ap.parse_args()

    pose = np.load(args.track)["pose"]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(args.start * fps)
    last = len(pose) if args.end is None else min(len(pose), int(args.end * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SCALE), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SCALE)
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(first, last):
        ok, bgr = cap.read()
        if not ok:
            break
        img = cv2.resize(bgr, (w, h))
        draw(img, pose[i])
        cv2.putText(img, f"{i / fps:6.2f} s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(img)
    out.release()


if __name__ == "__main__":
    main()
