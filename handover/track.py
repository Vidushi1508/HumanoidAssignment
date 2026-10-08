import argparse

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HUMAN, ROBOT = 0, 1
TORSO = [11, 12, 23, 24]


def track(video_path, model_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
    )
    frames = []
    with vision.PoseLandmarker.create_from_options(options) as detector:
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(i * 1000 / fps))

            people = np.full((2, 33, 4), np.nan, dtype=np.float32)
            poses = [np.array([[l.x * w, l.y * h, l.z, l.visibility] for l in p]) for p in result.pose_landmarks]
            poses.sort(key=lambda p: p[TORSO, 0].mean())
            if len(poses) == 2:
                people[HUMAN], people[ROBOT] = poses
            elif len(poses) == 1:
                people[HUMAN if poses[0][TORSO, 0].mean() < w / 2 else ROBOT] = poses[0]
            frames.append(people)

            i += 1
            if i % 500 == 0:
                print(f"{i} frames")
    cap.release()
    return np.stack(frames), fps, (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--model", default="pose_landmarker_heavy.task")
    args = ap.parse_args()

    pose, fps, size = track(args.video, args.model)
    np.savez_compressed(args.out, pose=pose, fps=fps, size=size)
    found = ~np.isnan(pose[:, :, 0, 0])
    print(f"{len(pose)} frames, {fps:.2f} fps; detected: human {found[:, HUMAN].mean():.0%}, "
          f"robot role {found[:, ROBOT].mean():.0%}, both {found.all(1).mean():.0%}")


if __name__ == "__main__":
    main()
