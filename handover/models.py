import numpy as np
import torch
from torch import nn

VEL_WINDOW = 3
KALMAN_ACC_STD = 3.0
KALMAN_MEAS_STD = 0.01


def stay(history, fps, horizon):
    return np.repeat(history[-1:], horizon, axis=0)


def const_velocity(history, fps, horizon):
    v = (history[-1] - history[-1 - VEL_WINDOW]) / VEL_WINDOW
    return history[-1] + np.arange(1, horizon + 1)[:, None] * v


def kalman_state(history, fps):
    dt = 1 / fps
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
    Q = G @ G.T * KALMAN_ACC_STD ** 2
    H = np.eye(2, 4)
    R = np.eye(2) * KALMAN_MEAS_STD ** 2
    x = np.r_[history[0], 0, 0]
    P = np.diag([KALMAN_MEAS_STD ** 2] * 2 + [1.0] * 2)
    for z in history[1:]:
        x, P = F @ x, F @ P @ F.T + Q
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        x, P = x + K @ (z - H @ x), (np.eye(4) - K @ H) @ P
    return x


def kalman(history, fps, horizon):
    x = kalman_state(history, fps)
    return x[:2] + np.arange(1, horizon + 1)[:, None] / fps * x[2:]


def min_jerk_profile(tau):
    tau = np.clip(tau, 0, 1)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def fit_min_jerk(history, fps, rest_frames, durations=np.arange(0.2, 4.0, 0.05)):
    t = np.arange(len(history)) / fps
    x0 = history[:rest_frames].mean(axis=0)
    rel = history - x0
    best = None
    for t0 in t[rest_frames // 2:]:
        s = min_jerk_profile((t[None, :] - t0) / durations[:, None])
        ss = (s ** 2).sum(axis=1)
        ok = ss > 1e-3
        amp = (s @ rel)[ok] / ss[ok, None]
        err = ((s[ok, :, None] * amp[:, None, :] - rel) ** 2).sum(axis=(1, 2))
        if len(err) and (best is None or err.min() < best[0]):
            i = np.argmin(err)
            best = (err[i], t0, durations[ok][i], x0 + amp[i])
    if best is None:
        return t[-1], durations[0], history[-1]
    _, t0, duration, end = best
    return t0, duration, end


def min_jerk(history, fps, horizon, rest_frames):
    t0, duration, end = fit_min_jerk(history, fps, rest_frames)
    t = (len(history) - 1 + np.arange(1, horizon + 1)) / fps
    x0 = history[:rest_frames].mean(axis=0)
    return x0 + min_jerk_profile((t - t0) / duration)[:, None] * (end - x0)


def min_jerk_point(history, fps, rest_frames):
    t0, duration, end = fit_min_jerk(history, fps, rest_frames)
    return end, max(0.0, t0 + duration - (len(history) - 1) / fps)


def features(h, t):
    g = h["giver_raw"][:t + 1]
    gv = np.vstack([np.zeros((1, 2)), np.diff(g, axis=0)]) * h["fps"]
    return np.hstack([g, gv]).astype(np.float32)


class WorldModel(nn.Module):
    def __init__(self, hidden=64, horizon=15):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(4, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2 * horizon + 3)

    def forward(self, x):
        out, _ = self.gru(x)
        y = self.head(out)
        traj = y[..., :2 * self.horizon].reshape(*y.shape[:-1], self.horizon, 2)
        point = y[..., 2 * self.horizon:2 * self.horizon + 2]
        ttc = nn.functional.softplus(y[..., -1])
        return traj, point, ttc
