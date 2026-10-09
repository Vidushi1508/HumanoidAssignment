import argparse
import csv
import os
from multiprocessing import Pool
from pathlib import Path

import mujoco
import numpy as np
import torch

from handover import evaluate as ev
from handover import force as F
from handover import policy
from handover import sim as S
from handover import train as world
from handover.dataset import load, split
from handover.models import features

N_OBS = 18
N_ACT = 4
RESIDUAL_SPEED = 0.5
POPULATION = 48
ELITES = 8
ITERATIONS = 60
BATCH = 16
INIT_STD = [0.05, 0.05, 0.5, 0.5]
SERVO_GAIN = 6.0
MIN_STD = 0.05
WORKERS = 16
TUG_PENALTY = 0.03
RETURN_REWARD = 0.5
REACH_REWARD = 0.2
PUSH_PENALTY = 1.0
CHECKPOINT_EVERY = 10
DROP_Z = 0.2

_scenes = {}


def set_reward(reach, push, back):
    global REACH_REWARD, PUSH_PENALTY, RETURN_REWARD
    REACH_REWARD, PUSH_PENALTY, RETURN_REWARD = reach, push, back


def init_worker(reach, push, back):
    torch.set_num_threads(1)
    set_reward(reach, push, back)


def object_gap(handovers, h):
    return policy.grasp_offset([x for x in handovers if x["object"] == h["object"] and x["take"] != h["take"]])


def world_targets(model, handovers):
    out = {}
    for h in handovers:
        with torch.no_grad():
            point = model(torch.from_numpy(features(h, len(h["giver"]) - 1))[None])[1][0].numpy()
        out[(h["take"], h["card"])] = h["giver_raw"] + point
    return out


def scene_for(menagerie, gap):
    key = tuple(np.round(gap, 6))
    if key not in _scenes:
        _scenes[key] = F.Scene(F.build(menagerie, gap))
    return _scenes[key]


def observe(scene, h, t, predicted, gap, wrist, prev_wrist, start_force, closed):
    fps, d = h["fps"], scene.d
    giver = h["giver_raw"]
    g_vel = (giver[t] - giver[max(t - 1, 0)]) * fps
    here = scene.to_human(wrist)
    vel = (here - scene.to_human(prev_wrist)) * fps
    to_box = (scene.box_pos() - d.site_xpos[scene.tip])[[0, 2]] / S.SCALE
    to_target = predicted[t] + gap - here
    force = (scene.wrist_force() - start_force)[[0, 2]] / (F.BOX_MASS * F.G)
    return np.r_[giver[t] - here, g_vel, vel, to_box, np.linalg.norm(to_box), to_target, scene.touch() / 10,
                 (d.qpos[7] + d.qpos[8]) / 0.08, force, float(closed), t >= h["release"], 1.0]


def bootstrap():
    params = np.zeros((N_ACT, N_OBS))
    params[2, 8], params[2, N_OBS - 1] = -1 / F.CLOSE_TOL_M, 1.0
    params[3, N_OBS - 1] = -1.0
    return params.ravel()


def act(params, obs):
    return params.reshape(N_ACT, N_OBS) @ obs


def episode(params, h, gap, predicted, menagerie, base_gain, renderer=None):
    scene = scene_for(menagerie, gap)
    m, d, fps = scene.m, scene.d, h["fps"]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    start = scene.to_world(h["receiver_raw"][0])
    q_des = scene.place_arm(start)
    d.ctrl[:7], d.ctrl[7] = q_des, F.S.GRIPPER_OPEN
    box_from_wrist, _ = S.hand_geometry(gap)
    giver = h["giver_raw"]
    scene.reset_box(scene.to_world(giver[0]) + box_from_wrist)
    mujoco.mj_forward(m, d)
    start_force = scene.wrist_force()
    target, prev_wrist, moving, closed = start.copy(), start.copy(), False, False
    tug, grasp_t, frames, closest, push = 0.0, None, [], np.inf, 0.0
    for t in range(len(giver)):
        hand = scene.to_world(giver[t])
        d.mocap_pos[scene.hand_mocap] = hand
        level = F.ramp(t, h["release"], fps, rising=False)
        mujoco.mj_forward(m, d)
        wrist = d.site_xpos[scene.site].copy()
        obs = observe(scene, h, t, predicted, gap, wrist, prev_wrist, start_force, closed)
        if t < h["release"]:
            closest = min(closest, np.linalg.norm(scene.box_pos() - d.site_xpos[scene.tip]))
        a = act(params, obs)
        moving = moving or np.linalg.norm(giver[t] - giver[0]) > policy.MOVE_START
        base = policy.limit(base_gain * (predicted[t] + gap - scene.to_human(wrist))) if moving else np.zeros(2)
        to_box = (scene.box_pos() - d.site_xpos[scene.tip])[[0, 2]] / S.SCALE
        if np.linalg.norm(to_box) * S.SCALE < F.SERVO_RADIUS_M:
            base = policy.limit(SERVO_GAIN * to_box)
        if a[3] > 0:
            back = (start - wrist)[[0, 2]] / S.SCALE
            base = policy.limit(base_gain * back)
        vel = policy.limit(base + RESIDUAL_SPEED * np.tanh(a[:2]))
        closed = a[2] > 0
        d.ctrl[7] = F.SQUEEZE if closed else S.GRIPPER_OPEN
        held = closed and scene.touch() > F.TOUCH_N and d.qpos[7] + d.qpos[8] > F.MIN_OPENING_M
        if held and grasp_t is None:
            grasp_t = t
        if held and level > 0.5:
            tug = max(tug, abs(scene.wrist_force()[0] - start_force[0]))
        if grasp_t is None and level > 0.5:
            push = max(push, np.linalg.norm(scene.box_pos() - (hand + box_from_wrist)))
        target = S.lead(target + S.SCALE * np.array([vel[0], 0.0, vel[1]]) / fps, wrist)
        q_des = scene.step(q_des, target, level, hand + box_from_wrist)
        prev_wrist = wrist
        if renderer is not None:
            renderer.update_scene(d, camera="side")
            frames.append(renderer.render())
    box = scene.box_pos()
    in_hand = np.linalg.norm(box - d.site_xpos[scene.tip]) < F.HELD_DIST and d.qpos[7] + d.qpos[8] > F.MIN_OPENING_M
    dropped = box[2] < DROP_Z
    returned = 1 - min(1.0, np.linalg.norm(d.site_xpos[scene.site] - start) / max(np.linalg.norm(hand - start), 1e-6))
    reach = REACH_REWARD * np.clip(1 - closest / F.SERVO_RADIUS_M, 0, 1)
    reward = (float(in_hand) - float(dropped) - TUG_PENALTY * tug - PUSH_PENALTY * push
              + (RETURN_REWARD * returned if in_hand else 0.0) + reach)
    return reward, dict(success=bool(in_hand), dropped=bool(dropped), peak_tug_n=tug, push_cm=100 * push,
                        returned=float(returned),
                        grasp_minus_contact_s=(grasp_t - h["contact"]) / fps if grasp_t is not None else np.nan), frames


def _run(job):
    params, h, gap, predicted, menagerie, gain = job
    return episode(params, h, gap, predicted, menagerie, gain)[:2]


def random_batches(n, rng):
    return lambda: rng.choice(n, BATCH, replace=False)


def hard_batches(fails, wins, rng):
    return lambda: np.r_[fails, rng.choice(wins, min(len(fails), len(wins)), replace=False)]


def cem(train_set, gaps, preds, menagerie, gain, seed, pool, init, noise_scale, checkpoint, batches=None):
    rng = np.random.default_rng(seed)
    batches = batches or random_batches(len(train_set), rng)
    mean, std = init.copy(), noise_scale * np.repeat(INIT_STD, N_OBS).astype(float)
    history = []
    for it in range(ITERATIONS):
        batch = batches()
        candidates = mean + std * rng.standard_normal((POPULATION, mean.size))
        jobs = [(c, train_set[i], gaps[i], preds[i], menagerie, gain) for c in candidates for i in batch]
        rewards = np.array([r for r, _ in pool.map(_run, jobs)]).reshape(POPULATION, len(batch)).mean(1)
        elite = candidates[np.argsort(rewards)[-ELITES:]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), MIN_STD)
        history.append((it, rewards.mean(), rewards.max()))
        print(f"  seed {seed} iteration {it:2d}: mean reward {rewards.mean():+.2f}, best {rewards.max():+.2f}", flush=True)
        if (it + 1) % CHECKPOINT_EVERY == 0:
            np.save(checkpoint, mean)
    return mean, history


def per_seed_table(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    print(f"{'policy':24s}{'seed':>6s}{'train success':>15s}{'test success':>14s}")
    for method in dict.fromkeys(r["method"] for r in rows):
        for seed in dict.fromkeys(r["seed"] for r in rows):
            rate = {sp: np.mean([r["success"] == "True" for r in rows
                                 if r["method"] == method and r["seed"] == seed and r["split"] == sp])
                    for sp in ("train", "test")}
            print(f"{method:24s}{seed:>6s}{rate['train']:15.0%}{rate['test']:14.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--menagerie", default="mujoco_menagerie")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=policy.SEEDS)
    ap.add_argument("--init", help="folder with rl_policy_seed*.npy to continue training from (default: bootstrap)")
    ap.add_argument("--init-noise", type=float, default=1.0, help="scale of the exploration noise")
    ap.add_argument("--reach-reward", type=float, default=REACH_REWARD)
    ap.add_argument("--push-penalty", type=float, default=PUSH_PENALTY, help="per metre the box is pushed before the grasp")
    ap.add_argument("--return-reward", type=float, default=RETURN_REWARD)
    ap.add_argument("--summary", help="print train/test success per policy and seed from an rl results csv")
    ap.add_argument("--start", help="one saved policy (.npy) to continue from; use with a single --seeds value")
    ap.add_argument("--hard", action="store_true",
                    help="batches = all training handovers the start policy fails + as many it succeeds on")
    ap.add_argument("--gif", nargs="*", default=[], help="TAKE:CARD test handovers to render (first seed)")
    ap.add_argument("--media", default="media")
    ap.add_argument("--render-only", action="store_true", help="skip training; load saved policies from --out")
    args = ap.parse_args()
    if args.summary:
        per_seed_table(args.summary)
        return

    torch.set_num_threads(4)
    reward = (args.reach_reward, args.push_penalty, args.return_reward)
    set_reward(*reward)
    handovers = load(args.data, args.sheet, tail_s=F.TAIL_S)
    train_set, test_set = split(handovers, "main")
    horizon = round(ev.HORIZON_S * handovers[0]["fps"])
    if args.render_only:
        render(args, handovers, train_set, test_set, horizon)
        return
    rows = []
    with Pool(WORKERS, initializer=init_worker, initargs=reward) as pool:
        for seed in args.seeds:
            wm = world.train(train_set, seed, horizon)
            targets = world_targets(wm, handovers)
            gain = policy.build_controllers(train_set, seed, horizon)["world_model_target"].gain
            tr_gaps = [object_gap(handovers, h) for h in train_set]
            tr_preds = [targets[(h["take"], h["card"])] for h in train_set]
            if args.start:
                start = np.load(args.start)
            elif args.init:
                start = np.load(Path(args.init) / f"rl_policy_seed{seed}.npy")
            else:
                start = bootstrap()
            batches = None
            if args.hard:
                ok = np.array([info["success"] for _, info in pool.map(
                    _run, [(start, h, g, pr, args.menagerie, gain) for h, g, pr in zip(train_set, tr_gaps, tr_preds)])])
                fails, wins = np.flatnonzero(~ok), np.flatnonzero(ok)
                print(f"  start policy: {len(wins)}/{len(train_set)} training handovers succeed; "
                      f"training on {len(fails)} failures + {min(len(fails), len(wins))} successes per batch", flush=True)
                batches = hard_batches(fails, wins, np.random.default_rng(seed))
            params, _ = cem(train_set, tr_gaps, tr_preds, args.menagerie, gain, seed, pool, start, args.init_noise,
                            Path(args.out) / f"rl_policy_seed{seed}_checkpoint.npy", batches)
            np.save(Path(args.out) / f"rl_policy_seed{seed}.npy", params)
            for split_name, hset in [("test", test_set), ("train", train_set)]:
                for method, p in [("rl_policy", params), ("before this training", start)]:
                    jobs = [(p, h, object_gap(handovers, h), targets[(h["take"], h["card"])], args.menagerie, gain)
                            for h in hset]
                    for h, (reward, info) in zip(hset, pool.map(_run, jobs)):
                        rows.append(dict(split=split_name, take=h["take"], card=h["card"], seed=seed, method=method,
                                         reward=reward, **info))
    ev.write_csv(rows, Path(args.out) / "rl_receiving.csv")
    print(f"Learned receiving in physical simulation, {len(args.seeds)} seeds; test = A3, B3 (box, unseen)")
    for split_name, method in [(s_, m_) for s_ in ("test", "train") for m_ in ("rl_policy", "before this training")]:
        sel = [r for r in rows if r["method"] == method and r["split"] == split_name]
        per_seed = [np.mean([r["success"] for r in sel if r["seed"] == s_]) for s_ in args.seeds]
        method = f"{split_name}: {method} (seeds {min(per_seed):.0%}-{max(per_seed):.0%})"
        print(f"  {method:52s} success {np.mean([r['success'] for r in sel]):4.0%}  dropped {np.mean([r['dropped'] for r in sel]):4.0%}"
              f"  tug {np.mean([r['peak_tug_n'] for r in sel]):4.1f} N  pushed {np.mean([r['push_cm'] for r in sel]):4.1f} cm"
              f"  brought back {np.mean([r['returned'] for r in sel if r['success']] or [np.nan]):4.0%}"
              f"  grasp vs contact {np.nanmean([r['grasp_minus_contact_s'] for r in sel]):+.2f} s")


def render(args, handovers, train_set, test_set, horizon):
    seed = args.seeds[0]
    targets = world_targets(world.train(train_set, seed, horizon), handovers)
    gain = policy.build_controllers(train_set, seed, horizon)["world_model_target"].gain
    params = np.load(Path(args.out) / f"rl_policy_seed{seed}.npy")
    Path(args.media).mkdir(exist_ok=True)
    for spec in args.gif:
        take, card = spec.split(":")
        h = next(x for x in test_set if x["take"] == take and x["card"] == int(card))
        gap = object_gap(handovers, h)
        renderer = mujoco.Renderer(scene_for(args.menagerie, gap).m, 240, 320)
        panels = []
        for name, p in [("bootstrap (before RL)", bootstrap()), (f"after RL + fine-tuning (seed {seed})", params)]:
            _, info, frames = episode(p, h, gap, targets[(take, int(card))], args.menagerie, gain, renderer)
            status = f"holds box, grasp {info['grasp_minus_contact_s']:+.2f} s" if info["success"] else "box not held"
            panels.append([S.label(f, f"{name}: {status}") for f in frames])
        S.save_gif([np.hstack(p) for p in zip(*panels)], Path(args.media) / f"rl_{take}_card{card}.gif")


if __name__ == "__main__":
    main()
