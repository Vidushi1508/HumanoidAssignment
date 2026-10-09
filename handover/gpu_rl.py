import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import torch
from mujoco import mjx
from mujoco.mjx._src import support

from handover import evaluate as ev
from handover import force as F
from handover import policy
from handover import rl
from handover import sim as S
from handover import train as world
from handover.dataset import load, split

HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
SUBSTEPS = 8
GRIP_GAIN = F.GRIP_FORCE_N / F.BOX_HALF[1]
HIDDEN = 32
N_LINEAR = rl.N_ACT * rl.N_OBS
N_PARAMS = N_LINEAR + rl.N_OBS * HIDDEN + HIDDEN + HIDDEN * rl.N_ACT + rl.N_ACT
PAIRS = 32
ITERATIONS = 100
LEARNING_RATE = 0.03
BATCH_HANDOVERS = 20
SIGMA_LINEAR = [0.02, 0.02, 0.2, 0.2]
SIGMA_NETWORK = 0.05


def build(menagerie):
    spec = mujoco.MjSpec.from_file(str(Path(menagerie) / "franka_emika_panda" / "mjx_panda.xml"))
    spec.body("link0").pos = [0, 0, S.BASE_Z]
    world_body = spec.worldbody
    world_body.add_light(pos=[0.5, -1.5, 2.5], dir=[0, 0.5, -1])
    world_body.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.85, 0.85, 0.85, 1],
                        contype=1, conaffinity=1)
    world_body.add_camera(name="side", pos=[0.65, -2.0, 0.95], xyaxes=[1, 0, 0, 0, 0, 1], fovy=45)
    hand = spec.body("hand")
    hand.add_site(name="wrist", size=[0.01, 0, 0], rgba=[1, 0, 0, 0])
    hand.add_site(name="fingertips", pos=[0, 0, S.FINGER_REACH_M], size=[0.01, 0, 0], rgba=[1, 0, 0, 0])
    human = world_body.add_body(name="human_hand", mocap=True)
    human.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], rgba=[0.85, 0.6, 0.45, 1],
                   contype=0, conaffinity=0)
    box = world_body.add_body(name="object", pos=[1, 0, 1])
    box.add_freejoint()
    box.add_geom(name="object", type=mujoco.mjtGeom.mjGEOM_BOX, size=F.BOX_HALF, mass=F.BOX_MASS,
                 rgba=[0.2, 0.45, 0.9, 1], contype=2, conaffinity=1, condim=3, friction=[1.5, 0.03, 0.003])
    spec.add_sensor(type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="wrist")
    gripper = spec.actuator("actuator8")
    gripper.gainprm[0] = GRIP_GAIN
    gripper.biasprm[1] = -GRIP_GAIN
    gripper.forcerange = [-2 * F.GRIP_FORCE_N, 2 * F.GRIP_FORCE_N]
    model = spec.compile()
    model.opt.timestep = 1 / (30 * SUBSTEPS)
    return model


class Ids:
    def __init__(self, m):
        self.wrist = m.site("wrist").id
        self.tip = m.site("fingertips").id
        self.hand = m.body("hand").id
        self.box = m.body("object").id
        self.box_q = m.jnt_qposadr[m.body("object").jntadr[0]]
        self.box_v = m.jnt_dofadr[m.body("object").jntadr[0]]
        self.box_geom = m.geom("object").id
        fingers = [m.body("left_finger").id, m.body("right_finger").id]
        self.finger_of_geom = np.array([fingers.index(b) if b in fingers else -1 for b in m.geom_bodyid])
        self.mocap = m.body("human_hand").mocapid[0]


def shoulder_world(m):
    d = mujoco.MjData(m)
    d.qpos[:9] = HOME
    mujoco.mj_forward(m, d)
    return d.xpos[m.body("link2").id].copy()


def to_world(shoulder, xy):
    xy = np.asarray(xy)
    return shoulder + S.SCALE * np.stack([xy[..., 0], np.zeros(xy.shape[:-1]), xy[..., 1]], axis=-1)


def initial_pose(m, ids, target):
    d = mujoco.MjData(m)
    q = HOME[:7].copy()
    for _ in range(300):
        d.qpos[:9] = np.r_[q, 0.04, 0.04]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, ids.wrist)
        R = d.site_xmat[ids.wrist].reshape(3, 3)
        rot = 0.5 * sum(np.cross(R[:, i], S.TARGET_R[:, i]) for i in range(3))
        err = np.r_[target - d.site_xpos[ids.wrist], S.ORIENTATION_WEIGHT * rot]
        J = np.vstack([jp, S.ORIENTATION_WEIGHT * jr])[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + S.IK_DAMPING ** 2 * np.eye(6), err)
        dq = dq + 0.1 * (np.eye(7) - np.linalg.pinv(J) @ J) @ (HOME[:7] - q)
        q = q + dq * min(1.0, S.MAX_JOINT_STEP / np.abs(dq).max())
    return q


def prepare(handovers, everything, m, ids, targets, seeds, T):
    shoulder = shoulder_world(m)
    out = {k: [] for k in ["giver", "box_off", "pred", "release", "contact", "n", "start", "q0", "box0"]}
    for h in handovers:
        n = len(h["giver"])
        pad = lambda a: np.concatenate([a, np.repeat(a[-1:], T - n, axis=0)])
        gap = rl.object_gap(everything, h)
        box_off, _ = S.hand_geometry(gap)
        giver = to_world(shoulder, pad(h["giver_raw"]))
        start = to_world(shoulder, h["receiver_raw"][0])
        out["giver"].append(giver)
        out["box_off"].append(box_off)
        out["pred"].append([to_world(shoulder, pad(targets[s][(h["take"], h["card"])] + gap)) for s in seeds])
        out["release"].append(h["release"])
        out["contact"].append(h["contact"])
        out["n"].append(n)
        out["start"].append(start)
        out["q0"].append(initial_pose(m, ids, start))
        out["box0"].append(giver[0] + box_off)
    ints = ("release", "contact", "n")
    return {k: jnp.asarray(np.array(v), dtype=jnp.int32 if k in ints else jnp.float32) for k, v in out.items()}, shoulder


def rotation_error(R):
    T = jnp.asarray(S.TARGET_R)
    return 0.5 * (jnp.cross(R[:, 0], T[:, 0]) + jnp.cross(R[:, 1], T[:, 1]) + jnp.cross(R[:, 2], T[:, 2]))


def act(params, obs):
    linear = params[:N_LINEAR].reshape(rl.N_ACT, rl.N_OBS)
    i = N_LINEAR
    w1 = params[i:i + rl.N_OBS * HIDDEN].reshape(HIDDEN, rl.N_OBS)
    i += rl.N_OBS * HIDDEN
    b1 = params[i:i + HIDDEN]
    i += HIDDEN
    w2 = params[i:i + HIDDEN * rl.N_ACT].reshape(rl.N_ACT, HIDDEN)
    b2 = params[i + HIDDEN * rl.N_ACT:]
    return linear @ obs + w2 @ jnp.tanh(w1 @ obs + b1) + b2


def initial_params(rng):
    params = np.zeros(N_PARAMS)
    params[:N_LINEAR] = rl.bootstrap()
    params[N_LINEAR:N_LINEAR + rl.N_OBS * HIDDEN] = rng.standard_normal(rl.N_OBS * HIDDEN) / np.sqrt(rl.N_OBS)
    return params


def sigmas():
    out = np.full(N_PARAMS, SIGMA_NETWORK)
    out[:N_LINEAR] = np.repeat(SIGMA_LINEAR, rl.N_OBS)
    return out


def make_episode(m, mx, ids, shoulder, T):
    home = jnp.asarray(HOME[:7])
    lam = S.IK_DAMPING ** 2 * jnp.eye(6)
    g_up = jnp.array([0.0, 0.0, F.BOX_MASS * F.G])
    damping = 2 * np.sqrt(F.HUMAN_STIFFNESS * F.BOX_MASS)
    finger_of_geom = jnp.asarray(ids.finger_of_geom)
    shoulder = jnp.asarray(shoulder)
    max_lead = S.MAX_LEAD_M
    fps = 30.0

    def to_h(p):
        return (p - shoulder)[jnp.array([0, 2])] / S.SCALE

    def lift(v):
        return S.SCALE * jnp.array([v[0], 0.0, v[1]])

    def limit(v, top):
        s = jnp.linalg.norm(v)
        return v * jnp.minimum(1.0, top / jnp.maximum(s, 1e-9))

    def ik(d, q_des, target):
        dq_full = d.qpos.at[:7].set(q_des)
        dk = mjx.kinematics(mx, d.replace(qpos=dq_full))
        dk = mjx.com_pos(mx, dk)
        p = dk.site_xpos[ids.wrist]
        R = dk.site_xmat[ids.wrist]
        jp, jr = support.jac(mx, dk, p, ids.hand)
        J = jnp.concatenate([jp.T, S.ORIENTATION_WEIGHT * jr.T])[:, :7]
        err = jnp.concatenate([target - p, S.ORIENTATION_WEIGHT * rotation_error(R)])
        dq = J.T @ jnp.linalg.solve(J @ J.T + lam, err)
        dq = dq + 0.1 * (jnp.eye(7) - jnp.linalg.pinv(J) @ J) @ (home - q_des)
        return q_des + dq * jnp.minimum(1.0, S.MAX_JOINT_STEP / jnp.max(jnp.abs(dq)))

    def touch(d):
        c = d._impl.contact
        rows = c.efc_address[:, None] + jnp.arange(4)
        normal = d._impl.efc_force[jnp.clip(rows, 0, d._impl.efc_force.shape[0] - 1)].sum(1)
        g1, g2 = c.geom[:, 0], c.geom[:, 1]
        other = jnp.where(g1 == ids.box_geom, g2, g1)
        finger = finger_of_geom[other]
        valid = ((g1 == ids.box_geom) | (g2 == ids.box_geom)) & (finger >= 0) & (c.dist <= 0) & (c.efc_address >= 0)
        left = jnp.where(valid & (finger == 0), normal, 0.0).sum()
        right = jnp.where(valid & (finger == 1), normal, 0.0).sum()
        return jnp.minimum(left, right)

    def wrist_force(d):
        return d.site_xmat[ids.wrist] @ d.sensordata[:3]

    def episode(params, ho):
        d = mjx.make_data(mx)
        qpos = jnp.zeros(mx.nq).at[:7].set(ho["q0"]).at[7:9].set(0.04)
        qpos = qpos.at[ids.box_q:ids.box_q + 3].set(ho["box0"]).at[ids.box_q + 3].set(1.0)
        d = d.replace(qpos=qpos, ctrl=jnp.zeros(mx.nu).at[:7].set(ho["q0"]).at[7].set(0.04),
                      mocap_pos=d.mocap_pos.at[ids.mocap].set(ho["giver"][0]))
        d = mjx.forward(mx, d)
        start_force = wrist_force(d)
        carry = dict(d=d, q_des=ho["q0"], target=ho["start"], prev=ho["start"], moving=False, closed=False,
                     grasp_t=jnp.int32(-1), tug=0.0, push=0.0, closest=jnp.inf, final_box=ho["box0"],
                     final_tip=ho["start"], final_open=0.08, final_wrist=ho["start"], final_hand=ho["giver"][0])

        def control(c, t):
            d = c["d"]
            valid = t < ho["n"]
            hand = ho["giver"][t]
            d = d.replace(mocap_pos=d.mocap_pos.at[ids.mocap].set(hand))
            level = 1.0 - jnp.clip((t - ho["release"]) / (F.GRIP_RAMP_S * fps), 0.0, 1.0)
            wrist, tip = d.site_xpos[ids.wrist], d.site_xpos[ids.tip]
            box = d.xpos[ids.box]
            here = to_h(wrist)
            g_now, g_prev = to_h(hand), to_h(ho["giver"][jnp.maximum(t - 1, 0)])
            to_box = (box - tip)[jnp.array([0, 2])] / S.SCALE
            force = (wrist_force(d) - start_force)[jnp.array([0, 2])] / (F.BOX_MASS * F.G)
            opening = d.qpos[7] + d.qpos[8]
            tch = touch(d)
            pred_h = to_h(ho["pred"][t])
            obs = jnp.concatenate([g_now - here, (g_now - g_prev) * fps, (here - to_h(c["prev"])) * fps, to_box,
                                   jnp.array([jnp.linalg.norm(to_box)]), pred_h - here,
                                   jnp.array([tch / 10, opening / 0.08]), force,
                                   jnp.array([c["closed"] * 1.0, (t >= ho["release"]) * 1.0, 1.0])])
            a = act(params, obs)
            moving = c["moving"] | (jnp.linalg.norm(g_now - to_h(ho["giver"][0])) > policy.MOVE_START)
            base = jnp.where(moving, limit(ho["gain"] * (pred_h - here), policy.MAX_SPEED), 0.0)
            near = jnp.linalg.norm(box - tip) < F.SERVO_RADIUS_M
            base = jnp.where(near, limit(rl.SERVO_GAIN * to_box, policy.MAX_SPEED), base)
            back = to_h(ho["start"]) - here
            base = jnp.where(a[3] > 0, limit(ho["gain"] * back, policy.MAX_SPEED), base)
            vel = limit(base + rl.RESIDUAL_SPEED * jnp.tanh(a[:2]), policy.MAX_SPEED)
            closed = a[2] > 0
            ctrl = d.ctrl.at[7].set(jnp.where(closed, 0.0, 0.04))
            held = closed & (tch > F.TOUCH_N) & (opening > F.MIN_OPENING_M)
            grasp_t = jnp.where((c["grasp_t"] < 0) & held & valid, t, c["grasp_t"])
            tug = jnp.where(held & (level > 0.5) & valid,
                            jnp.maximum(c["tug"], jnp.abs(wrist_force(d)[0] - start_force[0])), c["tug"])
            push = jnp.where((grasp_t < 0) & (level > 0.5) & valid,
                             jnp.maximum(c["push"], jnp.linalg.norm(box - (hand + ho["box_off"]))), c["push"])
            closest = jnp.where((t < ho["release"]) & valid, jnp.minimum(c["closest"], jnp.linalg.norm(box - tip)),
                                c["closest"])
            target = c["target"] + lift(vel) / fps
            ahead = target - wrist
            target = wrist + ahead * jnp.minimum(1.0, max_lead / jnp.maximum(jnp.linalg.norm(ahead), 1e-9))
            q_des = ik(d, c["q_des"], target)
            d = d.replace(ctrl=ctrl.at[:7].set(q_des))
            human_target = hand + ho["box_off"]

            def substep(_, d):
                pos, vel_b = d.xpos[ids.box], d.qvel[ids.box_v:ids.box_v + 3]
                spring = F.HUMAN_STIFFNESS * (human_target - pos) - damping * vel_b
                spring = spring * jnp.minimum(1.0, F.HUMAN_MAX_FORCE_N / (jnp.linalg.norm(spring) + 1e-9))
                quat, ang = d.qpos[ids.box_q + 3:ids.box_q + 7], d.qvel[ids.box_v + 3:ids.box_v + 6]
                rot = 2 * quat[1:] * jnp.sign(quat[0])
                xf = d.xfrc_applied.at[ids.box, :3].set(level * (spring + g_up))
                xf = xf.at[ids.box, 3:].set(level * (-F.HUMAN_ROT_STIFFNESS * rot - 0.02 * ang))
                d = d.replace(xfrc_applied=xf, qfrc_applied=d.qfrc_applied.at[:7].set(d.qfrc_bias[:7]))
                return mjx.step(mx, d)

            d = jax.lax.fori_loop(0, SUBSTEPS, substep, d)
            last = t == ho["n"] - 1
            new = dict(d=d, q_des=q_des, target=target, prev=wrist, moving=moving, closed=closed, grasp_t=grasp_t,
                       tug=tug, push=push, closest=closest,
                       final_box=jnp.where(last, d.xpos[ids.box], c["final_box"]),
                       final_tip=jnp.where(last, d.site_xpos[ids.tip], c["final_tip"]),
                       final_open=jnp.where(last, d.qpos[7] + d.qpos[8], c["final_open"]),
                       final_wrist=jnp.where(last, d.site_xpos[ids.wrist], c["final_wrist"]),
                       final_hand=jnp.where(last, hand, c["final_hand"]))
            return new, None

        c, _ = jax.lax.scan(control, carry, jnp.arange(T))
        in_hand = (jnp.linalg.norm(c["final_box"] - c["final_tip"]) < F.HELD_DIST) & (c["final_open"] > F.MIN_OPENING_M)
        dropped = c["final_box"][2] < rl.DROP_Z
        returned = 1 - jnp.minimum(1.0, jnp.linalg.norm(c["final_wrist"] - ho["start"])
                                   / jnp.maximum(jnp.linalg.norm(c["final_hand"] - ho["start"]), 1e-6))
        reach = rl.REACH_REWARD * jnp.clip(1 - c["closest"] / F.SERVO_RADIUS_M, 0, 1)
        reward = (in_hand * 1.0 - dropped * 1.0 - rl.TUG_PENALTY * c["tug"] - rl.PUSH_PENALTY * c["push"]
                  + jnp.where(in_hand, rl.RETURN_REWARD * returned, 0.0) + reach)
        grasp_s = jnp.where(c["grasp_t"] >= 0, (c["grasp_t"] - ho["contact"]) / fps, jnp.nan)
        return reward, dict(success=in_hand, dropped=dropped, peak_tug_n=c["tug"], push_cm=100 * c["push"],
                            returned=returned, grasp_minus_contact_s=grasp_s)

    return episode


def make_evaluate(episode):
    def run(params, data, s, b):
        ho = {k: v[b] for k, v in data.items() if k not in ("pred", "gain")}
        ho["pred"], ho["gain"] = data["pred"][b, s], data["gain"][s]
        return episode(params, ho)

    @jax.jit
    def evaluate(params, data):
        n_seeds, n_cand, _ = params.shape
        n_ho = data["n"].shape[0]
        s, c, b = jnp.meshgrid(jnp.arange(n_seeds), jnp.arange(n_cand), jnp.arange(n_ho), indexing="ij")
        flat = params[s.ravel(), c.ravel()]
        reward, info = jax.vmap(run, in_axes=(0, None, 0, 0))(flat, data, s.ravel(), b.ravel())
        shape = (n_seeds, n_cand, n_ho)
        return reward.reshape(shape), {k: v.reshape(shape) for k, v in info.items()}

    return evaluate


def chosen_gain(train_set, targets):
    def target(h, t):
        return targets[(h["take"], h["card"])][t]
    return policy.best_gain(train_set, target, policy.grasp_offset(train_set))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--menagerie", default="mujoco_menagerie")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=policy.SEEDS)
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    ap.add_argument("--pairs", type=int, default=PAIRS)
    args = ap.parse_args()

    torch.set_num_threads(4)
    everything = load(args.data, args.sheet, tail_s=F.TAIL_S)
    train_set, test_set = split(everything, "main")
    horizon = round(ev.HORIZON_S * everything[0]["fps"])
    targets = {s: rl.world_targets(world.train(train_set, s, horizon), everything) for s in args.seeds}
    m = build(args.menagerie)
    mx = mjx.put_model(m)
    ids = Ids(m)
    T = max(len(h["giver"]) for h in everything)
    tr, shoulder = prepare(train_set, everything, m, ids, targets, args.seeds, T)
    te, _ = prepare(test_set, everything, m, ids, targets, args.seeds, T)
    gains = jnp.asarray([chosen_gain(train_set, targets[s]) for s in args.seeds], dtype=jnp.float32)
    tr["gain"], te["gain"] = gains, gains
    evaluate = make_evaluate(make_episode(m, mx, ids, shoulder, T))
    n_seeds = len(args.seeds)
    print(f"{len(train_set)} train / {len(test_set)} test handovers, {n_seeds} seeds x {2 * args.pairs + 1} candidates "
          f"-> {n_seeds * (2 * args.pairs + 1) * len(train_set)} parallel episodes per iteration on {jax.devices()[0]}",
          flush=True)

    rng = np.random.default_rng(0)
    theta = np.stack([initial_params(np.random.default_rng(seed)) for seed in args.seeds])
    base = theta.copy()
    sigma = sigmas()
    z = np.zeros_like(theta)
    m1, m2 = np.zeros_like(theta), np.zeros_like(theta)
    for it in range(args.iterations):
        t0 = time.time()
        eps = rng.standard_normal((n_seeds, args.pairs, N_PARAMS))
        noise = np.concatenate([eps, -eps], axis=1)
        cand = np.concatenate([theta[:, None], theta[:, None] + sigma * noise], axis=1)
        pick = jnp.asarray(rng.choice(len(train_set), BATCH_HANDOVERS, replace=False))
        batch = {k: (v if k == "gain" else v[pick]) for k, v in tr.items()}
        reward, _ = evaluate(jnp.asarray(cand, dtype=jnp.float32), batch)
        score = np.asarray(reward).mean(-1)
        ranks = score[:, 1:].argsort(1).argsort(1) / (2 * args.pairs - 1) - 0.5
        grad = np.einsum("sp,spd->sd", ranks, noise) / (2 * args.pairs)
        m1 = 0.9 * m1 + 0.1 * grad
        m2 = 0.999 * m2 + 0.001 * grad ** 2
        z = z + LEARNING_RATE * (m1 / (1 - 0.9 ** (it + 1))) / (np.sqrt(m2 / (1 - 0.999 ** (it + 1))) + 1e-8)
        theta = base + sigma * z
        print(f"  iteration {it:3d} ({time.time() - t0:5.1f} s): current policy reward per seed "
              f"{' '.join(f'{v:+.2f}' for v in score[:, 0])}, best candidate {score.max():+.2f}", flush=True)
    mean = theta

    rows = []
    for split_name, data, hset in [("test", te, test_set), ("train", tr, train_set)]:
        for method, params in [("rl_policy_gpu", mean), ("bootstrap (before RL)", base)]:
            _, info = evaluate(jnp.asarray(params[:, None], dtype=jnp.float32), data)
            info = {k: np.asarray(v)[:, 0] for k, v in info.items()}
            for si, seed in enumerate(args.seeds):
                for bi, h in enumerate(hset):
                    rows.append(dict(split=split_name, take=h["take"], card=h["card"], seed=seed, method=method,
                                     **{k: float(v[si, bi]) for k, v in info.items()}))
    for si, seed in enumerate(args.seeds):
        np.save(Path(args.out) / f"rl_gpu_policy_seed{seed}.npy", mean[si])
    ev.write_csv(rows, Path(args.out) / "rl_gpu_receiving.csv")
    print(f"Learned receiving in physical simulation (MJX on GPU), {n_seeds} seeds; test = A3, B3 (box, unseen)")
    for split_name in ("test", "train"):
        for method in ("rl_policy_gpu", "bootstrap (before RL)"):
            sel = [r for r in rows if r["split"] == split_name and r["method"] == method]
            per_seed = [np.mean([r["success"] for r in sel if r["seed"] == s]) for s in args.seeds]
            print(f"  {split_name}: {method:24s} success {np.mean([r['success'] for r in sel]):4.0%} "
                  f"(seeds {min(per_seed):.0%}-{max(per_seed):.0%})  dropped {np.mean([r['dropped'] for r in sel]):4.0%}  "
                  f"tug {np.mean([r['peak_tug_n'] for r in sel]):4.1f} N  pushed {np.mean([r['push_cm'] for r in sel]):4.1f} cm  "
                  f"grasp vs contact {np.nanmean([r['grasp_minus_contact_s'] for r in sel]):+.2f} s")


if __name__ == "__main__":
    main()
