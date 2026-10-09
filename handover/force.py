import argparse
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
import torch

from handover import evaluate as ev
from handover import policy
from handover import release as rel
from handover import sim as S
from handover.dataset import load, split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOX_HALF = [0.03, 0.025, 0.04]
MIN_OPENING_M = 0.02
CLOSE_SETTLE_S = 0.2
BOX_MASS = 0.1
G = 9.81
GRIP_RAMP_S = 0.2
HUMAN_STIFFNESS = 80.0
HUMAN_MAX_FORCE_N = 8.0
PERSIST_S = 0.1
HUMAN_ROT_STIFFNESS = 0.5
SQUEEZE = 0
TOUCH_N = 0.3
SERVO_RADIUS_M = 0.10
CLOSE_TOL_M = 0.025
ALIGN_TOL_M = 0.02
STANDOFF_M = 0.05
GRIP_FORCE_N = 30.0
CLOSE_TIMEOUT_S = 0.4
WEIGHT_SHARE = 0.5
PULL_N = 3.0
SETTLE_S = 0.5
READY_FORWARD_M = 0.35
TAIL_S = 1.0
HELD_DIST = 0.06
RECEIVE_MODES = ["retract_after_grasp", "wait_for_weight"]
GIVE_RULES = ["at_contact", "mean_hold", "predicted_hold", "weight_share", "pull"]


def build(menagerie, offset):
    spec = mujoco.MjSpec.from_file(str(Path(menagerie) / "franka_emika_panda" / "panda.xml"))
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[1, 1, 1], rgb2=[0.75, 0.8, 0.88], width=256, height=256)
    spec.body("link0").pos = [0, 0, S.BASE_Z]
    world = spec.worldbody
    world.add_light(pos=[0.5, -1.5, 2.5], dir=[0, 0.5, -1])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.09, S.BASE_Z / 2, 0], pos=[0, 0, S.BASE_Z / 2],
                   rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0)
    world.add_camera(name="side", pos=[0.65, -2.0, 0.95], xyaxes=[1, 0, 0, 0, 0, 1], fovy=45)
    hand = spec.body("hand")
    hand.add_site(name="wrist", size=[0.01, 0, 0], rgba=[1, 0, 0, 0])
    hand.add_site(name="fingertips", pos=[0, 0, S.FINGER_REACH_M], size=[0.01, 0, 0], rgba=[1, 0, 0, 0])
    box_from_wrist, direction = S.hand_geometry(offset)
    human = world.add_body(name="human_hand", mocap=True)
    human.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.025, 0, 0], rgba=[0.85, 0.6, 0.45, 1],
                   fromto=[0, 0, 0, *(box_from_wrist - BOX_HALF[0] * direction)], contype=0, conaffinity=0)
    box = world.add_body(name="object")
    box.add_freejoint()
    box.add_geom(name="object", type=mujoco.mjtGeom.mjGEOM_BOX, size=BOX_HALF, mass=BOX_MASS,
                 rgba=[0.2, 0.45, 0.9, 1], friction=[1.5, 0.01, 0.001])
    spec.add_sensor(type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="wrist")
    gripper = spec.actuator("actuator8")
    stiffness = GRIP_FORCE_N / BOX_HALF[1]
    gripper.gainprm[0] = stiffness * 0.04 / 255
    gripper.biasprm[1] = -stiffness
    gripper.forcerange = [-2 * GRIP_FORCE_N, 2 * GRIP_FORCE_N]
    return spec.compile()


class Scene(S.Sim):
    def __init__(self, model):
        super().__init__(model)
        self.box = model.body("object").id
        self.box_geom = model.geom("object").id
        self.box_q = model.jnt_qposadr[model.body("object").jntadr[0]]
        self.box_v = model.jnt_dofadr[model.body("object").jntadr[0]]
        self.fingers = [model.body("left_finger").id, model.body("right_finger").id]
        self.wrist_site = model.site("wrist").id

    def reset_box(self, pos):
        self.d.qpos[self.box_q:self.box_q + 3] = pos
        self.d.qpos[self.box_q + 3:self.box_q + 7] = [1, 0, 0, 0]
        self.d.qvel[self.box_v:self.box_v + 6] = 0

    def box_pos(self):
        return self.d.xpos[self.box].copy()

    def touch(self):
        force = np.zeros(6)
        per_finger = {b: 0.0 for b in self.fingers}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if self.box_geom not in (c.geom1, c.geom2):
                continue
            other = self.m.geom_bodyid[c.geom2 if c.geom1 == self.box_geom else c.geom1]
            if other in per_finger:
                mujoco.mj_contactForce(self.m, self.d, i, force)
                per_finger[other] += abs(force[0])
        return min(per_finger.values())

    def wrist_force(self):
        return self.d.site_xmat[self.wrist_site].reshape(3, 3) @ self.d.sensordata[:3]

    def step(self, q_des, target, human_level, human_target):
        q_now, v_now = self.d.qpos.copy(), self.d.qvel.copy()
        q_prev, q_des = q_des, self.ik_solve(q_des, target)
        self.d.qpos[:], self.d.qvel[:] = q_now, v_now
        self.d.ctrl[:7] = self.servo_command(q_des, q_prev)
        for _ in range(S.SUBSTEPS):
            mujoco.mj_forward(self.m, self.d)
            self.d.qfrc_applied[:7] = self.d.qfrc_bias[:7]
            pos, vel = self.box_pos(), self.d.qvel[self.box_v:self.box_v + 3]
            quat, ang = self.d.qpos[self.box_q + 3:self.box_q + 7], self.d.qvel[self.box_v + 3:self.box_v + 6]
            rot = np.zeros(3)
            mujoco.mju_quat2Vel(rot, quat, 1.0)
            damping = 2 * np.sqrt(HUMAN_STIFFNESS * BOX_MASS)
            spring = HUMAN_STIFFNESS * (human_target - pos) - damping * vel
            spring *= min(1.0, HUMAN_MAX_FORCE_N / (np.linalg.norm(spring) + 1e-9))
            self.d.xfrc_applied[self.box, :3] = human_level * (spring + np.array([0, 0, BOX_MASS * G]))
            self.d.xfrc_applied[self.box, 3:] = human_level * (-HUMAN_ROT_STIFFNESS * rot - 0.02 * ang)
            mujoco.mj_step(self.m, self.d)
        return q_des


def ramp(t, start, fps, rising=True):
    x = np.clip((t - start) / (GRIP_RAMP_S * fps), 0, 1)
    return x if rising else 1 - x


def retract_target(target, start, fps):
    back = start - target
    step = S.RETRACT_SPEED / fps
    return target + (back if np.linalg.norm(back) < step else step * back / np.linalg.norm(back))


def receive(scene, controller, h, gap, mode, renderer=None):
    m, d, fps = scene.m, scene.d, h["fps"]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    start = scene.to_world(h["receiver_raw"][0])
    q_des = scene.place_arm(start)
    d.ctrl[:7], d.ctrl[7] = q_des, S.GRIPPER_OPEN
    box_from_wrist, _ = S.hand_geometry(gap)
    giver = h["giver_raw"]
    scene.reset_box(scene.to_world(giver[0]) + box_from_wrist)
    controller.reset(h)
    controller.offset = gap - np.array([STANDOFF_M / S.SCALE, 0.0])
    target, phase, closed_at, heavy = start.copy(), "approach", None, 0
    grasp_t = detect_t = None
    tug, frames, loads = 0.0, [], []
    for t in range(len(giver)):
        hand = scene.to_world(giver[t])
        d.mocap_pos[scene.hand_mocap] = hand
        level = ramp(t, h["release"], fps, rising=False)
        mujoco.mj_forward(m, d)
        wrist = d.site_xpos[scene.site].copy()
        load = scene.wrist_force()
        loads.append(load)
        if phase == "approach" and t < h["release"]:
            vel, _ = controller.step(h, t, scene.to_human(wrist))
            miss = scene.box_pos() - d.site_xpos[scene.tip]
            approach = d.site_xmat[scene.tip].reshape(3, 3)[:, 2]
            along = miss @ approach
            lateral = miss - along * approach
            if np.linalg.norm(lateral) < ALIGN_TOL_M and abs(along) < CLOSE_TOL_M:
                phase, closed_at, target = "closing", t, wrist.copy()
                d.ctrl[7] = SQUEEZE
            elif np.linalg.norm(miss) < SERVO_RADIUS_M:
                lined_up = np.linalg.norm(lateral) < ALIGN_TOL_M
                step = miss if lined_up else lateral + min(along - STANDOFF_M, 0.0) * approach
                target = S.lead(wrist + step, wrist)
            else:
                target = S.lead(target + S.SCALE * np.array([vel[0], 0.0, vel[1]]) / fps, wrist)
        elif phase == "closing" and t - closed_at >= CLOSE_SETTLE_S * fps:
            if scene.touch() > TOUCH_N and d.qpos[7] + d.qpos[8] > MIN_OPENING_M:
                phase, grasp_t, baseline = "holding", t, load[2]
            elif t - closed_at > CLOSE_TIMEOUT_S * fps:
                phase = "approach"
                d.ctrl[7] = S.GRIPPER_OPEN
        elif phase == "holding":
            heavy = heavy + 1 if load[2] - baseline > WEIGHT_SHARE * BOX_MASS * G else 0
            if mode == "retract_after_grasp" or heavy >= PERSIST_S * fps:
                phase, detect_t = "retract", t
        if phase == "retract":
            target = retract_target(target, start, fps)
        if grasp_t is not None and level > 0.5:
            tug = max(tug, np.linalg.norm(load[[0, 2]] - np.array([0, baseline])))
        q_des = scene.step(q_des, target, level, hand + box_from_wrist)
        if renderer is not None:
            renderer.update_scene(d, camera="side")
            frames.append(renderer.render())
    held = np.linalg.norm(scene.box_pos() - d.site_xpos[scene.tip]) < HELD_DIST
    return dict(
        success=bool(held and grasp_t is not None),
        grasp_minus_contact_s=(grasp_t - h["contact"]) / fps if grasp_t is not None else np.nan,
        weight_felt_after_release_s=(detect_t - h["release"]) / fps if detect_t is not None and mode == "wait_for_weight" else np.nan,
        peak_tug_n=tug,
    ), frames, np.array(loads)


def ready(scene, xy):
    p = scene.to_world(xy)
    p[0] = max(p[0], scene.shoulder[0] + READY_FORWARD_M)
    return p


def give(scene, h, gap, rule, predicted_hold, mean_hold, renderer=None):
    m, d, fps = scene.m, scene.d, h["fps"]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    path = h["giver_raw"]
    target = ready(scene, path[0])
    q_des = scene.place_arm(target)
    d.ctrl[:7], d.ctrl[7] = q_des, SQUEEZE
    mujoco.mj_forward(m, d)
    scene.reset_box(d.site_xpos[scene.tip].copy())
    receiver = h["receiver_raw"]
    d.mocap_pos[scene.hand_mocap] = scene.to_world(receiver[0])
    for _ in range(round(SETTLE_S * fps)):
        q_des = scene.step(q_des, target, 0.0, np.zeros(3))
    full = scene.wrist_force()
    gap_world = np.linalg.norm(S.SCALE * gap)
    seen_contact = released = None
    felt = 0
    grip_at = grip_offset = None
    tug, frames, trace = 0.0, [], []
    for t in range(len(path)):
        hand = scene.to_world(receiver[t])
        d.mocap_pos[scene.hand_mocap] = hand
        if t == h["contact"]:
            grip_at = scene.box_pos()
        if t == h["release"]:
            grip_offset = grip_at - hand
        human_target = hand + grip_offset if grip_offset is not None else grip_at
        level = ramp(t, h["contact"], fps) if grip_at is not None else 0.0
        mujoco.mj_forward(m, d)
        load = scene.wrist_force()
        share = 1 - (full[2] - load[2]) / (BOX_MASS * G)
        pull = -(load[0] - full[0])
        trace.append((share, pull, level))
        wrist = d.site_xpos[scene.site]
        if seen_contact is None and np.linalg.norm(hand - wrist) < 1.1 * gap_world:
            seen_contact = t
        if released is None:
            signal = {"weight_share": share < 1 - WEIGHT_SHARE, "pull": pull > PULL_N}.get(rule, False)
            felt = felt + 1 if signal and seen_contact is not None else 0
            due = {
                "at_contact": seen_contact is not None,
                "mean_hold": seen_contact is not None and t >= seen_contact + mean_hold * fps,
                "predicted_hold": seen_contact is not None and t >= seen_contact + predicted_hold * fps,
                "weight_share": felt >= PERSIST_S * fps,
                "pull": felt >= PERSIST_S * fps,
            }[rule]
            if due:
                released = t
                d.ctrl[7] = S.GRIPPER_OPEN
            elif level > 0.5:
                tug = max(tug, abs(pull))
        target = ready(scene, path[t])
        q_des = scene.step(q_des, target, level, human_target if human_target is not None else np.zeros(3))
        if renderer is not None:
            renderer.update_scene(d, camera="side")
            frames.append(renderer.render())
    with_human = grip_offset is not None and np.linalg.norm(scene.box_pos() - (hand + grip_offset)) < HELD_DIST
    in_gripper = np.linalg.norm(scene.box_pos() - d.site_xpos[scene.tip]) < HELD_DIST
    return dict(
        transferred=bool(with_human and released is not None),
        dropped=bool(not with_human and not in_gripper),
        release_minus_human_s=(released - h["release"]) / fps if released is not None else np.nan,
        released_before_human_grip=bool(released is not None and released < h["contact"] + GRIP_RAMP_S * fps / 2),
        peak_tug_n=tug,
    ), frames, np.array(trace)


def summarise(rows, rates, means):
    lines = [f"{'method':22s}" + "".join(f"{k:>28s}" for k in rates + means)]
    for method in dict.fromkeys(r["method"] for r in rows):
        sel = [r for r in rows if r["method"] == method]
        cells = [f"{np.mean([r[k] for r in sel]):28.0%}" for k in rates]
        cells += [f"{np.nanmean([r[k] for r in sel]):28.2f}" for k in means]
        lines.append(f"{method:22s}" + "".join(cells))
    return "\n".join(lines)


def plot_giving(traces, h, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for rule, trace in traces.items():
        t = (np.arange(len(trace)) - h["contact"]) / h["fps"]
        axes[0].plot(t, trace[:, 0], label=rule)
        axes[1].plot(t, trace[:, 1], label=rule)
    for ax in axes:
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.axvline((h["release"] - h["contact"]) / h["fps"], color="k", ls="--", lw=1)
        ax.set_xlabel("time from human contact (s); dashed = human giver let go in the recording")
        ax.grid()
    axes[0].set_ylabel("share of box weight on the robot")
    axes[1].set_ylabel("pull toward human at robot wrist (N)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Panda giving, {h['take']} card {h['card']}: what the wrist force sensor feels under each release rule")
    fig.tight_layout()
    fig.savefig(out, dpi=80)


def render_receive(args):
    receiving = load(args.data, args.sheet, tail_s=TAIL_S)
    recv_train, recv_test = split(receiving, "main")
    horizon = round(ev.HORIZON_S * receiving[0]["fps"])
    controller = policy.build_controllers(recv_train, args.seeds[0], horizon)["world_model_target"]
    take, card = args.render_receive.split(":")
    h = next(x for x in recv_test if x["take"] == take and x["card"] == int(card))
    gap = policy.grasp_offset([x for x in receiving if x["object"] == h["object"] and x["take"] != h["take"]])
    controller.offset = gap
    scene = Scene(build(args.menagerie, gap))
    renderer = mujoco.Renderer(scene.m, 360, 480)
    result, frames, _ = receive(scene, controller, h, gap, "wait_for_weight", renderer)
    status = f"holds box, grasp {result['grasp_minus_contact_s']:+.2f} s vs human" if result["success"] else "box not held"
    text = f"Panda receiving (world model + touch/weight): {status}"
    Path(args.media).mkdir(exist_ok=True)
    S.save_gif([S.label(f, text) for f in frames], Path(args.media) / f"receive_{take}_card{card}.gif")
    print(text, result)


def render_give(args):
    giving = load(args.data, args.sheet, direction="robot_to_human", tail_s=TAIL_S)
    give_train, give_test = split(giving, "main")
    take, card = args.render_give.split(":")
    i, h = next((i, x) for i, x in enumerate(give_test) if x["take"] == take and x["card"] == int(card))
    model = rel.train(give_train, args.seeds[0])
    X, last = rel.batch(give_test)
    with torch.no_grad():
        predicted = model(X, last).numpy()[i]
    mean_hold = np.mean([rel.hold_time(x) for x in give_train])
    gap = -policy.grasp_offset([x for x in giving if x["object"] == h["object"] and x["take"] != h["take"]])
    scene = Scene(build(args.menagerie, gap))
    renderer = mujoco.Renderer(scene.m, 360, 480)
    result, frames, _ = give(scene, h, gap, args.rule, predicted, mean_hold, renderer)
    status = "handed over" if result["transferred"] else ("dropped" if result["dropped"] else "not handed over")
    text = f"Panda giving, release rule '{args.rule}': {status}"
    Path(args.media).mkdir(exist_ok=True)
    S.save_gif([S.label(f, text) for f in frames], Path(args.media) / f"give_{take}_card{card}.gif")
    print(text, result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--menagerie", default="mujoco_menagerie")
    ap.add_argument("--out", default="results")
    ap.add_argument("--media", default="media")
    ap.add_argument("--seeds", type=int, nargs="+", default=policy.SEEDS)
    ap.add_argument("--gif", default="B3:6", help="robot-to-human handover TAKE:CARD for the force plot")
    ap.add_argument("--render-give", help="only render a giving GIF for this TAKE:CARD")
    ap.add_argument("--render-receive", help="only render a receiving GIF for this TAKE:CARD (wait for weight, first seed)")
    ap.add_argument("--rule", default="pull", choices=GIVE_RULES)
    args = ap.parse_args()
    if args.render_give:
        render_give(args)
        return
    if args.render_receive:
        render_receive(args)
        return

    torch.set_num_threads(4)
    receiving = load(args.data, args.sheet, tail_s=TAIL_S)
    giving = load(args.data, args.sheet, direction="robot_to_human", tail_s=TAIL_S)
    recv_train, recv_test = split(receiving, "main")
    give_train, give_test = split(giving, "main")
    horizon = round(ev.HORIZON_S * receiving[0]["fps"])
    Path(args.out).mkdir(exist_ok=True)

    recv_rows = []
    for seed in args.seeds:
        controllers = policy.build_controllers(recv_train, seed, horizon)
        for h in recv_test:
            gap = policy.grasp_offset([x for x in receiving if x["object"] == h["object"] and x["take"] != h["take"]])
            scene = Scene(build(args.menagerie, gap))
            for name in ["world_model_target", "reactive"]:
                controllers[name].offset = gap
                modes = RECEIVE_MODES if name == "world_model_target" else ["wait_for_weight"]
                for mode in modes:
                    result, _, _ = receive(scene, controllers[name], h, gap, mode)
                    label = mode if name == "world_model_target" else f"reactive, {mode}"
                    recv_rows.append(dict(take=h["take"], card=h["card"], seed=seed, method=label, **result))
    ev.write_csv(recv_rows, Path(args.out) / "force_receiving.csv")
    print(f"Panda receiving with touch and weight sensing (world model + controller unless marked), {len(recv_test)} test handovers "
          f"x {len(args.seeds)} seeds")
    print(summarise(recv_rows, ["success"], ["grasp_minus_contact_s", "weight_felt_after_release_s", "peak_tug_n"]))

    mean_hold = np.mean([rel.hold_time(h) for h in give_train])
    X, last = rel.batch(give_test)
    predicted = {}
    for seed in args.seeds:
        model = rel.train(give_train, seed)
        with torch.no_grad():
            predicted[seed] = model(X, last).numpy()
    give_rows, traces = [], {}
    gif_take, gif_card = args.gif.split(":")
    for i, h in enumerate(give_test):
        gap = -policy.grasp_offset([x for x in giving if x["object"] == h["object"] and x["take"] != h["take"]])
        scene = Scene(build(args.menagerie, gap))
        for rule in GIVE_RULES:
            seeds = args.seeds if rule == "predicted_hold" else [""]
            for seed in seeds:
                hold = predicted[seed][i] if seed != "" else mean_hold
                result, _, trace = give(scene, h, gap, rule, hold, mean_hold)
                give_rows.append(dict(take=h["take"], card=h["card"], seed=seed, method=rule, **result))
                if h["take"] == gif_take and h["card"] == int(gif_card) and seed in ("", args.seeds[0]):
                    traces[rule] = trace
    ev.write_csv(give_rows, Path(args.out) / "force_giving.csv")
    print(f"\nPanda giving (retargeted robot-role giver path), {len(give_test)} test handovers; "
          f"predicted_hold over {len(args.seeds)} seeds")
    print(summarise(give_rows, ["transferred", "dropped", "released_before_human_grip"],
                    ["release_minus_human_s", "peak_tug_n"]))
    h = next(x for x in give_test if x["take"] == gif_take and x["card"] == int(gif_card))
    plot_giving(traces, h, Path(args.out) / "force_giving_example.png")


if __name__ == "__main__":
    main()
