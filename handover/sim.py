import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from handover import evaluate as ev
from handover import policy
from handover.dataset import load, split

PANDA_REACH_M = 0.855
HUMAN_ARM_M = 0.52
SCALE = 0.85 * PANDA_REACH_M / HUMAN_ARM_M
BASE_Z = 0.6
SUBSTEPS = 16
IK_DAMPING = 0.1
MAX_JOINT_STEP = 0.05
ORIENTATION_WEIGHT = 0.3
RETRACT_SPEED = 0.4
FINGER_REACH_M = 0.1034
BOX_HALF = [0.04, 0.03, 0.04]
GRIPPER_OPEN, GRIPPER_CLOSED = 255, round(255 * BOX_HALF[1] / 0.04)
GIF_METHODS = ["reactive", "world_model_target", "learned_policy", "human_path"]
TARGET_R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)


def hand_geometry(offset):
    reach = SCALE * np.array([offset[0], 0.0, offset[1]])
    direction = reach / np.linalg.norm(reach)
    return reach - FINGER_REACH_M * direction, direction


def build_model(menagerie, offset):
    spec = mujoco.MjSpec.from_file(str(Path(menagerie) / "franka_emika_panda" / "panda.xml"))
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[1, 1, 1], rgb2=[0.75, 0.8, 0.88], width=256, height=256)
    spec.body("link0").pos = [0, 0, BASE_Z]
    world = spec.worldbody
    world.add_light(pos=[0.5, -1.5, 2.5], dir=[0, 0.5, -1])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.85, 0.85, 0.85, 1])
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.09, BASE_Z / 2, 0], pos=[0, 0, BASE_Z / 2],
                   rgba=[0.3, 0.3, 0.3, 1], contype=0, conaffinity=0)
    world.add_camera(name="side", pos=[0.65, -2.0, 0.95], xyaxes=[1, 0, 0, 0, 0, 1], fovy=45)
    spec.body("hand").add_site(name="wrist", size=[0.01, 0, 0], rgba=[1, 0, 0, 0])
    spec.body("hand").add_site(name="fingertips", pos=[0, 0, FINGER_REACH_M], size=[0.01, 0, 0], rgba=[1, 0, 0, 0])
    box_from_wrist, direction = hand_geometry(offset)
    hand = world.add_body(name="human_hand", mocap=True)
    hand.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.025, 0, 0], rgba=[0.85, 0.6, 0.45, 1],
                  fromto=[0, 0, 0, *(box_from_wrist - BOX_HALF[0] * direction)], contype=0, conaffinity=0)
    box = world.add_body(name="object", mocap=True)
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=BOX_HALF, rgba=[0.2, 0.45, 0.9, 1], contype=0, conaffinity=0)
    return spec.compile()


class Sim:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.m.opt.timestep = 1 / (30 * SUBSTEPS)
        self.site = model.site("wrist").id
        self.tip = model.site("fingertips").id
        self.hand_mocap = model.body("human_hand").mocapid[0]
        self.box_mocap = model.body("object").mocapid[0]
        self.home = model.key("home").qpos.copy()
        mujoco.mj_resetDataKeyframe(model, self.d, 0)
        mujoco.mj_forward(model, self.d)
        self.shoulder = self.d.xpos[model.body("link2").id].copy()

    def to_world(self, xy):
        return self.shoulder + SCALE * np.array([xy[0], 0.0, xy[1]])

    def to_human(self, p):
        return (p - self.shoulder)[[0, 2]] / SCALE

    def ik_step(self, q, target):
        self.d.qpos[:7] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        rot_err = 0.5 * (np.cross(R[:, 0], TARGET_R[:, 0]) + np.cross(R[:, 1], TARGET_R[:, 1])
                         + np.cross(R[:, 2], TARGET_R[:, 2]))
        err = np.r_[target - self.d.site_xpos[self.site], ORIENTATION_WEIGHT * rot_err]
        J = np.vstack([jp, ORIENTATION_WEIGHT * jr])[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), err)
        dq = dq + 0.1 * (np.eye(7) - np.linalg.pinv(J) @ J) @ (self.home[:7] - q)
        return q + dq * min(1.0, MAX_JOINT_STEP / np.abs(dq).max())

    def place_arm(self, target, iterations=300):
        q = self.home[:7].copy()
        for _ in range(iterations):
            q = self.ik_step(q, target)
        self.d.qpos[:7] = q
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)
        return q


def run(sim, controller, h, offset, renderer=None):
    m, d = sim.m, sim.d
    mujoco.mj_resetDataKeyframe(m, d, 0)
    start = sim.to_world(h["receiver_raw"][0])
    q_des = sim.place_arm(start)
    d.ctrl[:7], d.ctrl[7] = q_des, GRIPPER_OPEN
    controller.reset(h)
    target = start.copy()
    giver = h["giver_raw"]
    box_from_wrist, _ = hand_geometry(offset)
    held_at, frames, site_path = None, [], []
    for t in range(len(giver)):
        d.mocap_pos[sim.hand_mocap] = sim.to_world(giver[t])
        mujoco.mj_forward(m, d)
        site = d.site_xpos[sim.site].copy()
        site_path.append(site)
        pos = sim.to_human(site)
        if held_at is None:
            vel, grasp = controller.step(h, t, pos)
            box = sim.to_world(giver[t]) + box_from_wrist
            reach = np.linalg.norm(site - sim.to_world(giver[t] + offset))
            if grasp and reach < SCALE * policy.GRASP_DIST:
                held_at = t
                held = box - d.site_xpos[sim.tip]
                d.ctrl[7] = GRIPPER_CLOSED
            target = target + SCALE * np.array([vel[0], 0.0, vel[1]]) / h["fps"]
        else:
            box = d.site_xpos[sim.tip] + held
            back = start - target
            step = RETRACT_SPEED / h["fps"]
            target = target + (back if np.linalg.norm(back) < step else step * back / np.linalg.norm(back))
        d.mocap_pos[sim.box_mocap] = box
        q_now, v_now = d.qpos.copy(), d.qvel.copy()
        q_des = sim.ik_step(q_des, target)
        d.qpos[:], d.qvel[:] = q_now, v_now
        d.ctrl[:7] = q_des
        for _ in range(SUBSTEPS):
            mujoco.mj_forward(m, d)
            d.qfrc_applied[:7] = d.qfrc_bias[:7]
            mujoco.mj_step(m, d)
        if renderer is not None:
            renderer.update_scene(d, camera="side")
            frames.append(renderer.render())
    path = np.array(site_path)
    jerk = np.diff(path, n=3, axis=0) * h["fps"] ** 3
    result = dict(
        success=held_at is not None,
        grasp_minus_contact_s=(held_at - h["contact"]) / h["fps"] if held_at is not None else np.nan,
        path_length_m=np.linalg.norm(np.diff(path[:held_at or None], axis=0), axis=1).sum(),
        rms_jerk=np.sqrt((np.linalg.norm(jerk[:held_at or None], axis=1) ** 2).mean()),
    )
    return result, frames


def label(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 18], fill=(255, 255, 255))
    draw.text((6, 3), text, fill=(0, 0, 0))
    return np.asarray(img)


def save_gif(grid_frames, path, every=2):
    images = [Image.fromarray(f) for f in grid_frames[::every]]
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000 * every // 30, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--sheet", default="data/take_sheet.csv")
    ap.add_argument("--menagerie", default="mujoco_menagerie")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, nargs="+", default=policy.SEEDS)
    ap.add_argument("--gif", nargs="*", default=[], help="TAKE:CARD handovers to render, e.g. B3:5")
    ap.add_argument("--media", default="media")
    args = ap.parse_args()

    handovers = load(args.data, args.sheet)
    train_set, test_set = split(handovers, "main")
    horizon = round(ev.HORIZON_S * handovers[0]["fps"])
    offset = policy.grasp_offset(train_set)

    def human_path(h, t):
        return h["receiver"][min(t + 3, len(h["receiver"]) - 1)] - offset

    sim = Sim(build_model(args.menagerie, offset))
    rows, first = [], None
    for seed in args.seeds:
        controllers = policy.build_controllers(train_set, seed, horizon)
        controllers["human_path"] = policy.Chase(human_path, offset, 8)
        first = first or controllers
        for h in test_set:
            for name, controller in controllers.items():
                result, _ = run(sim, controller, h, offset)
                rows.append(dict(take=h["take"], card=h["card"], method=name, seed=seed, **result))
    Path(args.out).mkdir(exist_ok=True)
    ev.write_csv(rows, Path(args.out) / "sim.csv")

    print(f"Panda receiving from replayed human hands: {len(test_set)} test handovers (A3, B3), "
          f"seeds {args.seeds}")
    print(f"{'method':24s}{'success':>9s}{'':17s}{'grasp - contact s':>19s}{'path length m':>15s}{'rms jerk m/s3':>15s}")
    for name in first:
        sel = [r for r in rows if r["method"] == name]
        ok = [r for r in sel if r["success"]]
        per_seed = [np.mean([r["success"] for r in sel if r["seed"] == s]) for s in args.seeds]
        print(f"{name:24s}{len(ok) / len(sel):9.0%} (seeds {min(per_seed):.0%}-{max(per_seed):.0%})"
              f"{np.mean([r['grasp_minus_contact_s'] for r in ok]) if ok else np.nan:+19.2f}"
              f"{np.mean([r['path_length_m'] for r in sel]):15.2f}{np.mean([r['rms_jerk'] for r in sel]):15.1f}")

    if args.gif:
        Path(args.media).mkdir(exist_ok=True)
        renderer = mujoco.Renderer(sim.m, 240, 320)
        for spec in args.gif:
            take, card = spec.split(":")
            h = next(x for x in test_set if x["take"] == take and x["card"] == int(card))
            panels = []
            for name in GIF_METHODS:
                result, frames = run(sim, first[name], h, offset, renderer)
                status = f"grasp {result['grasp_minus_contact_s']:+.2f} s vs human" if result["success"] else "no grasp"
                panels.append([label(f, f"{name}: {status}") for f in frames])
            grid = [np.vstack([np.hstack(p[:2]), np.hstack(p[2:])]) for p in zip(*panels)]
            save_gif(grid, Path(args.media) / f"sim_{take}_card{card}.gif")


if __name__ == "__main__":
    main()
