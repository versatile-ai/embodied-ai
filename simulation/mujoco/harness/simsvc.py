#!/usr/bin/env python3
"""MuJoCo simulation service for the RoboDojo replication (put_bottles_into_dustbin first).

Implements the segment-boundary protocol in docs/sim_protocol.md:
  POST /session            {"layout": <path or inline dict>, "instruction": str}
                           -> {"session_id": ...}
  GET  /session/<id>/observe -> {"images": {cam: b64}, "shapes": {...}, "state": [14],
                                 "t": int, "done": bool, "score": float}
  POST /session/<id>/act   {"joints": [[14 floats] x K]}   (K <= remaining steps)
                           -> {"t": int, "done": bool, "score": float}
  GET  /session/<id>/result  -> {"score": float, "success": bool, "steps": int,
                                 "bottles_in": int}

Physics 1000Hz, control 25Hz: 40 substeps, interpolation over 32 substeps.
These are local MuJoCo stability settings, not an official-equivalence claim.
Scoring mirrors RoboDojo put_bottles_into_dustbin: transition scores
[10, 25, 40, 100] for [1,2,3,4] bottles in the dustbin (each tier also requires
all grippers open); success = 4 bottles in + arms back at origin.
Objects are primitive approximations (cylinder bottles, open-top bin shell);
real meshes are a later fidelity step.
"""
import base64
import json
import os
import threading
import uuid
import time
import hashlib
from pathlib import Path
from urllib.parse import urlsplit
from PIL import Image
from control import solve_step, validate_goals
from video import encode_video
from http.server import BaseHTTPRequestHandler, HTTPServer

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(os.environ.get("ASTRA_RUN", str(ROOT / "runs")))
RUN_ROOT.mkdir(parents=True, exist_ok=True)
SCENE_XML = os.environ.get("SIMSCENE", str(ROOT / "assets/x5/dual_x5_scene.xml"))
PORT = int(os.environ.get("SIMPORT", "8763"))
CAMERA_PROFILE = os.environ.get("SIM_CAMERA_PROFILE", "wide").lower()
HOST = os.environ.get("SIMHOST", "127.0.0.1")

CTRL_HZ = 25
PHYS_HZ = 1000
SUBSTEPS = PHYS_HZ // CTRL_HZ
INTERP_SUBSTEPS = SUBSTEPS * 4 // 5
STEP_LIM = 700                         # control steps (RoboDojo put_bottles)
TABLE_Z = 0.765
# Smooth the low-level target stream.  The policy runs at 25 Hz and may move
# the IK waypoint by several centimetres between calls; sending that target
# directly to a high-gain position actuator creates stop-go motion and visible
# joint chatter (especially in the wrist camera).  These limits are per
# control tick, in physical joint units.
ARM_CTRL_STEP = float(os.environ.get("SIM_ARM_CTRL_STEP", "0.028"))
GRIP_CTRL_STEP = float(os.environ.get("SIM_GRIP_CTRL_STEP", "0.008"))
CTRL_SMOOTH = float(os.environ.get("SIM_CTRL_SMOOTH", "0.85"))
ARM_KP = float(os.environ.get("SIM_ARM_KP", "2400"))
ARM_KD_FLOOR = float(os.environ.get("SIM_ARM_KD_FLOOR", "8"))
# RoboDojo X5 gripper joint7 range is [-0.01, 0.044] m.  Keep normalized
# policy values 0..1 aligned with the official affine scale.
GRIP_MIN, GRIP_MAX = -0.01, 0.044
# Physical contact is the default. Legacy assisted attachment is opt-in only;
# never enable it for physical-grasp acceptance or policy capability reports.
GRASP_ASSIST = os.environ.get("SIM_GRASP_ASSIST", "0").lower() not in ("0", "false", "no")

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
QPOS_IDS, CTRL_IDS = [], []

BIN_R_IN, BIN_H, BIN_WALL = 0.16, 0.30, 0.012
BOTTLE_R, BOTTLE_H = 0.032, 0.22
Z_GAP_MAX = 0.4
ORIGIN_TOL = 0.12                      # rad, arms back at home
GRIP_OPEN_TH = 0.8


def grip_norm(j7):
    return float(np.clip((j7 - GRIP_MIN) / (GRIP_MAX - GRIP_MIN), 0.0, 1.0))


def grip_denorm(g):
    return GRIP_MIN + float(np.clip(g, 0.0, 1.0)) * (GRIP_MAX - GRIP_MIN)


class Session:
    def __init__(self, layout, instruction, task="put_bottles", record_dir=None, overview_yaw=None):
        if task not in ("put_bottles", "classify_objects"):
            raise ValueError("Unsupported task")
        if overview_yaw is not None:
            overview_yaw = float(overview_yaw)
            if not np.isfinite(overview_yaw) or abs(overview_yaw) > 180:
                raise ValueError('overview_yaw must be finite and within +/-180 degrees')
        self.overview_yaw = overview_yaw
        self.record_dir = Path(record_dir) if record_dir else None
        if self.record_dir: self.record_dir.mkdir(parents=True, exist_ok=False)
        self.frame_cache = {}
        self.closed = False
        self.requests = {}
        self.success = False
        self.error = None
        self.grasped = {"left": None, "right": None}
        self.grasp_opening = {}
        self.commanded_grip = {"left": 0.0, "right": 0.0}
        self.grip_target_norm = {"left": 0.0, "right": 0.0}
        self.step_limit = 1100 if task == "classify_objects" else 700
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.model.opt.timestep = 1.0 / PHYS_HZ
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qpos_ids = [m.joint(f"left_{j}").qposadr for j in ARM_JOINTS] + \
                        [m.joint("left_joint7").qposadr] + \
                        [m.joint(f"right_{j}").qposadr for j in ARM_JOINTS] + \
                        [m.joint("right_joint7").qposadr]
        self.ctrl_ids = list(range(m.nu))  # 14 actuators in same order
        self.cam_ids = [m.camera(c).id for c in
                        ("cam_base", "left_cam_wrist", "right_cam_wrist")]
        self.task = task
        self._add_objects(layout)
        mujoco.mj_forward(self.model, self.data)
        self.grip_target_norm["left"] = grip_norm(self.data.qpos[self.qpos_ids[6]])
        self.grip_target_norm["right"] = grip_norm(self.data.qpos[self.qpos_ids[13]])
        self.commanded_grip = self.grip_target_norm.copy()
        self.q_home = self.data.qpos.copy()
        self.t = 0
        self.score = 0.0
        self.done = False
        self.instruction = instruction
        self.layout = layout
        self.lock = threading.RLock()
        self.ctrl_ids = [self.model.actuator(side + j).id for side in ("left_", "right_") for j in ARM_JOINTS + ["gripper"]]
        self.qpos_ids = [int(np.asarray(i).item()) for i in self.qpos_ids]
        self.data.ctrl[self.ctrl_ids] = self.data.qpos[self.qpos_ids]
        # Last physically applied target.  Keep this separate from the policy
        # target so consecutive action rows cannot introduce a discontinuous
        # command into the actuator interpolation below.
        self.command_ctrl = self.data.ctrl[self.ctrl_ids].copy()
        self.reset_check = self.verify_initial_state()
        if self.record_dir:
            (self.record_dir / "reset_check.json").write_text(json.dumps(self.reset_check, indent=2))
        if self.record_dir:
            (self.record_dir / "initialization.json").write_text(json.dumps({"layout": layout, "instruction": instruction, "task": task, "overview_yaw": self.overview_yaw}, indent=2))
        self.capture()

    def verify_initial_state(self):
        """A new episode must start from its own model/layout, never prior data."""
        checks = {
            "step_zero": self.t == 0 and self.data.time == 0,
            "joints_home": bool(np.allclose(self.data.qpos[self.qpos_ids], self.model.qpos0[self.qpos_ids], atol=1e-9, rtol=0)),
            "velocities_zero": bool(np.all(self.data.qvel == 0)),
            "controls_match_joints": bool(np.allclose(self.command_ctrl, self.data.qpos[self.qpos_ids], atol=1e-9, rtol=0)),
            "objects_at_layout": all(np.allclose(self.data.xpos[self.model.body(name).id], pos, atol=1e-9, rtol=0)
                                     and abs(float(np.dot(self.data.xquat[self.model.body(name).id], np.asarray(quat)/np.linalg.norm(quat)))) > 1-1e-8
                                     for name,(pos,quat) in self.obj_poses.items()),
            "no_grasp": all(v is None for v in self.grasped.values()),
            "no_weld": not any(self.data.eq_active[i] for i in range(self.model.neq)
                               if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD),
            "clean_episode": not self.requests and not self.done and not self.success and self.score == 0 and self.error is None,
        }
        if not all(checks.values()):
            raise RuntimeError("Initial reset verification failed: " + str(checks))
        return {"passed": True, "checks": checks, "initial_state14": self.state14()}

    # ---------- scene ----------
    def _add_objects(self, layout):
        """Append primitive bodies at layout poses. Spec keeps body ids."""
        m = self.model
        spec = mujoco.MjSpec.from_file(SCENE_XML)
        # rebuild via spec so we can add bodies, then recompile with renderer
        rigids = layout.get("Rigid", {})
        geoms = layout.get("Geometry", {})
        self.obj_bodies = {}
        self.obj_poses = {}
        world = spec.worldbody
        # table geometry follows the layout (height = pos_z + scale_z/2 = 0.765)
        tcfg = layout.get("Table", {})
        if tcfg:
            tbl = next(g for g in world.geoms if g.name == "table")
            tbl.pos = tcfg["default_pos"]
            tbl.size = [tcfg["scale"][0] / 2, tcfg["scale"][1] / 2, tcfg["scale"][2] / 2]
        # lighting: MuJoCo offscreen renders black without explicit lights.
        # Room as 5 inward-facing planes (box interiors are backface-culled):
        # four walls + ceiling; the floor plane already exists in the scene.
        for nm, pos, quat in [
            ("room_wall_n", [0.0, 3.0, 1.4], (0.7071, 0.7071, 0.0, 0.0)),
            ("room_wall_s", [0.0, -3.0, 1.4], (0.7071, -0.7071, 0.0, 0.0)),
            ("room_wall_e", [3.0, 0.0, 1.4], (0.7071, 0.0, -0.7071, 0.0)),
            ("room_wall_w", [-3.0, 0.0, 1.4], (0.7071, 0.0, 0.7071, 0.0)),
            ("room_ceil", [0.0, 0.0, 2.8], (0.0, 1.0, 0.0, 0.0)),
        ]:
            w = world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, pos=pos, quat=quat)
            w.name = nm
            w.contype = 0
            w.conaffinity = 0
            w.rgba = [0.72, 0.73, 0.76, 1.0]
            w.size = [4.0, 4.0, 1.0]
        # official-look palette: dark wood floor, off-white walls
        for g in world.geoms:
            if g.name == "floor":
                g.rgba = [0.24, 0.15, 0.09, 1.0]
        for nm in ("room_wall_n", "room_wall_s", "room_wall_e",
                   "room_wall_w", "room_ceil"):
            for g in world.geoms:
                if g.name == nm:
                    g.rgba = [0.88, 0.87, 0.85, 1.0]
        l1 = world.add_light()
        l1.pos = [0.0, -0.1, 1.9]
        l1.dir = [0.0, 0.15, -1.0]
        l1.diffuse = [1.4, 1.4, 1.4]
        l1.specular = [0.3, 0.3, 0.3]
        l1.castshadow = True
        l2 = world.add_light()
        l2.pos = [0.0, -1.4, 1.5]
        l2.dir = [0.0, 0.7, -0.7]
        l2.diffuse = [0.9, 0.9, 0.9]
        l2.specular = [0.2, 0.2, 0.2]
        entries = [(e.get("label", f"bottle{i}"), e)
                   for i, e in enumerate(rigids.get("bottle", []))]
        entries += [(e.get("label", "dustbin"), e)
                    for e in geoms.get("dustbin", [])]
        # The official camera stand is decorative but part of the visual
        # scene.  Keep it static and non-colliding so it cannot affect scores.
        for stand in geoms.get("camera_stand", []):
            sb = world.add_body(); sb.name = stand.get("label", "camera_stand")
            sb.pos = stand.get("default_pos", [0.0, -0.47, 0.765])
            sb.quat = stand.get("default_ori", [0.707, -0.707, 0.0, 0.0])
            for size, pos in [([0.025, 0.025, 0.20], [0, 0, 0.20]),
                              ([0.12, 0.025, 0.025], [0, 0, 0.40])]:
                sg = sb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos)
                sg.contype = 0; sg.conaffinity = 0
                sg.rgba = [0.12, 0.12, 0.14, 1.0]
            self.obj_bodies[sb.name] = sb
            self.obj_poses[sb.name] = (list(sb.pos), list(sb.quat))
        for label, entries in entries:
            pos = list(entries["default_pos"])
            quat = entries["default_ori"]  # wxyz
            scale = entries.get("scale", [1, 1, 1])
            is_bin = "dustbin" in label
            b = world.add_body()
            b.name = label
            b.quat = quat
            if is_bin:
                # The task layout defines the dustbin as a Geometry fixture on
                # the ground.  It must not acquire a free joint: gravity and
                # object contacts otherwise move the scoring volume between
                # observations.  Its local origin is the bin bottom.
                pos[2] = 0.0
                if list(scale) == [1.0, 1.0, 1.0]:   # native-mesh scale: use measured dims
                    hx, hy, hz = 0.15, 0.13, 0.40
                else:
                    hx, hy, hz = scale[0] / 2, scale[1] / 2, scale[2] / 2
                mass = entries.get("physics", {}).get("mass", 0.85)
                for sx, sy, sz, px, py, pz in [
                    (hx + BIN_WALL, hy + BIN_WALL, 0.01, 0, 0, 0.01),
                    (hx, BIN_WALL, hz - 0.05, 0, hy + BIN_WALL, hz),
                    (hx, BIN_WALL, hz - 0.05, 0, -(hy + BIN_WALL), hz),
                    (BIN_WALL, hy, hz - 0.05, hx + BIN_WALL, 0, hz),
                    (BIN_WALL, hy, hz - 0.05, -(hx + BIN_WALL), 0, hz),
                ]:
                    g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                   size=[sx, sy, sz], pos=[px, py, pz])
                    g.mass = mass / 5.0
                    g.friction = [1.0, 0.005, 0.0005]
                    g.rgba = [0.25, 0.3, 0.35, 1.0]
                self.bin_inner = (hx, hy)
            else:
                b.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
                mass = entries.get("physics", {}).get("mass", 0.15)
                kind = "bottle" if label.startswith("bottle") else "dustbin"
                mesh_file, bbox = self._find_mesh(kind, entries)
                if mesh_file:
                    mn = f"mesh_{label}"
                    ms = spec.add_mesh(name=mn, file=mesh_file)
                    gv = b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH)
                    gv.meshname = mn
                    gv.mass = mass
                    gv.contype = 1
                    gv.conaffinity = 1
                    gv.friction = [1.0, 0.005, 0.0005]
                    gv.condim = 4
                    gv.solref = [0.002, 1]
                    gv.solimp = [0.99, 0.999, 0.001, 0.5, 2]
                    gv.rgba = [0.85, 0.85, 0.88, 1.0]
                    lo = np.array(bbox["min"]); hi = np.array(bbox["max"])
                    # Bottle contact uses the visible mesh convex hull instead
                    # of an invisible axis-aligned bounding box.
                    gv.rgba = list(self.BOTTLE_COLOR.get(
                        entries.get("category_idx", 0), [0.85, 0.85, 0.88])) + [1.0]
                else:
                    if list(scale) == [1.0, 1.0, 1.0]:
                        r, h = 0.032, 0.186
                    else:
                        r, h = max(0.02, scale[0] / 2), max(4 * (scale[0] / 2), scale[2])
                    pos[2] += 0.002
                    g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                   size=[r, h / 2, 0], pos=[0, 0, 0])
                    g.mass = mass
                    g.friction = [1.0, 0.005, 0.0005]
                    g.rgba = [0.7, 0.2, 0.2, 1.0]
            b.pos = pos
            self.obj_bodies[label] = b
            self.obj_poses[label] = (pos, list(quat))
        if self.task == "classify_objects":
            self._build_classify(layout, world, spec)
        # URDF meshes self-penetrate ~2mm between adjacent arm links in the
        # home pose; exclude those pairs from contact generation
        for sd in ("left_", "right_"):
            chain = [sd + "base_link"] + [sd + f"link{i}" for i in range(1, 9)]
            for a, b in zip(chain, chain[1:]):
                ex = spec.add_exclude()
                ex.bodyname1 = a
                ex.bodyname2 = b
        # Contact-triggered assisted grasp, solved by MuJoCo rather than
        # teleporting a bottle and zeroing its velocity after every step.
        for side in ("left", "right"):
            for label in self.obj_poses:
                if not label.startswith("bottle"):
                    continue
                eq = spec.add_equality()
                eq.name = f"grasp_{side}_{label}"
                eq.type = mujoco.mjtEq.mjEQ_WELD
                eq.objtype = mujoco.mjtObj.mjOBJ_BODY
                eq.name1, eq.name2 = side+"_link6", label
                eq.active = False
                eq.solref = [0.008, 1]
                eq.solimp = [0.99, 0.999, 0.001, 0.5, 2]
        # Calibrate cameras using the small robot-only model. Task meshes do
        # not change robot kinematics; compiling them twice doubles peak RAM.
        m2 = self.model
        mujoco.mj_forward(self.model, self.data)
        # wrist cams: remount relative to the EE site so the view matches the
        # official demo wrist frames (above/behind gripper, looking down-forward
        # at the table in front of the fingers)
        wrist_locals = {}
        for cname, eename in (("left_cam_wrist", "left_ee"),
                              ("right_cam_wrist", "right_ee")):
            cid = int(m2.camera(cname).id)
            bid = int(m2.body("left_camera" if cname.startswith("left")
                              else "right_camera").id)
            # official mount = URDF camera link position (kept); only the
            # orientation is fitted: ~62 deg below horizontal, looking forward
            # over the gripper, matching the official demo wrist frames
            # empirical mount reproducing the official demo wrist framing:
            # 15cm above / 10cm behind the EE, optical axis on the table point
            # in front of the gripper (fingertips enter frame at bottom)
            bpos = self.data.xpos[bid].copy()
            bx = self.data.xmat[bid].reshape(3, 3)
            ee = self.data.site_xpos[int(m2.site(eename).id)].copy()
            if CAMERA_PROFILE == "official":
                cam_world = ee + np.array([0.0, 0.04, 0.20])
                target = np.array([ee[0], ee[1] + 0.30, TABLE_Z])
            else:
                # Wide profile: aim just below the jaw centre rather than a
                # distant table point. Convert this mount to body-local pose
                # once so it follows the wrist without auto-tracking.
                cam_world = ee + np.array([0.0, -0.04, 0.18])
                target = ee + np.array([0.0, 0.025, -0.025])
            f = target - cam_world
            f /= np.linalg.norm(f)
            up0 = np.array([0.0, 0.0, 1.0])
            z = -f
            x = np.cross(up0, z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            qw = np.zeros(4)
            mujoco.mju_mat2Quat(qw, np.stack([x, y, z], axis=1).ravel())
            qbn = np.zeros(4)
            mujoco.mju_negQuat(qbn, self.data.xquat[bid])
            ql = np.zeros(4)
            mujoco.mju_mulQuat(ql, qbn, qw)
            wrist_locals[cname] = (bx.T @ (cam_world - bpos), ql)
        # apply camera edits at SPEC level and recompile so they survive
        # mj_forward (post-compile cam_pos writes are overwritten each step)
        for cname, (lp, lq) in wrist_locals.items():
            for sc in spec.cameras:
                if sc.name == cname:
                    if lp is not None:
                        sc.pos = list(map(float, lp))
                    sc.quat = list(map(float, lq))
                    if CAMERA_PROFILE != "official":
                        sc.fovy = 70.0
        for sc in spec.cameras:
            if sc.name == "cam_base":
                sc.pos = ([0.0, -0.41, 1.308] if CAMERA_PROFILE == "official"
                          else [0.0, -0.72, 1.45])
                hq = np.zeros(4)
                hdeg = 30.0 if CAMERA_PROFILE == "official" else 38.0
                mujoco.mju_euler2Quat(hq, np.array([hdeg, 0.0, 0.0]) * np.pi / 180.0, "xyz")
                sc.quat = list(map(float, hq))
                sc.fovy = 71.1 if CAMERA_PROFILE == "official" else 78.0
                if self.overview_yaw is not None:
                    # Explicit visual-demo override only; default policy camera is unchanged.
                    yaw = np.deg2rad(self.overview_yaw)
                    target = np.array([-0.3, 0.0, 0.65])
                    offset = np.array([1.8*np.sin(yaw), -1.8*np.cos(yaw), 1.4])
                    z = offset / np.linalg.norm(offset)
                    x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
                    y = np.cross(z, x)
                    mujoco.mju_mat2Quat(hq, np.stack([x, y, z], axis=1).ravel())
                    sc.pos = list(map(float, target + offset))
                    sc.quat = list(map(float, hq))
        m2 = spec.compile()
        m2.opt.timestep = 1.0 / PHYS_HZ
        self.model = m2
        self.data = mujoco.MjData(m2)
        mujoco.mj_resetData(m2, self.data)
        for label, (pos, quat) in self.obj_poses.items():
            body = m2.body(label)
            if int(body.jntnum[0]) == 0:
                continue
            qa = int(m2.jnt_qposadr[body.jntadr[0]])
            self.data.qpos[qa:qa + 3] = pos
            self.data.qpos[qa + 3:qa + 7] = quat
        mujoco.mj_forward(m2, self.data)
        # lighter arm material (official arms are white/black, not pure black)
        for i in range(m2.nbody):
            bname = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if bname.startswith(("left_link", "right_link", "left_base", "right_base")):
                b = m2.body(i)
                for gid in range(int(b.geomnum[0])):
                    geom = m2.geom(int(b.geomadr[0]) + gid)
                    geom.rgba = [0.82, 0.83, 0.86, 1.0]
                    geom.contype = 1
                    geom.conaffinity = 1
                    if bname.endswith(("_link7", "_link8")):
                        geom.friction = [1.0, 0.005, 0.0005]
                        geom.condim = 4
                        geom.solref = [0.002, 1]
                        geom.solimp = [0.99, 0.999, 0.001, 0.5, 2]
        # Local arm servo tuning; not a copy of Isaac drive parameters.
        m2.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        m2.opt.iterations = 50
        m2.opt.tolerance = 1e-8
        for i in range(m2.nu):
            aname = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if "gripper" not in aname and "follower" not in aname:
                # Keep the same bounded, critically damped position drive on
                # every arm joint.  The old mass-based cap reduced distal
                # joints to single-digit kp, allowing pose drift and contact
                # impulses during a 40 ms control interval.
                kp = ARM_KP
                dof = int(m2.joint(aname).dofadr[0])
                m0 = float(m2.dof_M0[dof])
                # The previous fixed 80 damping floor was 3–60x the
                # critical value for this X5's 0.00046–0.195 kg m² joint
                # inertias.  It made short measured-state EEF segments move
                # only millimetres and consumed the task step budget before a
                # grasp could be carried.  Use critical damping per joint;
                # retain a small floor for the light wrist joints.
                kd = max(ARM_KD_FLOOR, 2.0 * np.sqrt(kp * max(m0, 1e-4)))
                m2.actuator_gainprm[i][0] = kp
                m2.actuator_biasprm[i][0] = float(self.data.qfrc_bias[dof])
                m2.actuator_biasprm[i][1] = -kp
                m2.actuator_biasprm[i][2] = -kd

        m = self.model
        self.qpos_ids = [m.joint(f"left_{j}").qposadr for j in ARM_JOINTS] + \
                        [m.joint("left_joint7").qposadr] + \
                        [m.joint(f"right_{j}").qposadr for j in ARM_JOINTS] + \
                        [m.joint("right_joint7").qposadr]
        self.cam_ids = [m.camera(c).id for c in
                        ("cam_base", "left_cam_wrist", "right_cam_wrist")]
        mujoco.mj_forward(m, self.data)  # refresh data-side camera frames
        self.renderer = mujoco.Renderer(m, 480, 640)
        try:
            self.renderer.scene.flags |= int(mujoco.mjtVisFlag.mjVIS_HEADLIGHT)
        except Exception:  # noqa: BLE001
            pass
        self.obj_body_ids = {label: m.body(label).id for label in self.obj_bodies}
        # Do not retain MjsBody handles (and their owning mesh specification).
        self.obj_bodies = dict.fromkeys(self.obj_body_ids)
        self.bottle_labels = {entry.get('label', f'bottle{i}')
                              for i, entry in enumerate(rigids.get('bottle', []))}
        self.bin_id = next((v for k, v in self.obj_body_ids.items()
                            if "dustbin" in k), None)

        self.finger_body_ids = {
            side: {int(self.model.body(f"{side}_link7").id),
                   int(self.model.body(f"{side}_link8").id)}
            for side in ("left", "right")
        }



    MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "assets", "meshes")

    def _find_mesh(self, kind, entry):
        import json as _json
        cidx = entry.get("category_idx", 0)
        grp = "Rigid" if kind in ("bottle", "pen", "watch", "garage") else "Geometry"
        name = f"{grp}_{kind}_{cidx:05d}"
        obj = os.path.join(self.MESH_DIR, name + ".obj")
        bb = os.path.join(self.MESH_DIR, name + ".bbox.json")
        if os.path.exists(obj) and os.path.exists(bb):
            return obj, _json.loads(Path(bb).read_text())
        return None, None


    BOTTLE_COLOR = {22: [0.9, 0.9, 0.85], 25: [0.85, 0.7, 0.1],
                    41: [0.1, 0.6, 0.15], 60: [0.9, 0.75, 0.7]}

    def _add_collider(self, b, lo, hi, mass, open_top):
        c = (lo + hi) / 2
        half = (hi - lo) / 2
        if open_top:
            parts = [
                (half[0], half[1], 0.006, 0, 0, lo[2] + 0.006 - c[2]),
                (0.006, half[1], half[2], -half[0] + 0.006 - 0.006, 0, 0),
                (0.006, half[1], half[2], half[0] - 0.006 + 0.006, 0, 0),
                (half[0], 0.006, half[2], 0, -half[1] + 0.006, 0),
                (half[0], 0.006, half[2], 0, half[1] - 0.006, 0),
            ]
            parts = [
                (half[0], half[1], 0.006, 0, 0, -half[2] + 0.006),
                (0.006, half[1], half[2], -half[0] + 0.006, 0, 0),
                (0.006, half[1], half[2], half[0] - 0.006, 0, 0),
                (half[0], 0.006, half[2], 0, -half[1] + 0.006, 0),
                (half[0], 0.006, half[2], 0, half[1] - 0.006, 0),
            ]
            m_each = mass / 5
        else:
            parts = [(half[0], half[1], half[2], 0, 0, 0)]
            m_each = mass
        for sx, sy, sz, px, py, pz in parts:
            g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[max(sx, 0.002), max(sy, 0.002), max(sz, 0.002)],
                           pos=(c + np.array([px, py, pz])).tolist())
            g.mass = m_each
            g.friction = [1.0, 0.005, 0.0005]
            g.rgba = [0.8, 0.8, 0.8, 0.0]

    # ---------- classify_objects ----------
    CAT_STYLE = {
        0: (mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.07, 0], [0.8, 0.1, 0.1], 0.02),
        1: (mujoco.mjtGeom.mjGEOM_BOX, [0.02, 0.02, 0.008], [0.1, 0.2, 0.8], 0.05),
        2: (mujoco.mjtGeom.mjGEOM_BOX, [0.06, 0.04, 0.03], [0.1, 0.7, 0.2], 0.12),
    }

    def _build_classify(self, layout, world, spec):
        self.baskets = {}
        self.cat_items = {}
        for grp in ("Rigid", "Geometry"):
            for kind, entries in layout.get(grp, {}).items():
                for e in entries:
                    label = e.get("label", kind)
                    pos = list(e["default_pos"])
                    if label.startswith("basket"):
                        hx, hy, hz = 0.10, 0.10, 0.09
                        b = world.add_body()
                        b.name = label
                        b.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
                        mesh_file, bbox = self._find_mesh("basket", e)
                        if mesh_file:
                            mn = f"mesh_{label}"
                            spec.add_mesh(name=mn, file=mesh_file)
                            gv = b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH)
                            gv.meshname = mn
                            gv.mass = 0
                            gv.contype = 0
                            gv.conaffinity = 0
                            gv.rgba = [0.3, 0.35, 0.45, 1.0]
                            lo = np.array(bbox["min"]); hi = np.array(bbox["max"])
                            self._add_collider(b, lo, hi, 0.3, open_top=True)
                            hx, hy = (hi[0] - lo[0]) / 2 - 0.01, (hi[1] - lo[1]) / 2 - 0.01
                            hz = (hi[2] - lo[2]) / 2
                        else:
                            for sx, sy, sz, px, py, pz in [
                                (hx + BIN_WALL, hy + BIN_WALL, 0.008, 0, 0, 0.008),
                                (hx, BIN_WALL, hz - 0.02, 0, hy + BIN_WALL, hz),
                                (hx, BIN_WALL, hz - 0.02, 0, -(hy + BIN_WALL), hz),
                                (BIN_WALL, hy, hz - 0.02, hx + BIN_WALL, 0, hz),
                                (BIN_WALL, hy, hz - 0.02, -(hx + BIN_WALL), 0, hz),
                            ]:
                                g = b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                               size=[sx, sy, sz], pos=[px, py, pz])
                                g.mass = 0.1
                                g.friction = [1.0, 0.005, 0.0005]
                                g.rgba = [0.3, 0.35, 0.45, 1.0]
                        b.pos = pos
                        b.quat = e["default_ori"]
                        self.obj_bodies[label] = b
                        self.obj_poses[label] = (pos, e["default_ori"])
                        self.baskets[label] = (hx, hy, hz)
                    elif label.startswith("cat"):
                        cidx = int(label.split("_")[0][3:])
                        catname = label.split("_")[0]
                        mass = e.get("physics", {}).get("mass", 0.05)
                        b = world.add_body()
                        b.name = label
                        b.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)
                        mesh_file, bbox = self._find_mesh(kind, e)
                        if mesh_file:
                            mn = f"mesh_{label}"
                            spec.add_mesh(name=mn, file=mesh_file)
                            gv = b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH)
                            gv.meshname = mn
                            gv.mass = 0
                            gv.contype = 0
                            gv.conaffinity = 0
                            gv.rgba = [0.85, 0.85, 0.88, 1.0]
                            lo = np.array(bbox["min"]); hi = np.array(bbox["max"])
                            self._add_collider(b, lo, hi, mass, open_top=False)
                        else:
                            gtype, size, rgba, _ = self.CAT_STYLE.get(
                                cidx, self.CAT_STYLE[0])
                            g = b.add_geom(type=gtype, size=size, pos=[0, 0, 0])
                            g.mass = mass
                            g.friction = [1.0, 0.005, 0.0005]
                            g.rgba = list(rgba) + [1.0]
                        b.pos = pos
                        b.quat = e["default_ori"]
                        self.obj_bodies[label] = b
                        self.obj_poses[label] = (pos, e["default_ori"])
                        # category_idx identifies an asset variant (for example
                        # pen[1] vs pen[4]); the official task groups by the
                        # semantic category name.  Using cidx here would turn
                        # one class into several baskets and make the success
                        # condition impossible on real layouts.
                        semantic_category = e.get("category", kind)
                        self.cat_items.setdefault(semantic_category, []).append(label)

    def _score_classify(self):
        if not getattr(self, "baskets", None):
            return
        pure = 0
        for blabel, (hx, hy, zmax) in self.baskets.items():
            bpos = self.data.xpos[self.obj_body_ids[blabel]]
            filler = None
            ok = True
            for cidx, items in self.cat_items.items():
                inside = []
                for it in items:
                    p = self.data.xpos[self.obj_body_ids[it]]
                    in_xy = abs(p[0] - bpos[0]) < hx and abs(p[1] - bpos[1]) < hy
                    settled = -zmax <= p[2] - bpos[2] <= zmax + 0.01
                    inside.append(in_xy and settled)
                if all(inside):
                    if filler is None:
                        filler = cidx
                    else:
                        ok = False  # two categories fully inside: mixed
                elif any(inside):
                    ok = False  # partial: mixed
            if ok and filler is not None:
                pure += 1
        if not self.grippers_open():
            return
        for need, sc in ((1, 15), (2, 40), (3, 100)):
            if pure >= need:
                self.score = max(self.score, sc)
        if pure >= 3 and self.arms_home():
            self.success = True

    # ---------- io ----------
    def state14(self):
        if self.closed:
            return self.final_state
        q = self.data.qpos
        s = [float(q[i]) for i in self.qpos_ids[:6]] + [grip_norm(q[self.qpos_ids[6]])] + \
            [float(q[i]) for i in self.qpos_ids[7:13]] + [grip_norm(q[self.qpos_ids[13]])]
        return s

    def capture(self):
        import io
        for name, cid in zip(("cam_base", "left_cam_wrist", "right_cam_wrist"), self.cam_ids):
            self.renderer.update_scene(self.data, cid)
            pix = self.renderer.render()
            buf = io.BytesIO(); Image.fromarray(pix).save(buf, format="PNG")
            self.frame_cache[name] = buf.getvalue()
            if self.record_dir:
                directory = self.record_dir / name
                directory.mkdir(exist_ok=True)
                (directory / f"{self.t:06d}.png").write_bytes(buf.getvalue())
        if self.record_dir:
            with (self.record_dir / "states.jsonl").open("a") as f:
                f.write(json.dumps({**self._status(), "state": self.state14(), "ee": self.ee_poses(), "fingers": self.finger_state()})+"\n")

    def finger_state(self):
        return {side: {"q": [float(self.data.qpos[self.model.joint(f"{side}_joint{i}").qposadr[0]]) for i in (7, 8)],
                       "target": float(self.command_ctrl[idx])}
                for side, idx in (("left", 6), ("right", 13))}

    def ee_poses(self):
        if self.closed:
            return self.final_ee
        result = {}
        for side in ("left", "right"):
            site = self.model.site(side+"_ee").id
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.site_xmat[site])
            result[side] = {"xyz": self.data.site_xpos[site].tolist(), "quat_wxyz": quat.tolist(),
                            "gripper": self.state14()[6 if side == "left" else 13]}
        return result

    def images(self):
        import io
        out, shapes = {}, {}
        for name, raw in self.frame_cache.items():
            pix = np.asarray(Image.open(io.BytesIO(raw)))
            out[name] = base64.b64encode(pix.tobytes()).decode()
            shapes[name] = list(pix.shape)
        return out, shapes

    # ---------- scoring ----------
    def bottles_in_bin(self):
        if self.bin_id is None:
            return 0
        bin_xpos = self.data.xpos[self.bin_id]
        hx, hy = getattr(self, "bin_inner", (BIN_R_IN, BIN_R_IN))
        n = 0
        for label, bid in self.obj_body_ids.items():
            if label not in self.bottle_labels:
                continue
            p = self.data.xpos[bid]
            inside_xy = abs(p[0] - bin_xpos[0]) < hx and abs(p[1] - bin_xpos[1]) < hy
            z_gap = p[2] - bin_xpos[2]
            if inside_xy and 0.0 <= z_gap <= Z_GAP_MAX:
                n += 1
        return n

    def grippers_open(self):
        q = self.data.qpos
        return grip_norm(q[self.qpos_ids[6]]) > GRIP_OPEN_TH and \
            grip_norm(q[self.qpos_ids[13]]) > GRIP_OPEN_TH

    def arms_home(self):
        dq = np.abs(self.data.qpos[self.qpos_ids[:6]] - self.q_home[self.qpos_ids[:6]]).max()
        dq2 = np.abs(self.data.qpos[self.qpos_ids[7:13]] - self.q_home[self.qpos_ids[7:13]]).max()
        return max(dq, dq2) < ORIGIN_TOL

    def update_score(self):
        if self.task == "classify_objects":
            self._score_classify()
        elif self.grippers_open():
            n = self.bottles_in_bin()
            for k, sc in ((1, 10), (2, 25), (3, 40), (4, 100)):
                if n >= k: self.score = max(self.score, sc)
            self.success = bool(n >= 4 and self.arms_home())
        self.done = bool(self.success or self.t >= self.step_limit or self.error)

    def _step(self, target, *, measured_anchor=False):
        cur = self.command_ctrl.copy()  # already physical units
        if measured_anchor:
            # Do not carry a pending Pi joint target into a measured-state
            # correction. Re-anchor arm commands, without changing qpos/qvel.
            arm_cols = [i for i in range(14) if i not in (6, 13)]
            cur[arm_cols] = self.data.qpos[[self.qpos_ids[i] for i in arm_cols]]
        requested = np.asarray(target, float).copy()
        for side, idx in (("left", 6), ("right", 13)):
            # π0.5 emits a continuous normalized jaw position.  Do not turn
            # the middle of [0, 1] into a hold band: its typical 0.55--0.8
            # commands then leave a jaw at its reset position forever, so no
            # physical grasp can happen.  The policy contract is the official
            # affine mapping [0,1] -> [GRIP_MIN, GRIP_MAX].
            raw = float(np.clip(requested[idx], 0.0, 1.0))
            self.grip_target_norm[side] = raw
            requested[idx] = raw
            self.commanded_grip[side] = raw
        requested[6], requested[13] = grip_denorm(requested[6]), grip_denorm(requested[13])
        for side, idx in (("left", 6), ("right", 13)):
            if GRASP_ASSIST and self.grasped[side] is not None and self.commanded_grip[side] <= 0.8:
                requested[idx] = self.grasp_opening[side]
        # Slew-limit and low-pass the target in actuator space.  This removes
        # the high-frequency component while retaining the exact end target.
        delta = requested - cur
        limits = np.full(14, ARM_CTRL_STEP, dtype=float)
        limits[[6, 13]] = GRIP_CTRL_STEP
        bounded = cur + np.clip(delta, -limits, limits)
        alpha = float(np.clip(CTRL_SMOOTH, 0.0, 1.0))
        goal = cur + alpha * (bounded - cur)
        self.command_ctrl = goal.copy()
        for sub in range(SUBSTEPS):
            frac = min(1.0, (sub + 1) / INTERP_SUBSTEPS)
            self.data.ctrl[self.ctrl_ids] = cur + (goal - cur) * frac
            for side, idx in (("left", 6), ("right", 13)):
                self.data.ctrl[self.model.actuator(side+"_follower").id] = self.data.ctrl[self.ctrl_ids[idx]]
            warnings = [w.number for w in self.data.warning]
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self._update_grasps()
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or any(w.number > n for w,n in zip(self.data.warning,warnings)):
                self.error = "physics_error"; self.done = True
                raise RuntimeError("Physics warning/nonfinite state; episode terminated")
        mujoco.mj_forward(self.model, self.data)
        self.t += 1
        self.update_score()
        self.capture()

    def _update_grasps(self):
        """Acquire/release stable grasp constraints from real finger contacts."""
        if not GRASP_ASSIST:
            # Observation only: no position/velocity writes or equality activation.
            for side in ("left", "right"):
                touching = {}
                for c in self.data.contact:
                    if c.dist > 0: continue
                    b1, b2 = int(self.model.geom_bodyid[c.geom1]), int(self.model.geom_bodyid[c.geom2])
                    for finger, other in ((b1,b2),(b2,b1)):
                        if finger not in self.finger_body_ids[side]: continue
                        for name, bid in self.obj_body_ids.items():
                            if bid == other and name.startswith("bottle"):
                                touching.setdefault(name,set()).add(finger)
                self.grasped[side] = next((name for name, fingers in touching.items()
                                           if len(fingers) == 2), None)
            return
        # First release explicitly opened grippers.
        for side in ("left", "right"):
            if self.grasped[side] is not None:
                if self.commanded_grip[side] > 0.8:
                    eq = self.model.equality(f"grasp_{side}_{self.grasped[side]}").id
                    self.data.eq_active[eq] = False
                    self.grasp_opening.pop(side, None)
                    self.grasped[side] = None

        # Acquire only when both fingers touch the same bottle while closed.
        for side in ("left", "right"):
            if self.grasped[side] is not None:
                continue
            if self.commanded_grip[side] > 0.55:
                continue
            touched = {}
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                b1 = int(np.asarray(self.model.geom(int(c.geom1)).bodyid).item())
                b2 = int(np.asarray(self.model.geom(int(c.geom2)).bodyid).item())
                finger = b1 if b1 in self.finger_body_ids[side] else b2 if b2 in self.finger_body_ids[side] else None
                other = b2 if finger == b1 else b1 if finger == b2 else None
                if finger is None or other is None:
                    continue
                label = next((name for name, bid in self.obj_body_ids.items()
                              if bid == other and name.startswith("bottle")), None)
                if label:
                    touched.setdefault(label, set()).add(finger)
            candidate = next((label for label, fingers in touched.items() if len(fingers) >= 2), None)
            if candidate is None:
                continue
            bid = self.obj_body_ids[candidate]
            self.grasped[side] = candidate
            self.grasp_opening[side] = float(self.data.qpos[self.model.joint(side+"_joint7").qposadr[0]])
            parent = self.model.body(side+"_link6").id
            eq = self.model.equality(f"grasp_{side}_{candidate}").id
            rot = self.data.xmat[parent].reshape(3,3)
            relpos = rot.T @ (self.data.xpos[bid] - self.data.xpos[parent])
            inv = np.zeros(4); relquat = np.zeros(4)
            mujoco.mju_negQuat(inv, self.data.xquat[parent])
            mujoco.mju_mulQuat(relquat, inv, self.data.xquat[bid])
            self.model.eq_data[eq, :3] = 0
            self.model.eq_data[eq, 3:6] = relpos
            self.model.eq_data[eq, 6:10] = relquat
            self.model.eq_data[eq, 10] = 0.1
            self.data.eq_active[eq] = True

    def act(self, joints):
        rows = np.asarray(joints, float)
        if rows.ndim != 2 or rows.shape[1] != 14 or not 1 <= len(rows) <= 50 or not np.isfinite(rows).all():
            raise ValueError("Expected 1..50 finite joint rows of width 14")
        if np.any((rows[:, [6,13]] < 0) | (rows[:, [6,13]] > 1)):
            raise ValueError("Grippers must be in [0,1]")
        if len(rows) > self.step_limit - self.t:
            raise ValueError("Action exceeds remaining step budget")
        for col, qi in enumerate(self.qpos_ids):
            if col in (6,13): continue
            jid = self.model.dof_jntid[self.model.jnt_dofadr[self.model.actuator_trnid[self.ctrl_ids[col],0]]]
            if self.model.jnt_limited[jid] and np.any((rows[:,col]<self.model.jnt_range[jid,0]) | (rows[:,col]>self.model.jnt_range[jid,1])):
                raise ValueError("Joint target outside model range")
        with self.lock:
            for row in rows:
                if self.done: break
                self._step(row)
            return self._status()

    def eef(self, goals, steps):
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=5 or steps>self.step_limit-self.t:
            raise ValueError("EEF step count must be 1..5 within budget")
        validate_goals(goals)
        with self.lock:
            # Validate the initial decision only; disturbances during tracking
            # are handled by bounded updates, not by reapplying this gate.
            solve_step(self.model, self.data, goals)
            for _ in range(steps):
                if self.done: break
                try:
                    row = solve_step(self.model, self.data, goals, check_initial_bound=False)
                except ValueError as exc:
                    self.error = "eef_tracking_error"; self.done = True
                    raise RuntimeError(str(exc)) from exc
                self._step(row, measured_anchor=True) # fresh IK only after real physics
            return self._status()

    def _status(self):
        if self.closed:
            return dict(self.final_status)
        return {"t": self.t, "done": bool(self.done), "success": bool(self.success),
                "score": float(self.score), "bottles_in": self.bottles_in_bin(),
                "grasped": {side: label for side, label in self.grasped.items()},
                "object_positions": {name: self.data.xpos[bid].tolist() for name,bid in self.obj_body_ids.items() if name.startswith("bottle") or name == "dustbin"},
                "remaining_steps": max(0,self.step_limit-self.t),
                "termination": self.error or ("success" if self.success else "step_limit" if self.t>=self.step_limit else None)}

    def close(self):
        """Release heavy simulation resources, retaining final API evidence."""
        if self.closed:
            return
        if not self.done:
            raise ValueError('Finish the active session before releasing it')
        self.final_status = self._status()
        self.final_state = self.state14()
        self.final_ee = self.ee_poses()
        self.renderer.close()
        self.renderer = self.data = self.model = None
        self.closed = True


SESSIONS = {}
LATEST = None

class Handler(BaseHTTPRequestHandler):
    # Browsers may preconnect without sending a request. Keep the renderer on
    # this single server thread, but bound idle socket waits so actions proceed.
    timeout = 2

    def log_message(self, *args): pass

    def send_bytes(self, body, content_type, code=200):
        try:
            self.send_response(code)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _send(self, obj, code=200):
        self.send_bytes(json.dumps(obj,allow_nan=False).encode(),'application/json',code)

    def do_GET(self):
        path=urlsplit(self.path).path
        parts=path.strip('/').split('/')
        if path in ('/','/live'):
            return self.send_bytes((Path(__file__).with_name('live.html')).read_bytes(),'text/html; charset=utf-8')
        if path=='/health':
            return self._send({'ok':True,'version':'astra-isolated-v5-physical-grasp','port':PORT,
                               'render_backend':os.environ.get('MUJOCO_GL','default'),
                               'physics_hz':PHYS_HZ,'control_hz':CTRL_HZ,
                               'eef':'jaw_center','eef_tracking':'bounded_dls_v2','grasp_assist':GRASP_ASSIST,
                               'arm_ctrl_step':ARM_CTRL_STEP,
                               'grip_ctrl_step':GRIP_CTRL_STEP,
                               'ctrl_smooth':CTRL_SMOOTH,
                               'arm_kp':ARM_KP,
                               'arm_kd_floor':ARM_KD_FLOOR})
        if path=='/status':
            sess=SESSIONS.get(LATEST)
            return self._send({**sess._status(),'session_id':LATEST,'instruction':sess.instruction} if sess else {'session_id':None})
        if len(parts)==3 and parts[0]=='frame':
            sess=SESSIONS.get(parts[1]); raw=sess.frame_cache.get(parts[2]) if sess else None
            return self.send_bytes(raw,'image/png') if raw else self._send({'error':'frame not found'},404)
        if len(parts)==3 and parts[0]=='session':
            sess=SESSIONS.get(parts[1])
            if not sess:return self._send({'error':'session not found'},404)
            if parts[2]=='result':return self._send(sess._status())
            if parts[2]=='state':
                return self._send({**sess._status(),'state':sess.state14(),
                                   'ee':sess.ee_poses(),'instruction':sess.instruction,
                                   'observation_id':f'{parts[1]}:{sess.t}'})
            if parts[2]=='observe':
                imgs,shapes=sess.images()
                return self._send({**sess._status(),'images':imgs,'shapes':shapes,'state':sess.state14(),'ee':sess.ee_poses(),'instruction':sess.instruction,'observation_id':f'{parts[1]}:{sess.t}'})
        self._send({'error':'not found'},404)

    def do_POST(self):
        global LATEST
        parts=urlsplit(self.path).path.strip('/').split('/')
        try:
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<10_000_000:raise ValueError('Invalid body size')
            req=json.loads(self.rfile.read(size))
            if parts==['session']:
                if any(not s.done for s in SESSIONS.values()):
                    return self._send({'error':'Finish the active session before creating another'},409)
                for old in SESSIONS.values():
                    old.close()
                while len(SESSIONS) >= 8:
                    SESSIONS.pop(next(iter(SESSIONS)))
                sid=uuid.uuid4().hex[:12]
                sess=Session(req['layout'],req['instruction'],req.get('task','put_bottles'),RUN_ROOT/sid,req.get('overview_yaw'))
                SESSIONS[sid]=sess;LATEST=sid
                return self._send({'session_id':sid,**sess._status(),'record_dir':str(sess.record_dir)})
            if len(parts)!=3 or parts[0]!='session' or parts[2] not in ('act','eef','finish'):
                return self._send({'error':'not found'},404)
            sess=SESSIONS.get(parts[1])
            if not sess:return self._send({'error':'session not found'},404)
            rid=req.get('request_id')
            if not isinstance(rid,str) or not 1<=len(rid)<=128:raise ValueError('request_id required')
            digest=hashlib.sha256(json.dumps({'route':parts[2],'body':req},sort_keys=True).encode()).hexdigest()
            if rid in sess.requests:
                old,result,code=sess.requests[rid]
                if old!=digest:return self._send({'error':'request_id payload conflict'},409)
                return self._send(result,code)
            if type(req.get('expected_step')) is not int or req['expected_step']!=sess.t:return self._send({'error':'stale observation',**sess._status()},409)
            if sess.done and parts[2] in ('act','eef'):
                return self._send({'error':'Session has ended',**sess._status()},409)
            with (sess.record_dir/'requests.jsonl').open('a') as f:f.write(json.dumps({'route':parts[2],'request':req})+'\n')
            try:
                if parts[2]=='act': result=sess.act(req['joints'])
                elif parts[2]=='eef': result=sess.eef(req['goals'],req.get('steps',1))
                else:
                    if not sess.done:sess.error='stopped_by_operator';sess.done=True
                    videos=[]
                    for cam in sess.frame_cache:
                        dest=sess.record_dir/(cam+'.mp4')
                        encode_video(ROOT, sess.record_dir/cam, dest, CTRL_HZ)
                        videos.append(str(dest))
                    result={**sess._status(),'videos':videos}
                    sess.close()
                code=200
            except ValueError as exc:
                result={'error':str(exc),**sess._status()};code=400
            except Exception as exc:
                if parts[2]!='finish': sess.error='execution_error';sess.done=True
                result={'error':str(exc),**sess._status()};code=500
            sess.requests[rid]=(digest,result,code)
            with (sess.record_dir/'responses.jsonl').open('a') as f:f.write(json.dumps({'request_id':rid,'code':code,'response':result})+'\n')
            self._send(result,code)
        except (ValueError,KeyError,TypeError) as exc:self._send({'error':str(exc)},400)
        except Exception as exc:self._send({'error':f'{type(exc).__name__}: {exc}'},500)

if __name__ == '__main__':
    print(f'simsvc on :{PORT}',flush=True)
    HTTPServer.allow_reuse_address = True
    HTTPServer((HOST,PORT),Handler).serve_forever()
