#!/usr/bin/env python3
"""Deterministic dual-X5 pick/place smoke test (no GPT or π0.5).

Exercises the real HTTP EEF path: approach bottle0, close until the grasp
contacts support it, lift/carry to the dustbin, release, and assert that the
scorer sees the bottle inside.  Every waypoint is subdivided to respect the
service's 5 cm EEF safety bound.
"""
import argparse
import copy
import json
import pathlib
import sys

import numpy as np

import client


def move_arm(sid, target, grip, label, max_delta=0.035, side="left"):
    target = np.asarray(target, float)
    for _ in range(200):
        obs, _ = client.observe(sid)
        cur = np.asarray(obs["ee"][side]["xyz"], float)
        delta = target - cur
        dist = float(np.linalg.norm(delta))
        if dist <= 0.012:
            # Hold for two control ticks so the gripper actuator settles.
            for _ in range(2):
                obs, _ = client.observe(sid)
                goals = {s: {"xyz": obs["ee"][s]["xyz"],
                             "quat_wxyz": obs["ee"][s]["quat_wxyz"],
                             "gripper": obs["ee"][s]["gripper"]}
                         for s in ("left", "right")}
                goals[side]["gripper"] = float(grip)
                client.execute(sid, obs["t"], "eef", {"goals": goals, "steps": 5}, label)
            return obs
        nxt = cur + delta * min(1.0, max_delta / max(dist, 1e-9))
        goals = {s: {"xyz": obs["ee"][s]["xyz"],
                     "quat_wxyz": obs["ee"][s]["quat_wxyz"],
                     "gripper": obs["ee"][s]["gripper"]}
                 for s in ("left", "right")}
        goals[side]["xyz"] = nxt.tolist()
        goals[side]["gripper"] = float(grip)
        client.execute(sid, obs["t"], "eef", {"goals": goals, "steps": 5}, label)
    raise RuntimeError("waypoint did not converge: " + label)


def move_left(sid, target, grip, label, max_delta=0.035):
    return move_arm(sid, target, grip, label, max_delta)


def main(side="left"):
    sign = 1 if side == "left" else -1
    def move(sid, target, grip, label):
        return move_arm(sid, target, grip, label, side=side)
    root = pathlib.Path(__file__).resolve().parent
    layout = json.loads((root / "layouts/put_bottles_into_dustbin_0.json").read_text())
    if side == "right":
        # Explicit mirrored smoke fixture, not an official benchmark layout.
        layout = copy.deepcopy(layout)
        layout["Table"]["default_pos"][0] *= -1
        for group in (layout.get("Rigid", {}), layout.get("Geometry", {})):
            for entries in group.values():
                for entry in entries:
                    if "default_pos" in entry: entry["default_pos"][0] *= -1
                    if "default_ori" in entry:
                        q = entry["default_ori"]
                        entry["default_ori"] = [q[0], q[1], -q[2], -q[3]]
    sid = client.http(client.SIM + "/session", {
        "layout": layout,
        "instruction": f"{side} arm smoke test: pick bottle0 and place in bin; mirrored fixture={side == 'right'}.",
        "task": "put_bottles",
    })["session_id"]
    bottle = layout["Rigid"]["bottle"][0]["default_pos"]
    move(sid, [bottle[0], bottle[1], 1.12], 1.0, "approach")
    move(sid, [bottle[0], bottle[1], 0.86], 1.0, "descend")
    move(sid, [bottle[0], bottle[1], 0.86], 0.0, "close")
    obs, _ = client.observe(sid)
    if obs.get("grasped", {}).get(side) != "bottle0":
        raise AssertionError(f"grasp not acquired: {obs.get('grasped')}")
    move(sid, [bottle[0], bottle[1], 1.12], 0.0, "lift")
    # Hold the measured arm pose for 3 simulated seconds with closed jaws.
    obs, _ = client.observe(sid)
    held = obs["ee"]
    held[side]["gripper"] = 0.0
    for _ in range(15):
        obs, _ = client.observe(sid)
        client.execute(sid, obs["t"], "eef", {"goals": held, "steps": 5}, "hold 3 seconds")
        after, _ = client.observe(sid)
        if after.get("grasped", {}).get(side) != "bottle0":
            raise AssertionError("lost bilateral contact during hold")
    move(sid, [-0.63*sign, -0.10, 1.12], 0.0, "carry")
    # Stay above the other bottles and bin rim; the old diagonal carry and
    # z=.72 insertion collided with bottle3/table and forced the jaws apart.
    move(sid, [-0.63*sign, -0.10, 1.12], 1.0, "release")
    # Let the released bottle settle through the opening before evaluating
    # containment; contact resolution can take several control ticks.
    for _ in range(25):
        obs, _ = client.observe(sid)
        if obs.get("bottles_in", 0) >= 1:
            break
        goals = {s: {"xyz": obs["ee"][s]["xyz"],
                     "quat_wxyz": obs["ee"][s]["quat_wxyz"],
                     "gripper": 1.0}
                 for s in ("left", "right")}
        client.execute(sid, obs["t"], "eef", {"goals": goals, "steps": 2}, "settle after release")
    obs, _ = client.observe(sid)
    if obs.get("bottles_in", 0) < 1:
        raise AssertionError(f"bottle not in bin: {obs.get('bottles_in')}")
    # Require the target itself, not another displaced bottle, to be in bin.
    for _ in range(10):
        obs, _ = client.observe(sid)
        goals = obs["ee"]
        goals["left"]["gripper"] = goals["right"]["gripper"] = 1.0
        client.execute(sid, obs["t"], "eef", {"goals": goals, "steps": 5}, "settle target in bin")
    obs, _ = client.observe(sid)
    positions = obs["object_positions"]
    relative = np.array(positions["bottle0"]) - np.array(positions["dustbin"])
    if not (abs(relative[0]) < 0.15 and abs(relative[1]) < 0.13 and 0 < relative[2] < 0.4):
        raise AssertionError(f"target bottle0 outside bin: {relative}")
    result = client.execute(sid, obs["t"], "finish", {}, "pick-place smoke test complete")
    print(json.dumps({"session_id": sid, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"), default="left")
    main(parser.parse_args().side)
