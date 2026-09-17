#!/usr/bin/env python3
"""Deterministic dual-X5 pick/place smoke test (no GPT or π0.5).

Exercises the real HTTP EEF path: approach bottle0, close until the grasp
assist acquires it, lift/carry to the dustbin, release, and assert that the
scorer sees the bottle inside.  Every waypoint is subdivided to respect the
service's 5 cm EEF safety bound.
"""
import json
import pathlib
import sys

import numpy as np

import client


def move_left(sid, target, grip, label, max_delta=0.035):
    target = np.asarray(target, float)
    for _ in range(200):
        obs, _ = client.observe(sid)
        cur = np.asarray(obs["ee"]["left"]["xyz"], float)
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
                goals["left"]["gripper"] = float(grip)
                client.execute(sid, obs["t"], "eef", {"goals": goals, "steps": 5}, label)
            return obs
        nxt = cur + delta * min(1.0, max_delta / max(dist, 1e-9))
        goals = {s: {"xyz": obs["ee"][s]["xyz"],
                     "quat_wxyz": obs["ee"][s]["quat_wxyz"],
                     "gripper": obs["ee"][s]["gripper"]}
                 for s in ("left", "right")}
        goals["left"]["xyz"] = nxt.tolist()
        goals["left"]["gripper"] = float(grip)
        client.execute(sid, obs["t"], "eef", {"goals": goals, "steps": 5}, label)
    raise RuntimeError("waypoint did not converge: " + label)


def main():
    root = pathlib.Path(__file__).resolve().parent
    layout = json.loads((root / "layouts/put_bottles_into_dustbin_0.json").read_text())
    sid = client.http(client.SIM + "/session", {
        "layout": layout,
        "instruction": "Pick up bottle0 and place it into the dustbin.",
        "task": "put_bottles",
    })["session_id"]
    bottle = layout["Rigid"]["bottle"][0]["default_pos"]
    move_left(sid, [bottle[0], bottle[1], 1.12], 1.0, "approach")
    move_left(sid, [bottle[0], bottle[1], 0.86], 1.0, "descend")
    move_left(sid, [bottle[0], bottle[1], 0.86], 0.0, "close")
    obs, _ = client.observe(sid)
    if obs.get("grasped", {}).get("left") != "bottle0":
        raise AssertionError(f"grasp not acquired: {obs.get('grasped')}")
    move_left(sid, [bottle[0], bottle[1], 1.12], 0.0, "lift")
    move_left(sid, [-0.63, -0.10, 0.95], 0.0, "carry")
    # Release above the bin floor, through the opening.  Releasing at the
    # nominal body centre (z=.40) can put the grasp offset below the collider.
    move_left(sid, [-0.63, -0.10, 0.72], 0.0, "lower into bin")
    move_left(sid, [-0.63, -0.10, 0.72], 1.0, "release")
    obs, _ = client.observe(sid)
    if obs.get("bottles_in", 0) < 1:
        raise AssertionError(f"bottle not in bin: {obs.get('bottles_in')}")
    result = client.execute(sid, obs["t"], "finish", {}, "pick-place smoke test complete")
    print(json.dumps({"session_id": sid, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
