#!/usr/bin/env python3
"""Live, deterministic motion validation for the dual-arm simulation.

This uses a fresh direct-mode simulator session and no GPT or pi0.5 request.
It verifies that small Cartesian translations, a small world-frame yaw change,
and gripper open/close commands all reduce their measured error.  The task
score is intentionally irrelevant: this is a controller health check.
"""
import copy
import json
import math
from pathlib import Path

import client


ROOT = Path(__file__).resolve().parent


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw]


def angular_error(a, b):
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    return 2 * math.acos(min(1.0, max(-1.0, dot)))


def goals_from_observation(obs):
    return {
        side: {
            "xyz": list(obs["ee"][side]["xyz"]),
            "quat_wxyz": list(obs["ee"][side]["quat_wxyz"]),
            "gripper": float(obs["ee"][side]["gripper"]),
        }
        for side in ("left", "right")
    }


def execute_eef(sid, obs, goals, label):
    return client.execute(
        sid, obs["t"], "eef", {"goals": goals, "steps": 5}, label
    )


def check_translation(sid, side, delta, label):
    before, _ = client.observe(sid)
    start = before["ee"][side]["xyz"]
    target = [start[i] + delta[i] for i in range(3)]
    goals = goals_from_observation(before)
    goals[side]["xyz"] = target
    execute_eef(sid, before, goals, label)
    after, _ = client.observe(sid)
    end = after["ee"][side]["xyz"]
    start_error = distance(start, target)
    end_error = distance(end, target)
    direction = sum((end[i] - start[i]) * delta[i] for i in range(3))
    return {
        "kind": "translation", "side": side, "label": label,
        "delta_m": delta, "start_error_m": start_error,
        "end_error_m": end_error, "progress_projection_m2": direction,
        # A numerical reduction below 10% is too small to call a useful
        # five-tick controller response.
        "passed": end_error <= start_error * 0.90 and direction > 1e-6,
    }


def check_yaw(sid, side):
    before, _ = client.observe(sid)
    start = before["ee"][side]["quat_wxyz"]
    half = 0.10 / 2
    target = quat_multiply([math.cos(half), 0.0, 0.0, math.sin(half)], start)
    goals = goals_from_observation(before)
    goals[side]["quat_wxyz"] = target
    execute_eef(sid, before, goals, f"{side} wrist yaw +0.10 rad")
    after, _ = client.observe(sid)
    end = after["ee"][side]["quat_wxyz"]
    start_error = angular_error(start, target)
    end_error = angular_error(end, target)
    return {
        "kind": "yaw", "side": side, "target_rad": 0.10,
        "start_error_rad": start_error, "end_error_rad": end_error,
        # Five 25 Hz control ticks are only 200 ms.  Require at least 5%
        # rotation-error reduction in that window: this catches the prior
        # cancellation bug (0.17%) without pretending a bounded correction is
        # a full 0.1 rad pose settle.
        "passed": end_error <= start_error * 0.95,
    }


def check_gripper(sid, side, target, label):
    before, _ = client.observe(sid)
    start = float(before["ee"][side]["gripper"])
    goals = goals_from_observation(before)
    goals[side]["gripper"] = target
    execute_eef(sid, before, goals, label)
    after, _ = client.observe(sid)
    end = float(after["ee"][side]["gripper"])
    return {
        "kind": "gripper", "side": side, "target": target,
        "start": start, "end": end,
        "passed": end > start + 0.01 if target > start else end < start - 0.01,
    }


def main():
    layout = json.loads((ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text())
    created = client.http(client.SIM + "/session", {
        "layout": layout,
        "instruction": "Deterministic controller motion validation; no policy model.",
        "task": "put_bottles",
    })
    sid = created["session_id"]
    checks = []
    try:
        for side, sign in (("left", 1.0), ("right", -1.0)):
            # 15 mm moves are within the 5 cm EEF target safety bound and keep
            # the home pose clear of all bottles.
            checks.append(check_translation(sid, side, [0.015 * sign, 0.0, 0.0], f"{side} lateral"))
            checks.append(check_translation(sid, side, [0.0, 0.0, 0.015], f"{side} lift"))
            checks.append(check_yaw(sid, side))
            checks.append(check_gripper(sid, side, 1.0, f"{side} open"))
            checks.append(check_gripper(sid, side, 0.0, f"{side} close"))
        final, _ = client.observe(sid)
        result = client.execute(sid, final["t"], "finish", {}, "motion validation complete")
        report = {
            "session_id": sid,
            "passed": all(c["passed"] for c in checks),
            "checks": checks,
            "final": result,
            "interpretation": "Each check requires measured error to decrease after a five-control-step EEF segment.",
        }
    except Exception as exc:
        report = {"session_id": sid, "passed": False, "checks": checks, "error": repr(exc)}
        raise
    finally:
        out = ROOT / "runs" / sid / "motion_validation.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
