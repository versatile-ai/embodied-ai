"""Replay saved pi actions with real contacts; diagnostic, not a benchmark.

No model calls, object teleportation, or collision disabling. The optional
gain multiplier is process-local and never changes the live service.
"""

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))
import simsvc


def replay(run: Path, gain: float, steps: int, mass: float | None = None) -> dict:
    if not np.isfinite(gain) or not 0 < gain <= 4 or not 10 <= steps <= 200:
        raise ValueError("gain must be finite (0,4], steps 10..200")
    if mass is not None and (not np.isfinite(mass) or not 0 < mass <= 0.5):
        raise ValueError("mass override must be (0,0.5] kg; runtime cap still applies")
    layout = json.loads((ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text())
    if mass is not None:
        next(e for e in layout["Rigid"]["bottle"] if e["label"] == "bottle3")[
            "physics"
        ]["mass"] = mass
    chunks = []
    for t in range(0, steps, 10):
        chunk = np.asarray(
            json.loads((run / f"proposal_{t:06d}.json").read_text())["actions"],
            dtype=float,
        )
        if chunk.shape != (50, 14) or not np.isfinite(chunk).all():
            raise ValueError(f"Invalid 50x14 proposal at step {t}")
        chunks.append(chunk[:10])
    actions = np.concatenate(chunks)[:steps]
    if actions.shape != (steps, 14):
        raise ValueError("Incomplete recorded action sequence")
    records = []
    with patch.object(simsvc.mujoco, "Renderer", MagicMock()), patch.object(
        simsvc.Session, "capture"
    ):
        s = simsvc.Session(layout, "recorded pi grip diagnostic")
        try:
            for side in ("left", "right"):
                for name in ("gripper", "follower"):
                    idx = s.model.actuator(f"{side}_{name}").id
                    s.model.actuator_gainprm[idx, 0] *= gain
                    s.model.actuator_biasprm[idx, 1] *= gain
            bid = s.model.body("bottle3").id
            initial_z = float(s.data.xpos[bid, 2])
            weight = float(s.model.body_mass[bid] * abs(s.model.opt.gravity[2]))
            for row in actions:
                s._step(row)
                contacts = []
                for cidx, c in enumerate(s.data.contact):
                    bodies = [int(s.model.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
                    names = [s.model.body(b).name for b in bodies]
                    if bid not in bodies or not any(
                        n in ("left_link7", "left_link8") for n in names
                    ):
                        continue
                    wrench = np.zeros(6)
                    simsvc.mujoco.mj_contactForce(s.model, s.data, cidx, wrench)
                    contacts.append(
                        dict(
                            bodies=names,
                            normal_N=float(wrench[0]),
                            depth_mm=float(-c.dist * 1000),
                        )
                    )
                records.append(
                    dict(
                        t=s.t,
                        height=float(s.data.xpos[bid, 2]),
                        command=float(row[6]),
                        fingers=s.finger_state()["left"]["q"],
                        actuator_N=[
                            float(
                                s.data.actuator_force[s.model.actuator("left_" + n).id]
                            )
                            for n in ("gripper", "follower")
                        ],
                        contacts=contacts,
                    )
                )
            return dict(
                gain=gain,
                steps=steps,
                bottle_weight_N=weight,
                peak_lift_m=max(r["height"] for r in records) - initial_z,
                final_lift_m=records[-1]["height"] - initial_z,
                records=records,
            )
        finally:
            s.done = True
            s.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/80e6faab9826")
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--mass",
        type=float,
        help="Bottle3 mass override (0,0.5] kg; runtime cap still applies",
    )
    parser.add_argument("--assert-held", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gain <= 4 or not 10 <= args.steps <= 200:
        parser.error("gain must be (0,4], steps 10..200")
    if args.mass is not None and (
        not np.isfinite(args.mass) or not 0 < args.mass <= 0.5
    ):
        parser.error("mass must be (0,0.5] kg; cannot reproduce old 22 kg baseline")
    result = replay(args.run, args.gain, args.steps, args.mass)
    print(
        json.dumps(
            {
                **result,
                "records": [
                    r
                    for r in result["records"]
                    if 30 <= r["t"] <= 55 and r["t"] % 5 == 0
                ],
            },
            indent=2,
        )
    )
    if (
        args.assert_held
        and min(r["height"] for r in result["records"][-10:])
        - result["records"][0]["height"]
        < 0.05
    ):
        raise SystemExit(
            "FAIL: bottle not lifted and held 5 cm during final ten frames"
        )
