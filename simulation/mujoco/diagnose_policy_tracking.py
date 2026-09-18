"""Unresolved tracking diagnostic, not a portable acceptance test.

Requires the preserved 20260917 run. The assertion intentionally reproduces the
observed tracking failure; it does not establish that its targets are feasible.
Disabling contacts is an offline ablation only, never a proposed production fix.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))
import simsvc


def replay_tracking(
    *,
    recorded: Path | None = None,
    disable_contacts: bool = False,
    unfiltered: bool = False,
) -> dict:
    """Diagnostic ablations are process-local and never affect the live service."""
    recorded = (
        recorded or ROOT / "runs/author_hybrid_20260917_225148/030fd4325941/decisions"
    )
    rows = []
    evidence_hash = hashlib.sha256()
    for index in range(3):
        directory = recorded / f"{index:04d}"
        for p in sorted(directory.glob("execution_request_*.json")):
            raw = p.read_bytes()
            request = json.loads(raw)
            if request["expected_step"] != len(rows) or len(request["joints"]) != 1:
                raise ValueError(
                    "Fixture must contain consecutive single-step requests"
                )
            evidence_hash.update(raw)
            rows.append(request["joints"][0])
    array = np.asarray(rows, dtype=float)
    if array.shape != (42, 14) or not np.isfinite(array).all():
        raise ValueError("Expected complete finite 42x14 recorded fixture")
    layout = json.loads((ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text())
    with patch.object(simsvc.mujoco, "Renderer", MagicMock()), patch.object(
        simsvc.Session, "capture"
    ):
        session = simsvc.Session(layout, "recorded tracking regression")
        if disable_contacts:
            session.model.opt.disableflags |= int(
                simsvc.mujoco.mjtDisableBit.mjDSBL_CONTACT
            )
        with patch.object(
            simsvc, "ARM_CTRL_STEP", 100.0 if unfiltered else simsvc.ARM_CTRL_STEP
        ), patch.object(
            simsvc, "CTRL_SMOOTH", 1.0 if unfiltered else simsvc.CTRL_SMOOTH
        ):
            for row in rows:
                session._step(row)
            target = np.asarray(rows[-1])
            for _ in range(150):
                session._step(target)
        actual = session.data.qpos[session.qpos_ids].copy()
        if actual.shape != (14,) or target.shape != (14,):
            raise ValueError("Tracking comparison requires flat 14-element state")
        arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        contacts = []
        for c in session.data.contact:
            if c.dist < 0:
                contacts.append(
                    dict(
                        a=session.model.geom(c.geom1).name,
                        b=session.model.geom(c.geom2).name,
                        body_a=session.model.body(
                            session.model.geom_bodyid[c.geom1]
                        ).name,
                        body_b=session.model.body(
                            session.model.geom_bodyid[c.geom2]
                        ).name,
                        depth_mm=float(-c.dist * 1000),
                    )
                )
        result = dict(
            diagnostic_only=True,
            fixture_sha256=evidence_hash.hexdigest(),
            source_sha256=hashlib.sha256(
                Path(simsvc.__file__).read_bytes()
            ).hexdigest(),
            target=target.tolist(),
            applied=session.command_ctrl.tolist(),
            actual=actual.tolist(),
            max_arm_error=float(
                np.max(np.abs(target[arm_indices] - actual[arm_indices]))
            ),
            max_filter_error=float(
                np.max(np.abs(target[arm_indices] - session.command_ctrl[arm_indices]))
            ),
            contacts=contacts,
            disable_contacts=disable_contacts,
            unfiltered=unfiltered,
        )
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded", type=Path)
    parser.add_argument("--disable-contacts", action="store_true")
    parser.add_argument("--unfiltered", action="store_true")
    parser.add_argument(
        "--assert-tracking",
        action="store_true",
        help="Reproduce the unresolved failure with nonzero exit",
    )
    args = parser.parse_args()
    result = replay_tracking(
        recorded=args.recorded,
        disable_contacts=args.disable_contacts,
        unfiltered=args.unfiltered,
    )
    print(json.dumps(result, indent=2))
    if args.assert_tracking and result["max_arm_error"] >= 0.15:
        raise SystemExit("Recorded target not tracked; feasibility is not established")
