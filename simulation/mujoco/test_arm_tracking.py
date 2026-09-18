"""Feasible dual-arm tracking regression with real dynamics and contacts."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))
import simsvc


class ArmTrackingTest(unittest.TestCase):
    def test_policy_speed_ramp_does_not_accumulate_filter_lag(self):
        layout = json.loads(
            (ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text()
        )
        with patch.object(simsvc.mujoco, "Renderer", MagicMock()), patch.object(
            simsvc.Session, "capture"
        ):
            session = simsvc.Session(layout, "2.5 rad/s policy ramp")
            row = np.asarray(session.state14())
            for step in range(1, 11):
                row[[1, 2, 8, 9]] = step * 0.1
                session.act([row])
            lag = float(np.max(np.abs(session.command_ctrl[[1, 2, 8, 9]] - 1)))
            error = float(
                np.max(np.abs(np.asarray(session.state14())[[1, 2, 8, 9]] - 1))
            )
            session.done = True
            session.close()
            self.assertLess(lag, 0.03, "Limiter discards policy timing")
            self.assertLess(error, 0.25, "Fast trajectory is not physically tracked")

    def test_large_target_changes_remain_slew_limited(self):
        layout = json.loads(
            (ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text()
        )
        with patch.object(simsvc.mujoco, "Renderer", MagicMock()), patch.object(
            simsvc.Session, "capture"
        ):
            session = simsvc.Session(layout, "slew limit regression")
            row = np.asarray(session.state14())
            bounds = np.full(14, simsvc.ARM_CTRL_STEP * simsvc.CTRL_SMOOTH)
            bounds[[6, 13]] = simsvc.GRIP_CTRL_STEP * simsvc.CTRL_SMOOTH
            for target in (1.0, 0.0, 1.0):
                row[[1, 2, 8, 9]] = target
                row[[6, 13]] = target
                previous = session.command_ctrl.copy()
                session.act([row])
                delta = np.abs(session.command_ctrl - previous)
                self.assertTrue(np.all(delta <= bounds + 1e-12))
                self.assertGreater(delta[1], 0)
                self.assertTrue(np.isfinite(session.data.qpos).all())
                self.assertTrue(np.isfinite(session.data.qvel).all())
                self.assertIsNone(session.error)

    def test_bilateral_one_radian_per_second_ramp(self):
        layout = json.loads(
            (ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text()
        )
        with patch.object(simsvc.mujoco, "Renderer", MagicMock()), patch.object(
            simsvc.Session, "capture"
        ):
            session = simsvc.Session(layout, "dual-arm tracking regression")
            self.assertFalse(simsvc.GRASP_ASSIST)
            self.assertFalse(
                session.model.opt.disableflags
                & int(simsvc.mujoco.mjtDisableBit.mjDSBL_CONTACT)
            )
            row = np.asarray(session.state14())
            max_error = 0.0
            max_target_lag = 0.0
            max_bilateral_error = 0.0
            arms = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
            # Positive shoulder and elbow angles unfold above the table.
            # The same one-second ramp on both arms must not accumulate delay.
            for step in range(1, simsvc.CTRL_HZ + 1):
                row[[1, 2, 8, 9]] = step / simsvc.CTRL_HZ
                session.act([row])
                actual = session.data.qpos[session.qpos_ids]
                max_error = max(
                    max_error, float(np.max(np.abs(actual[arms] - row[arms])))
                )
                max_target_lag = max(
                    max_target_lag,
                    float(np.max(np.abs(session.command_ctrl[arms] - row[arms]))),
                )
                max_bilateral_error = max(
                    max_bilateral_error,
                    float(np.max(np.abs(actual[:6] - actual[7:13]))),
                )
            print(
                json.dumps(
                    dict(
                        max_error_rad=max_error,
                        max_target_lag_rad=max_target_lag,
                        max_bilateral_error_rad=max_bilateral_error,
                    )
                )
            )
            self.assertLess(
                max_target_lag,
                0.02,
                "Target limiter cannot keep up with a 1 rad/s trajectory",
            )
            self.assertLess(max_error, 0.12, "Actual joints accumulate trajectory lag")
            self.assertLess(
                max_bilateral_error, 0.01, "Symmetric arms lose synchronization"
            )
            for _ in range(simsvc.CTRL_HZ):
                session.act([row])
            np.testing.assert_allclose(
                session.data.qpos[session.qpos_ids][arms], row[arms], atol=0.02
            )
            self.assertFalse(session.done)


if __name__ == "__main__":
    unittest.main()
