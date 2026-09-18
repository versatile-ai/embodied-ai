"""Native ARX gripper semantics, keeping physics and actual-state diagnostics."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))
import simsvc


class GripperContractTest(unittest.TestCase):
    def setUp(self):
        self.renderer = patch.object(simsvc.mujoco, "Renderer", MagicMock())
        self.capture = patch.object(simsvc.Session, "capture")
        self.renderer.start()
        self.capture.start()
        self.addCleanup(self.renderer.stop)
        self.addCleanup(self.capture.stop)
        layout = json.loads(
            (ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text()
        )
        self.s = simsvc.Session(layout, "gripper contract")

    def test_both_fingers_start_open(self):
        for side in ("left", "right"):
            for joint in (7, 8):
                qi = self.s.model.joint(f"{side}_joint{joint}").qposadr[0]
                self.assertAlmostEqual(self.s.data.qpos[qi], simsvc.GRIP_MAX)
        self.assertTrue(self.s.reset_check["passed"])

    def test_intermediate_commands_are_continuous(self):
        row = self.s.state14()
        for opening in (0.0, 0.55, 0.95, 0.3):
            row[6] = opening
            self.s.act([row])
            self.assertAlmostEqual(self.s.commanded_grip["left"], opening)

    def test_policy_grip_is_command_but_measured_state_is_retained(self):
        row = self.s.state14()
        row[6] = 0.0
        self.s.act([row])
        self.assertEqual(self.s.policy_state14()[6], 0.0)
        self.assertGreater(
            self.s.state14()[6], 0.5
        )  # Physical closing is not instantaneous.
        self.s.done = True
        before = self.s.policy_state14()
        self.s.close()
        self.assertEqual(self.s.policy_state14(), before)


if __name__ == "__main__":
    unittest.main()
