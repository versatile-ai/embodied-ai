"""RoboDojo runtime mass semantics, not raw asset metadata mass."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "harness"))
import simsvc


class RigidMassTests(unittest.TestCase):
    def test_author_mass_defaults_and_invalid_input(self):
        for source, expected in (
            ({}, 0.5),
            ({"mass": 0}, 0.05),
            ({"mass": -1}, 0.05),
            ({"mass": 0.15}, 0.15),
            ({"mass": 22}, 0.5),
        ):
            self.assertEqual(simsvc.rigid_mass(source), expected)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                simsvc.rigid_mass({"mass": value})

    def test_layout_mass_is_capped_without_changing_source(self):
        layout = json.loads(
            (ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text()
        )
        original = json.dumps(layout, sort_keys=True)
        with patch.object(simsvc.mujoco, "Renderer", MagicMock()), patch.object(
            simsvc.Session, "capture"
        ):
            session = simsvc.Session(layout, "mass contract regression")
            try:
                self.assertAlmostEqual(session.model.body("bottle0").mass[0], 0.15)
                self.assertAlmostEqual(session.model.body("bottle3").mass[0], 0.5)
                self.assertEqual(json.dumps(layout, sort_keys=True), original)
            finally:
                session.done = True
                session.close()


if __name__ == "__main__":
    unittest.main()
