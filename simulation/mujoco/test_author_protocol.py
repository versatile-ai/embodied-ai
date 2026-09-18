import copy
import unittest
from unittest.mock import patch
import numpy as np
from author_protocol import RobotFK, validate, policy_observation


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fk = RobotFK()

    def setUp(self):
        self.row = [0.0] * 14
        self.obs = dict(
            observation_id="s:0",
            t=1,
            state=self.row,
            current_eef=self.fk.poses(self.row),
        )
        self.d = dict(
            observation_id="s:0",
            mode="student",
            steps=15,
            reason="Visible alignment",
            assessment=dict(
                task_progress=dict(
                    verified_completed=[],
                    currently_attempting="approach",
                    remaining=["grasp"],
                ),
                current_subgoal="approach",
                execution_status="uncertain",
                execution_evidence="occluded",
                expected_next_intent="approach",
                predicted_next_intent="approach",
                intent_status="aligned",
                intent_evidence="FK",
            ),
            edit={
                s: dict(
                    delta_position=[0, 0, 0],
                    delta_rotation_vector=[0, 0, 0],
                    gripper="keep",
                )
                for s in ("left", "right")
            },
            target={
                s: dict(
                    position=p["position"],
                    quaternion_wxyz=p["quaternion_wxyz"],
                    gripper_closed=False,
                )
                for s, p in self.obs["current_eef"].items()
            },
        )

    def test_dynamic_student_limits(self):
        for n in (1, 3, 15):
            self.d["steps"] = n
            validate(self.d, self.obs)
        for n in (0, 16, True, 1.5):
            self.d["steps"] = n
            with self.assertRaises(Exception):
                validate(self.d, self.obs)

    def test_takeover_gate(self):
        self.d.update(mode="eef", steps=5)
        with self.assertRaises(ValueError):
            validate(self.d, self.obs)
        self.d["assessment"]["execution_status"] = "failed"
        validate(self.d, self.obs)
        self.d["steps"] = 6
        with self.assertRaises(ValueError):
            validate(self.d, self.obs)

    def test_stale_and_bounds(self):
        self.d["observation_id"] = "s:1"
        with self.assertRaises(ValueError):
            validate(self.d, self.obs)
        self.d.update(observation_id="s:0", mode="edit", steps=1)
        self.d["assessment"]["intent_status"] = "misaligned"
        self.d["edit"]["left"]["delta_position"] = [0.051, 0, 0]
        with self.assertRaises(ValueError):
            validate(self.d, self.obs)

    def test_no_ground_truth_or_reward_inputs(self):
        raw = dict(
            self.obs,
            score=10,
            object_positions={"bottle": [1, 2, 3]},
            grasped={"left": "bottle"},
        )
        visible = policy_observation(raw)
        self.assertEqual(set(visible), {"observation_id", "t", "state"})
        self.assertIn("score", raw)

    def test_zero_edit_preserves_student_not_hold(self):
        self.d.update(mode="edit", steps=1)
        student = [0.01] * 14
        actual = self.fk.correction(self.d, self.obs, student)
        np.testing.assert_allclose(actual, student)

    def test_eef_joint_increment_bounded(self):
        self.d.update(mode="eef", steps=1)
        self.d["target"]["left"]["position"][2] += 0.04
        before = copy.deepcopy(self.obs)
        actual = self.fk.correction(self.d, self.obs, self.row)
        self.assertLessEqual(
            max(abs(actual[i]) for i in list(range(6)) + list(range(7, 13))), 0.050001
        )
        self.assertEqual(self.obs, before)

    def test_fk_does_not_advance_physics(self):
        with patch("mujoco.mj_step", side_effect=AssertionError("No physics allowed")):
            self.fk.poses(self.row)

    def test_assessment_required_even_when_student_is_accepted(self):
        self.d["assessment"]["execution_evidence"] = " "
        with self.assertRaises(ValueError):
            validate(self.d, self.obs)

    def test_not_started_only_at_step_zero(self):
        self.obs["t"] = 0
        with self.assertRaises(ValueError):
            validate(self.d, self.obs)
        self.d["assessment"]["execution_status"] = "not_started"
        validate(self.d, self.obs)
        self.obs["t"] = 1
        with self.assertRaises(ValueError):
            validate(self.d, self.obs)

    def test_edit_ramps_over_prefix(self):
        self.d.update(mode="edit", steps=5)
        self.d["edit"]["left"]["delta_position"] = [0, 0, 0.01]
        self.d["edit"]["left"]["gripper"] = "closed"
        initial_z = self.obs["current_eef"]["left"]["position"][2]
        for index in (0, 4):
            with patch("author_protocol.solve_step", return_value=self.row) as solve:
                self.fk.correction(self.d, self.obs, self.row, prefix_index=index)
            goal = solve.call_args.args[2]["left"]
            self.assertAlmostEqual(goal["xyz"][2] - initial_z, 0.01 * (index + 1) / 5)
            self.assertEqual(goal["gripper"], 0)


if __name__ == "__main__":
    unittest.main()
