import copy
import unittest
from run_gpt6_hybrid import action, validate, validate_completion, policy_visible


class DecisionTests(unittest.TestCase):
    def test_object_ground_truth_removed_from_observation_and_history(self):
        original = {
            "observation": {"object_positions": {"bottle0": [1, 2, 3]}, "state": [1]},
            "history": [
                {"result": {"object_positions": {"dustbin": [0, 0, 0]}, "t": 3}}
            ],
        }
        filtered = policy_visible(original)
        self.assertNotIn("object_positions", filtered["observation"])
        self.assertNotIn("object_positions", filtered["history"][0]["result"])
        self.assertIn("object_positions", original["observation"])
        self.assertEqual(filtered["observation"]["state"], [1])

    def test_pilot_rejects_errors_and_early_stop(self):
        for termination, step in (
            ("physics_error", 6),
            ("eef_tracking_error", 3),
            ("stopped_by_operator", 3),
        ):
            with self.assertRaises(RuntimeError):
                validate_completion({"termination": termination, "t": step}, 6)
        validate_completion({"termination": "stopped_by_operator", "t": 6}, 6)

    def setUp(self):
        self.obs = {
            "observation_id": "s:0",
            "ee": {
                s: {
                    "xyz": [0.0, 0.0, 1.0],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "gripper": 1.0,
                }
                for s in ("left", "right")
            },
        }
        self.decision = {
            "observation_id": "s:0",
            "decision": "follow",
            "reason": "safe",
            "left_delta": [0.0, 0.0, 0.0],
            "right_delta": [0.0, 0.0, 0.0],
            "left_gripper": 1.0,
            "right_gripper": 1.0,
        }
        self.proposal = {
            "observation_id": "s:0",
            "actions": [[0.0] * 14 for _ in range(50)],
        }

    def test_follow_bounded(self):
        route, payload = action(self.decision, self.obs, self.proposal, 3)
        self.assertEqual(route, "act")
        self.assertEqual(len(payload["joints"]), 3)

    def test_stale_rejected(self):
        self.decision["observation_id"] = "s:1"
        with self.assertRaises(ValueError):
            validate(self.decision, self.obs)

    def test_correction_preserves_orientation_and_input(self):
        original = copy.deepcopy(self.obs)
        self.decision.update(decision="correct", left_delta=[0.0, 0.0, 0.018])
        route, payload = action(self.decision, self.obs, self.proposal, 2)
        self.assertEqual(route, "eef")
        self.assertEqual(payload["goals"]["left"]["xyz"], [0.0, 0.0, 1.018])
        self.assertEqual(self.obs, original)
        self.assertEqual(payload["steps"], 2)

    def test_bad_deltas_rejected(self):
        for value in ([0.019, 0, 0], [0.018, 0.018, 0], [float("nan"), 0, 0], [0, 0]):
            self.decision["left_delta"] = value
            with self.assertRaises(ValueError):
                validate(self.decision, self.obs)

    def test_stop_and_replan_cannot_execute(self):
        for value in ("stop", "replan"):
            self.decision["decision"] = value
            with self.assertRaises(ValueError):
                action(self.decision, self.obs, self.proposal, 3)

    def test_stale_proposal(self):
        self.proposal["observation_id"] = "s:2"
        with self.assertRaises(ValueError):
            action(self.decision, self.obs, self.proposal, 3)


if __name__ == "__main__":
    unittest.main()
