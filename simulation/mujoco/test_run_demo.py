"""Fast, network-free coverage for the bounded demo controller."""

import unittest
from unittest.mock import patch

import run_demo


class DemoTests(unittest.TestCase):
    def observation(self, step, done=False):
        return {
            "t": step,
            "done": done,
            "remaining_steps": 700 - step,
            "observation_id": f"s:{step}",
        }, {}

    def test_bounded_batches_and_fresh_inference(self):
        with patch.object(
            run_demo.client,
            "observe",
            side_effect=[self.observation(3), self.observation(6), self.observation(8)],
        ), patch.object(
            run_demo.client,
            "infer",
            side_effect=lambda sid, obs: (
                None,
                {"observation_id": obs["observation_id"], "actions": [[0] * 14] * 50},
            ),
        ) as infer, patch.object(
            run_demo.client,
            "execute",
            side_effect=[
                {"t": 6, "score": 0, "done": False},
                {"t": 8, "score": 0, "done": False},
                {"videos": ["test.mp4"]},
            ],
        ) as execute:
            result = run_demo.run_session("s", steps=5, pause=0)
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(
            [len(c.args[3]["joints"]) for c in execute.call_args_list[:2]], [3, 2]
        )
        self.assertEqual(execute.call_args.args[1:3], (8, "finish"))
        self.assertEqual(result["videos"], ["test.mp4"])

    def test_done_session_does_not_send_actions(self):
        with patch.object(
            run_demo.client, "observe", return_value=self.observation(700, True)
        ), patch.object(run_demo.client, "infer") as infer, patch.object(
            run_demo.client, "execute", return_value={}
        ) as execute:
            run_demo.run_session("s", pause=0)
        infer.assert_not_called()
        self.assertEqual(execute.call_args.args[2], "finish")

    def test_timeout_never_retries_or_finishes(self):
        with patch.object(
            run_demo.client, "observe", return_value=self.observation(0)
        ), patch.object(
            run_demo.client,
            "infer",
            return_value=(None, {"observation_id": "s:0", "actions": [[0] * 14] * 50}),
        ), patch.object(
            run_demo.client, "execute", side_effect=TimeoutError
        ) as execute:
            with self.assertRaises(TimeoutError):
                run_demo.run_session("s", pause=0)
        self.assertEqual(execute.call_count, 1)

    def test_stale_proposal_rejected(self):
        with patch.object(
            run_demo.client, "observe", return_value=self.observation(1)
        ), patch.object(
            run_demo.client, "infer", return_value=(None, {"observation_id": "s:0"})
        ), patch.object(
            run_demo.client, "execute"
        ) as execute:
            with self.assertRaises(ValueError):
                run_demo.run_session("s", pause=0)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
