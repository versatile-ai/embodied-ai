"""Experiment bookkeeping tests, without simulation or network calls."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_experiment as experiment


class TrialsTest(unittest.TestCase):
    def test_three_fresh_trials(self):
        def http(url, payload=None):
            if url.endswith("/health"):
                return {"ok": True}
            if url.endswith("/status"):
                return {"session_id": None}
            self.assertNotIn("overview_yaw", payload)
            return {"session_id": "test"}

        with tempfile.TemporaryDirectory() as directory, patch.object(
            experiment.client, "http", side_effect=http
        ), patch.object(
            experiment,
            "run_session",
            return_value={
                "termination": "step_limit",
                "success": False,
                "score": 0,
                "t": 700,
            },
        ) as run:
            path = Path(directory) / "report.json"
            result = experiment.run_trials(path)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(result["state"], "completed")
            self.assertEqual(json.loads(path.read_text())["success_rate"], 0)

    def test_error_stops_without_creating_next_trial(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            experiment.client,
            "http",
            side_effect=[
                {"ok": True},
                {"ok": True},
                {"session_id": None},
                {"session_id": "one"},
            ],
        ) as http, patch.object(experiment, "run_session", side_effect=TimeoutError):
            path = Path(directory) / "report.json"
            with self.assertRaises(TimeoutError):
                experiment.run_trials(path)
            report = json.loads(path.read_text())
            self.assertEqual(report["state"], "interrupted")
            self.assertEqual(len(report["trials"]), 1)
            self.assertEqual(http.call_count, 4)
            self.assertNotIn("success_rate", report)


if __name__ == "__main__":
    unittest.main()
