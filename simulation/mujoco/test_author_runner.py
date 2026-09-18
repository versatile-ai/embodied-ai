import json
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import client
from run_author_hybrid import Teacher, audit_events, check_stop
import test_author_protocol


class RunnerTests(unittest.TestCase):
    def test_tool_audit_fails_closed(self):
        root = Path(tempfile.gettempdir()) / "teacher_test"
        audit_events(
            [{"type": "item.completed", "item": {"type": "agent_message"}}], root
        )
        audit_events(
            [{"type": "item.completed", "item": {"type": "command_execution"}}], root
        )
        for kind in ("mcp_tool_call", "unknown"):
            with self.assertRaises(RuntimeError):
                audit_events([{"type": "item.completed", "item": {"type": kind}}], root)
        with self.assertRaises(RuntimeError):
            audit_events(
                [
                    {
                        "type": "item.completed",
                        "item": {
                            "type": "image_view",
                            "path": str(root.parent / "secret.png"),
                        },
                    }
                ],
                root,
            )

    def test_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            check_stop(root)
            (root / "STOP").touch()
            with self.assertRaises(RuntimeError):
                check_stop(root)

    def test_persistent_resume_and_mismatch(self):
        test_author_protocol.ProtocolTests.setUpClass()
        fixture = test_author_protocol.ProtocolTests()
        fixture.setUp()
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            for cam in client.CAM:
                (root / (cam + ".png")).write_bytes(b"mock image")
            thread = "test-thread"
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                output = Path(cmd[cmd.index("-o") + 1])
                output.write_text(json.dumps(fixture.d), encoding="utf-8")
                kwargs["stdout"].write(
                    json.dumps(dict(type="thread.started", thread_id=thread)) + "\n"
                )
                kwargs["stdout"].write(json.dumps(dict(type="turn.completed")) + "\n")

                class Result:
                    returncode = 0

                return Result()

            with patch(
                "run_author_hybrid.tempfile.mkdtemp", return_value=outside
            ), patch(
                "run_author_hybrid.sandbox.probe", return_value={"ok": True}
            ), patch(
                "run_author_hybrid.sandbox.wsl_command", side_effect=lambda x: x
            ), patch(
                "run_author_hybrid.sandbox.linux_path", side_effect=lambda x: str(x)
            ), patch(
                "run_author_hybrid.subprocess.run", side_effect=fake_run
            ):
                image_patch = patch("run_author_hybrid.session_images", return_value=[])
                image_patch.start()
                self.addCleanup(image_patch.stop)
                teacher = Teacher(root / "record")
                for i in range(2):
                    directory = root / f"{i:04d}"
                    directory.mkdir()
                    teacher.decide(directory, fixture.obs, {}, root, None, {})
                self.assertNotIn("resume", calls[0])
                self.assertIn("resume", calls[1])
                self.assertIn("test-thread", calls[1])
                self.assertNotIn("--ephemeral", calls[0])
                thread = "wrong-thread"
                directory = root / "0002"
                directory.mkdir()
                with self.assertRaises(RuntimeError):
                    teacher.decide(directory, fixture.obs, {}, root, None, {})

    def test_timeout_before_activity_retries_without_stale_output(self):
        from run_author_hybrid import run_teacher_call

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calls = []

            def fake_run(cmd, **kwargs):
                calls.append(cmd)
                if len(calls) == 1:
                    (root / "brain_output.json").write_text("")
                    raise subprocess.TimeoutExpired(cmd, 300)
                self.assertFalse((root / "brain_output.json").exists())
                return subprocess.CompletedProcess(cmd, 0)

            with patch("run_author_hybrid.subprocess.run", side_effect=fake_run):
                result = run_teacher_call(
                    ["mock"], "original", root, retry_guard=lambda events: True
                )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(len(calls), 2)
            self.assertTrue((root / "attempt_1" / "brain_output.json").exists())

    def test_timeout_after_tool_activity_never_retries(self):
        from run_author_hybrid import run_teacher_call

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def fake_run(cmd, **kwargs):
                kwargs["stdout"].write(
                    json.dumps(
                        {"type": "item.started", "item": {"type": "command_execution"}}
                    )
                    + "\n"
                )
                raise subprocess.TimeoutExpired(cmd, 300)

            with patch(
                "run_author_hybrid.subprocess.run", side_effect=fake_run
            ) as launch:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_teacher_call(
                        ["mock"], "original", root, retry_guard=lambda events: True
                    )
            self.assertEqual(launch.call_count, 1)

    def test_timeout_attempt_budget_and_stop(self):
        from run_author_hybrid import run_teacher_call

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch(
                "run_author_hybrid.subprocess.run",
                side_effect=subprocess.TimeoutExpired("mock", 300),
            ) as launch:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_teacher_call(
                        ["mock"], "original", root, retry_guard=lambda events: True
                    )
            self.assertEqual(launch.call_count, 3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "STOP").touch()
            with patch("run_author_hybrid.subprocess.run") as launch:
                with self.assertRaisesRegex(RuntimeError, "Operator STOP"):
                    run_teacher_call(["mock"], "original", root)
            launch.assert_not_called()

    def test_partial_output_prevents_retry(self):
        from run_author_hybrid import run_teacher_call

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def fake_run(cmd, **kwargs):
                (root / "brain_output.json").write_text('{"mode":')
                raise subprocess.TimeoutExpired(cmd, 300)

            with patch(
                "run_author_hybrid.subprocess.run", side_effect=fake_run
            ) as launch:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_teacher_call(
                        ["mock"], "original", root, retry_guard=lambda events: True
                    )
            self.assertEqual(launch.call_count, 1)
            self.assertEqual((root / "brain_output.json").read_text(), '{"mode":')

    def test_native_transcript_gate(self):
        from author_tool_log import startup_only

        rows = [{"type": "turn_context", "payload": {}}]
        with patch("author_tool_log.session_rows", return_value=rows):
            self.assertTrue(startup_only("test"))
            rows.append(
                {"type": "response_item", "payload": {"type": "custom_tool_call"}}
            )
            self.assertFalse(startup_only("test"))
        with patch("author_tool_log.session_rows", side_effect=RuntimeError("missing")):
            self.assertFalse(startup_only("test"))


if __name__ == "__main__":
    unittest.main()
