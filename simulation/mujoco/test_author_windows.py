import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import author_windows
from author_tool_log import session_images
from run_author_hybrid import Teacher, audit_events


class WindowsBackendTests(unittest.TestCase):
    def test_http_transport_preserves_official_auth(self):
        args = author_windows.transport_config()
        self.assertIn("model_providers.author_http.supports_websockets=false", args)
        self.assertIn("model_providers.author_http.requires_openai_auth=true", args)
        self.assertIn(
            'model_providers.author_http.base_url="https://chatgpt.com/backend-api/codex"',
            args,
        )

    def test_native_command_and_paths(self):
        self.assertIn(
            'default_permissions=":danger-full-access"',
            author_windows.permissions(Path(".")),
        )
        self.assertNotIn("--ignore-rules", author_windows.permissions(Path(".")))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "image.png"
            path.touch()
            self.assertTrue(author_windows.canonical_image_allowed(str(path), root))
            self.assertFalse(
                author_windows.canonical_image_allowed(str(root.parent), root)
            )
            self.assertNotIn("/mnt/", author_windows.linux_path(root))
            with patch(
                "author_windows.shutil.which", return_value="C:/tools/codex.exe"
            ):
                cmd = author_windows.wsl_command(author_windows.prefix() + ["exec"])
                self.assertEqual(cmd, ["C:/tools/codex.exe", "exec"])
            audit_events(
                [
                    dict(
                        type="item.completed",
                        item=dict(type="image_view", path=str(path)),
                    )
                ],
                root,
                author_windows,
            )

    def test_teacher_windows_adapter(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "author_windows.probe", return_value={"backend": "windows"}
        ):
            with patch("run_author_hybrid.tempfile.mkdtemp", return_value=folder):
                teacher = Teacher(Path(folder) / "record", "windows", "medium")
                self.assertIs(teacher.adapter, author_windows)
                self.assertEqual(teacher.backend, "windows")
                self.assertEqual(teacher.effort, "medium")
        with self.assertRaises(ValueError):
            Teacher(Path("unused"), "invalid")

    def test_native_transcript_lookup(self):
        thread = "12345678-1234-1234-1234-123456789abc"
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "sessions/2026/09/17"
            base.mkdir(parents=True)
            (base / f"rollout-{thread}.jsonl").write_text(
                json.dumps(dict(type="turn_context", payload=dict(turn_id="turn1")))
                + "\n"
            )
            with patch.dict("os.environ", {"CODEX_HOME": folder}):
                self.assertEqual(session_images(thread, backend="windows"), [])


if __name__ == "__main__":
    unittest.main()
