"""Bounded acceptance checks; never starts the three-trial experiment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from unittest.mock import patch

import client
import run_gpt6_hybrid as hybrid


def main():
    root = (
        client.ROOT
        / "runs"
        / ("acceptance_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    root.mkdir()
    report = {"state": "running", "checks": [], "sessions": []}

    def checkpoint():
        hybrid.save(root / "report.json", report)

    def check(name, fn):
        entry = {"name": name, "state": "running"}
        report["checks"].append(entry)
        checkpoint()
        began = time.monotonic()
        try:
            entry["result"] = fn()
            entry["state"] = "passed"
        except Exception as exc:
            entry.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        entry["seconds"] = time.monotonic() - began
        checkpoint()
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    def command(name, args, timeout=900):
        def run():
            with (root / (name + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, *args],
                    cwd=client.ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            if result.returncode:
                raise RuntimeError(f"exit={result.returncode}; see {name}.log")
            return {"log": name + ".log"}

        check(name, run)

    def finish_current():
        status = client.http(client.SIM + "/status")
        if status.get("session_id") and not status["done"]:
            return client.execute(
                status["session_id"],
                status["t"],
                "finish",
                {},
                "Acceptance case cleanup, no action replay",
            )
        return status

    def start():
        result = client.http(
            client.SIM + "/session",
            {
                "layout": json.loads(
                    (
                        client.ROOT / "layouts/put_bottles_into_dustbin_0.json"
                    ).read_text()
                ),
                "instruction": "Acceptance test only",
                "task": "put_bottles",
            },
        )
        report["sessions"].append(result["session_id"])
        return result

    print("AUDIT_REPORT " + str(root / "report.json"), flush=True)
    report["sim_health"] = client.http(client.SIM + "/health")
    assert report["sim_health"]["grasp_assist"] is False
    status = client.http(client.SIM + "/status")
    assert (
        not status.get("session_id") or status["done"]
    ), "Another active session exists"
    checkpoint()
    command(
        "unit_and_physics",
        [
            "-m",
            "unittest",
            "test_service",
            "test_gripper_stability",
            "test_client",
            "test_run_demo",
            "test_experiment",
            "test_gpt6_hybrid",
            "-v",
        ],
    )
    command("http_protocol", ["verify_http.py"])
    finish_current()
    for side in ("left", "right"):
        command("visual_pick_place_" + side, ["test_pick_place.py", "--side", side])
        result = finish_current()
        report["sessions"].append(result["session_id"])
        checkpoint()

    def cycles():
        samples = []
        for _ in range(6):
            created = start()
            sid = created["session_id"]
            reset = json.loads(
                (client.ROOT / "runs" / sid / "reset_check.json").read_text()
            )
            assert reset["passed"] and created["t"] == 0 and created["score"] == 0
            obs, _ = client.observe(sid)
            result = client.execute(
                sid, 0, "act", {"joints": [obs["state"]]}, "One-step lifecycle check"
            )
            assert result["t"] == 1
            finish_current()
            final = client.http(f"{client.SIM}/session/{sid}/state")
            assert final["done"] and final["t"] == 1
            ps = "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'harness[/\\\\]simsvc.py' } | ForEach-Object { Get-Process -Id $_.ProcessId } | Measure-Object -Property PrivateMemorySize64 -Sum | Select-Object -ExpandProperty Sum"
            memory = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", ps], text=True
            ).strip()
            samples.append({"session_id": sid, "private_bytes": int(memory)})
        growth = samples[-1]["private_bytes"] - samples[1]["private_bytes"]
        hybrid.save(
            root / "memory_cycles.json",
            {"samples": samples, "post_warmup_growth": growth},
        )
        assert growth < 256 * 1024**2, f"Post-warmup private memory grew {growth} bytes"
        return {
            "cycles": samples,
            "note": "Short 6-cycle screen, not a long-duration leak proof",
        }

    check("reset_and_memory_6_cycles", cycles)
    finish_current()

    def fault(kind):
        directory = root / ("fault_" + kind)
        directory.mkdir()
        trial = {"number": kind}
        original = client.http
        infer_calls = []

        def failing_http(url, payload=None):
            if url == client.PI + "/infer":
                infer_calls.append(url)
                raise TimeoutError("Injected inference timeout")
            return original(url, payload)

        if kind == "stop":
            (directory / "STOP").touch()
        try:
            if kind == "timeout":
                with patch.object(client, "http", failing_http):
                    hybrid.episode(
                        directory,
                        trial,
                        6,
                        lambda: hybrid.save(directory / "trial.json", trial),
                    )
            elif kind == "gpt_timeout":
                with patch.object(
                    hybrid, "brain", side_effect=TimeoutError("Injected GPT timeout")
                ) as brain_mock:
                    hybrid.episode(
                        directory,
                        trial,
                        6,
                        lambda: hybrid.save(directory / "trial.json", trial),
                    )
            elif kind == "disconnect":
                with patch.object(client, "PI", "http://127.0.0.1:1"):
                    hybrid.episode(
                        directory,
                        trial,
                        6,
                        lambda: hybrid.save(directory / "trial.json", trial),
                    )
            else:
                hybrid.episode(
                    directory,
                    trial,
                    6,
                    lambda: hybrid.save(directory / "trial.json", trial),
                )
        except (TimeoutError, RuntimeError, OSError) as exc:
            status = original(client.SIM + "/status")
            assert status["session_id"] == trial["session_id"] and status["t"] == 0
            if kind == "timeout":
                assert len(infer_calls) == 1
                assert str(exc) == "Injected inference timeout"
            elif kind == "gpt_timeout":
                assert brain_mock.call_count == 1 and str(exc) == "Injected GPT timeout"
            elif kind == "stop":
                assert str(exc) == "Operator STOP file detected"
            elif kind == "disconnect":
                assert "10061" in str(exc) or "refused" in str(exc).lower()
            result = {
                "caught": str(exc),
                "step": status["t"],
                "session_id": status["session_id"],
                "injection": kind,
            }
            hybrid.save(directory / "result.json", result)
            finish_current()
            return result
        raise AssertionError("Fault did not stop execution")

    for kind in ("stop", "timeout", "disconnect", "gpt_timeout"):
        check("fault_" + kind, lambda kind=kind: fault(kind))
        finish_current()

    def pilot():
        directory = root / "hybrid_pilot"
        directory.mkdir()
        trial = {"number": "acceptance-pilot"}
        hybrid.episode(
            directory, trial, 6, lambda: hybrid.save(directory / "trial.json", trial)
        )
        report["sessions"].append(trial["session_id"])
        return trial

    check("live_gpt6_pi05_6_steps", pilot)
    finish_current()

    def videos():
        import imageio_ffmpeg

        records = []
        for sid in report["sessions"]:
            for camera in client.CAM:
                path = client.ROOT / "runs" / sid / (camera + ".mp4")
                assert (
                    path.is_file() and path.stat().st_size > 0
                ), f"Missing video {path}"
                result = subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        "-f",
                        "null",
                        "-",
                    ],
                    capture_output=True,
                    timeout=90,
                )
                assert result.returncode == 0, result.stderr.decode(errors="replace")
                records.append(str(path))
        assert records
        return {"decoded": records}

    check("decode_videos", videos)
    report["restored_initial_session"] = start()["session_id"]
    report["state"] = (
        "completed_with_failures"
        if any(c["state"] == "failed" for c in report["checks"])
        else "passed"
    )
    checkpoint()
    print("AUDIT_COMPLETE " + str(root), flush=True)


if __name__ == "__main__":
    main()
