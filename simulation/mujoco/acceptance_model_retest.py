"""Supplemental model checks after restoring the tunnel; no formal trials."""

import json
from datetime import datetime
from unittest.mock import patch
import client
import run_gpt6_hybrid as hybrid


def main():
    root = (
        client.ROOT
        / "runs"
        / ("model_retest_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    root.mkdir()
    report = {"state": "running", "tests": []}

    def checkpoint():
        hybrid.save(root / "report.json", report)

    def finish():
        s = client.http(client.SIM + "/status")
        if s.get("session_id") and not s["done"]:
            client.execute(
                s["session_id"], s["t"], "finish", {}, "Acceptance-only cleanup"
            )

    initial = client.http(client.SIM + "/status")
    assert (
        initial.get("done") or initial.get("t") == 0
    ), "Do not interrupt another active task"
    finish()
    print("MODEL_RETEST " + str(root), flush=True)
    try:
        assert client.http(client.PI + "/health")["ok"]
        trial = {"number": "gpt_timeout"}
        report["tests"].append(trial)
        directory = root / "gpt_timeout"
        directory.mkdir()
        with patch.object(
            hybrid, "brain", side_effect=TimeoutError("Injected GPT timeout")
        ) as brain:
            try:
                hybrid.episode(directory, trial, 6, checkpoint)
            except TimeoutError as exc:
                assert str(exc) == "Injected GPT timeout" and brain.call_count == 1
                status = client.http(client.SIM + "/status")
                assert status["session_id"] == trial["session_id"] and status["t"] == 0
                trial.update(state="passed", caught=str(exc))
            else:
                raise AssertionError("GPT timeout did not stop controller")
        finish()
        checkpoint()
        trial = {"number": "live_six_steps"}
        report["tests"].append(trial)
        directory = root / "pilot"
        directory.mkdir()
        hybrid.episode(directory, trial, 6, checkpoint)
        report["state"] = "passed"
    except Exception as exc:
        report.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        checkpoint()
        finish()
        result = client.http(
            client.SIM + "/session",
            {
                "layout": json.loads(
                    (
                        client.ROOT / "layouts/put_bottles_into_dustbin_0.json"
                    ).read_text()
                ),
                "task": "put_bottles",
                "instruction": "Initial scene after acceptance checks; no experiment running",
            },
        )
        report["restored_session"] = result["session_id"]
        checkpoint()
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
