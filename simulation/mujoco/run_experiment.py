"""Run three fixed-layout pi0.5 trials; stop and preserve evidence on errors."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import client
from run_demo import run_session


def save(path: Path, report: dict) -> None:
    """Atomically publish progress so readers never see partial JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_trials(report_path: Path) -> dict:
    """Fresh scene per trial, no manual action fallback or automatic retries."""
    layout_path = client.ROOT / "layouts/put_bottles_into_dustbin_0.json"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "preflight",
        "policy": "pi0.5",
        "trials_requested": 3,
        "layout": layout_path.name,
        "layout_sha256": hashlib.sha256(layout_path.read_bytes()).hexdigest(),
        "max_steps": 700,
        "batch_steps": 3,
        "pause_seconds": 1,
        "camera": "default",
        "sim_endpoint": client.SIM,
        "policy_endpoint": client.PI,
        "seed": "Inference service seed is not configured by this client",
        "trials": [],
        "source_sha256": {
            name: hashlib.sha256((client.ROOT / name).read_bytes()).hexdigest()
            for name in ("client.py", "run_demo.py", "harness/simsvc.py")
        },
    }
    save(report_path, report)
    try:
        report["sim_health"] = client.http(client.SIM + "/health")
        report["policy_health"] = client.http(client.PI + "/health")
        if not report["sim_health"].get("ok") or not report["policy_health"].get("ok"):
            raise RuntimeError("Health check failed")
        current = client.http(client.SIM + "/status")
        if current.get("session_id") and not current.get("done"):
            raise RuntimeError("Another session is active; will not interrupt it")
        report["state"] = "running"
        for number in range(1, 4):
            trial = {"number": number, "state": "creating"}
            report["trials"].append(trial)
            save(report_path, report)
            created = client.http(
                client.SIM + "/session",
                {
                    "layout": json.loads(layout_path.read_text()),
                    "task": "put_bottles",
                    "instruction": "Pick up the bottles and throw them into the dustbin, using handover when needed.",
                },
            )
            trial.update(session_id=created["session_id"], state="running")
            save(report_path, report)
            print(
                json.dumps({"trial": number, "session_id": trial["session_id"]}),
                flush=True,
            )
            began = time.monotonic()
            result = run_session(trial["session_id"], steps=700, batch=3, pause=1)
            trial.update(result=result, wall_seconds=time.monotonic() - began)
            if result.get("termination") not in ("success", "step_limit"):
                trial["state"] = "environment_error"
                raise RuntimeError(
                    f'Invalid experiment termination: {result.get("termination")}'
                )
            trial["state"] = "completed"
            save(report_path, report)
        report["state"] = "completed"
        report["successes"] = sum(
            bool(t["result"]["success"]) for t in report["trials"]
        )
        report["success_rate"] = report["successes"] / 3
        save(report_path, report)
        return report
    except (Exception, KeyboardInterrupt) as exc:
        report["state"] = "interrupted"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if report["trials"] and report["trials"][-1]["state"] != "completed":
            report["trials"][-1]["state"] = "interrupted"
        save(report_path, report)
        raise


if __name__ == "__main__":
    directory = (
        client.ROOT
        / "runs"
        / (
            "experiment_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:6]
        )
    )
    directory.mkdir(parents=True, exist_ok=False)
    print(f'EXPERIMENT_REPORT {directory / "report.json"}', flush=True)
    run_trials(directory / "report.json")
