"""Three audited author-style hybrid trials on our MuJoCo adapter."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Callable

import numpy as np
import client
import author_sandbox as sandbox
from author_tool_log import session_images, startup_only
from author_protocol import SCHEMA, PROMPT, RobotFK, policy_observation, validate
from run_gpt6_hybrid import save, stamp, report_html, validate_completion

MODEL = "gpt-6-astra"
AUDIT_TIMEOUT_SECONDS = 300
AUDIT_MAX_ATTEMPTS = 3


def run_teacher_call(
    cmd: list[str],
    prompt: str,
    directory: Path,
    on_status: Callable[[dict], None] | None = None,
    retry_guard: Callable[[list[dict]], bool] | None = None,
) -> subprocess.CompletedProcess:
    """Retry only silent timeouts; never retry a decision or any tool activity."""
    began = time.monotonic()
    if (directory / "brain_output.json").exists():
        raise RuntimeError("Refusing to overwrite existing teacher output")
    for attempt in range(1, AUDIT_MAX_ATTEMPTS + 1):
        for parent in (directory, *directory.parents):
            check_stop(parent)
        finished = threading.Event()

        def publish(state: str) -> None:
            status = dict(
                state=state,
                attempt=attempt,
                max_attempts=AUDIT_MAX_ATTEMPTS,
                elapsed_seconds=round(time.monotonic() - began, 1),
                timeout_seconds=AUDIT_TIMEOUT_SECONDS,
                updated_utc=stamp(),
            )
            save(directory / "audit_status.json", status)
            if on_status:
                on_status(status)
            print("TEACHER_STATUS " + json.dumps(status), flush=True)

        def heartbeat() -> None:
            while not finished.wait(15):
                publish("waiting_for_teacher")

        publish("waiting_for_teacher")
        monitor = threading.Thread(target=heartbeat, daemon=True)
        monitor.start()
        try:
            with (directory / "brain_events.jsonl").open(
                "w", encoding="utf-8"
            ) as out, (directory / "brain_stderr.txt").open(
                "w", encoding="utf-8"
            ) as err:
                result = subprocess.run(
                    cmd,
                    input=prompt,
                    encoding="utf-8",
                    stdout=out,
                    stderr=err,
                    timeout=AUDIT_TIMEOUT_SECONDS,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
        except subprocess.TimeoutExpired:
            finished.set()
            monitor.join()
            # Only a strictly known startup-only transcript is safe to repeat.
            # Parse failures, tool calls, reasoning and partial decisions fail closed.
            events = [
                json.loads(line)
                for line in (directory / "brain_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            silent = all(
                e.get("type") in ("thread.started", "turn.started")
                or (
                    e.get("type") == "item.completed"
                    and e.get("item", {}).get("type") == "error"
                    and e["item"]
                    .get("message", "")
                    .startswith(
                        "Skill descriptions were shortened to fit the skills context budget."
                    )
                )
                for e in events
            )
            output = directory / "brain_output.json"
            retry = (
                silent
                and attempt < AUDIT_MAX_ATTEMPTS
                and not (output.exists() and output.stat().st_size)
                and retry_guard is not None
                and retry_guard(events)
            )
            publish("retrying_silent_timeout" if retry else "timeout_stopped")
            archive = directory / f"attempt_{attempt}"
            archive.mkdir()
            for name in (
                "brain_events.jsonl",
                "brain_stderr.txt",
                "brain_output.json",
                "audit_status.json",
            ):
                path = directory / name
                if path.exists():
                    shutil.copy2(path, archive / name)
            if not retry:
                raise
            (directory / "brain_output.json").unlink(missing_ok=True)
            continue
        finally:
            finished.set()
            monitor.join()
        publish("response_received" if result.returncode == 0 else "process_failed")
        return result
    raise RuntimeError("Teacher attempt budget exhausted")


def check_stop(root: Path) -> None:
    if (root / "STOP").exists():
        raise RuntimeError("Operator STOP")


def audit_events(events: list[dict], root: Path, adapter=sandbox) -> None:
    """Audit tool surfaces; Windows does not claim verified read confinement."""
    for event in events:
        if event.get("type") not in ("item.started", "item.updated", "item.completed"):
            continue
        item = event.get("item", {})
        kind = item.get("type")
        if kind in (
            "agent_message",
            "reasoning",
            "error",
            "command_execution",
            "file_change",
            "todo_list",
        ):
            continue
        if kind == "image_view":
            path = item.get("path")
            if path and adapter.canonical_image_allowed(str(path), root):
                continue
        raise RuntimeError(
            f"Unauditable teacher tool event {kind}; trial invalidated before control"
        )


class Teacher:
    """One persistent Codex conversation per episode, never resume --last."""

    def __init__(self, root: Path, backend: str = "wsl", effort: str = "xhigh") -> None:
        if effort not in ("medium", "high", "xhigh"):
            raise ValueError("Unsupported reasoning effort")
        self.effort = effort
        self.backend = backend
        self.adapter = sandbox
        if backend == "windows":
            import author_windows

            self.adapter = author_windows
        elif backend != "wsl":
            raise ValueError("Unknown teacher backend")
        root.mkdir(parents=True)
        self.root = Path(tempfile.mkdtemp(prefix="robodojo_teacher_"))
        save(
            root / "workspace.json", dict(path=str(self.root), retained_for_audit=True)
        )
        self.thread_id = None
        self.on_status = None
        self.probe_result = self.adapter.probe(self.root, Path(__file__).resolve())
        save(root / "sandbox_probe.json", self.probe_result)

    def decide(
        self,
        directory: Path,
        observation: dict,
        proposal: dict,
        image_dir: Path,
        last_execution: dict | None,
        initial: dict,
    ) -> dict:
        sandbox = self.adapter
        evidence = self.root / "evidence" / directory.name
        evidence.mkdir(parents=True)
        for cam in client.CAM:
            shutil.copy2(image_dir / (cam + ".png"), evidence / (cam + ".png"))
        context = dict(
            observation=observation,
            proposal=proposal,
            last_execution=last_execution,
            initial_eef=initial,
            image_directory=sandbox.linux_path(evidence),
            python_interpreter=sandbox.PYTHON,
        )
        save(evidence / "context.json", context)
        save(directory / "schema.json", SCHEMA)
        policy_prompt = PROMPT.replace("GPT-6/xhigh", "GPT-6/" + self.effort)
        if self.backend == "windows":
            policy_prompt = policy_prompt.replace(
                "this isolated workspace", "this analysis workspace"
            )
            policy_prompt = policy_prompt.replace(
                "Evidence is read-only.",
                "Do not modify supplied evidence (policy constraint).",
            )
            policy_prompt = policy_prompt.replace(
                "Commands run in a Linux sandbox;\noutside reads and all command networking are blocked.",
                "Commands run natively on Windows in PowerShell; use Windows paths and the supplied Python interpreter. "
                "This is a functional run without verified read isolation; "
                "do not access unrelated files or credentials.",
            )
        prompt = policy_prompt + "\n" + json.dumps(context, ensure_ascii=False)
        (directory / "brain_input.txt").write_text(prompt, encoding="utf-8")
        cmd = sandbox.prefix() + [
            "exec",
            "--ignore-user-config",
            "--skip-git-repo-check",
            "-C",
            sandbox.linux_path(self.root),
            *sandbox.permissions(self.root),
            "-c",
            "model_reasoning_effort=" + self.effort,
        ]
        if self.backend == "windows":
            cmd += sandbox.transport_config()
        if self.thread_id:
            cmd += ["resume"]
        cmd += [
            "-m",
            MODEL,
            "--json",
            "--output-schema",
            sandbox.linux_path(directory / "schema.json"),
            "-o",
            sandbox.linux_path(directory / "brain_output.json"),
        ]
        for cam in client.CAM:
            cmd += ["-i", sandbox.linux_path(evidence / (cam + ".png"))]
        cmd += ["--"] + ([self.thread_id] if self.thread_id else []) + ["-"]
        cmd = sandbox.wsl_command(cmd)
        save(
            directory / "brain_call.json",
            dict(
                model=MODEL,
                reasoning_effort=self.effort,
                resume_thread=self.thread_id,
                command=cmd,
                started_utc=stamp(),
            ),
        )

        def retry_guard(events: list[dict]) -> bool:
            if self.thread_id is not None:
                return False
            thread = next(
                (
                    e.get("thread_id")
                    for e in events
                    if e.get("type") == "thread.started"
                ),
                None,
            )
            return bool(thread and startup_only(thread, backend=self.backend))

        result = run_teacher_call(cmd, prompt, directory, self.on_status, retry_guard)
        events = [
            json.loads(line)
            for line in (directory / "brain_events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        if result.returncode or not any(
            e.get("type") == "turn.completed" for e in events
        ):
            raise RuntimeError(f"GPT failed; see {directory}")
        audit_events(events, self.root, sandbox)
        thread = next(
            (e["thread_id"] for e in events if e.get("type") == "thread.started"), None
        )
        if not thread or (self.thread_id and thread != self.thread_id):
            raise RuntimeError("Persistent teacher thread mismatch")
        views = session_images(thread, backend=self.backend)
        save(directory / "image_tool_evidence.json", {"views": views})
        for view in views:
            if not sandbox.canonical_image_allowed(view["path"], self.root):
                raise RuntimeError("Teacher image path outside workspace")
        self.thread_id = thread
        decision = json.loads(
            (directory / "brain_output.json").read_text(encoding="utf-8")
        )
        validate(decision, observation)
        save(directory / "teacher_session.json", dict(thread_id=thread))
        return decision


def episode(
    root: Path,
    trial: dict,
    limit: int,
    checkpoint: Callable[[], None],
    backend: str = "wsl",
    effort: str = "xhigh",
) -> None:
    check_stop(root)
    created = client.http(
        client.SIM + "/session",
        dict(
            layout=json.loads(
                (client.ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text()
            ),
            task="put_bottles",
            instruction="Pick up the bottles and throw them into the dustbin, using handover when needed.",
        ),
    )
    sid = created["session_id"]
    trial.update(
        session_id=sid,
        state="running",
        started_utc=stamp(),
        step=0,
        decisions=0,
        corrections=0,
        corrected_steps=0,
    )
    checkpoint()
    fk = RobotFK()
    teacher = Teacher(root / sid / "teacher", backend, effort)

    def audit_status(status: dict) -> None:
        trial.update(phase="teacher_audit", audit=status)
        checkpoint()

    teacher.on_status = audit_status
    previous, initial = None, None
    while trial["step"] < limit:
        check_stop(root)
        directory = root / sid / "decisions" / f"{trial['decisions']:04d}"
        directory.mkdir(parents=True)
        timings = {}
        began = time.monotonic()
        obs, summary = client.observe(sid)
        save(directory / "observation.json", obs)
        if obs["done"]:
            break
        visible = policy_observation(obs)
        visible["current_eef"] = fk.verify(obs)
        visible["remaining_steps"] = min(limit - obs["t"], obs["remaining_steps"])
        if initial is None:
            initial = visible["current_eef"]
        timings["observe_fk"] = time.monotonic() - began
        payload = dict(
            images={v: obs["images"][k] for k, v in client.CAM.items()},
            shapes={v: obs["shapes"][k] for k, v in client.CAM.items()},
            state=obs["state"],
            prompt=obs["instruction"],
        )
        save(directory / "pi_request.json", payload)
        began = time.monotonic()
        raw = client.http(client.PI + "/infer", payload)
        save(directory / "pi_raw_response.json", raw)
        values = np.asarray(raw["actions"], dtype=float)
        if values.shape == (1, 50, 14):
            values = values[0]
        if values.shape != (50, 14) or not np.isfinite(values).all():
            raise ValueError("Invalid H50 proposal")
        values[:, [6, 13]] = np.clip(values[:, [6, 13]], 0, 1)
        proposal = dict(
            observation_id=obs["observation_id"],
            actions=values.tolist(),
            link6_trajectory=[fk.poses(row) for row in values],
        )
        save(directory / "pi_proposal.json", proposal)
        timings["pi05_fk"] = time.monotonic() - began
        check_stop(root)
        began = time.monotonic()
        decision = teacher.decide(
            directory, visible, proposal, Path(summary["image_dir"]), previous, initial
        )
        timings["gpt6"] = time.monotonic() - began
        save(directory / "decision.json", decision)
        trial["phase"] = "executing_approved_segment"
        check_stop(root)
        count = min(decision["steps"], limit - obs["t"])
        actual = []
        began = time.monotonic()
        # One acknowledged control at a time allows STOP checks and measured-state IK.
        # Teacher is invoked only at segment boundaries, never at these substeps.
        for i in range(count):
            check_stop(root)
            row = (
                values[i].tolist()
                if decision["mode"] == "student"
                else fk.correction(decision, obs, values[i], prefix_index=i)
            )
            request = dict(
                request_id=uuid.uuid4().hex,
                expected_step=obs["t"],
                decision_summary=decision["reason"],
                joints=[row],
            )
            save(directory / f"execution_request_{i:02d}.json", request)
            result = client.http(f"{client.SIM}/session/{sid}/act", request)
            save(directory / f"execution_response_{i:02d}.json", result)
            if result["t"] != obs["t"] + 1:
                raise RuntimeError("Unexpected ACK step; no action retry")
            trial["step"] = result["t"]
            actual.append(row)
            if decision["mode"] != "student":
                trial["corrected_steps"] += 1
            checkpoint()
            if result.get("done"):
                break
            obs, _ = client.observe(sid)
            save(directory / f"control_observation_{i:02d}.json", obs)
        timings["execution"] = time.monotonic() - began
        trial["decisions"] += 1
        trial["corrections"] += int(decision["mode"] != "student")
        trial["teacher_thread_id"] = teacher.thread_id
        previous = dict(
            decision=decision,
            before_step=visible["t"],
            after_step=trial["step"],
            executed_actions=actual,
        )
        save(
            directory / "event.json",
            dict(**previous, timings=timings, finished_utc=stamp()),
        )
        checkpoint()
        print(
            json.dumps(
                dict(
                    trial=trial["number"],
                    step=trial["step"],
                    mode=decision["mode"],
                    steps=len(actual),
                    timings=timings,
                )
            ),
            flush=True,
        )
        if result.get("done"):
            break
    result = client.execute(
        sid, trial["step"], "finish", {}, "Author-style episode finished"
    )
    validate_completion(result, limit)
    trial.update(state="completed", result=result, finished_utc=stamp())
    checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--backend", choices=("wsl", "windows"), default="wsl")
    parser.add_argument("--trials", type=int, choices=range(1, 4), default=3)
    parser.add_argument(
        "--reasoning-effort", choices=("medium", "high", "xhigh"), default="xhigh"
    )
    args = parser.parse_args()
    root = (
        client.ROOT
        / "runs"
        / ("author_hybrid_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    root.mkdir(exist_ok=False)
    report = dict(
        state="preflight",
        created_utc=stamp(),
        requested_model=MODEL,
        reasoning_effort=args.reasoning_effort,
        backend=args.backend,
        transport="http" if args.backend == "windows" else "default",
        audit_retry_policy=dict(
            timeout_seconds=AUDIT_TIMEOUT_SECONDS,
            max_attempts=AUDIT_MAX_ATTEMPTS,
            retry_only="first-turn startup-only timeout verified against native transcript",
            resumed_timeout_may_repeat_context=False,
        ),
        requested_trials=args.trials,
        pilot=dict(number="pilot", state="pending"),
        trials=[],
        max_steps=700,
        batch_steps="student 1–15 / edit,eef 1–5 (teacher selected)",
        protocol="author-style MuJoCo adaptation",
        seed="fixed layout 0; policy sampling not seeded",
        differences=[
            "Reasoning effort=" + args.reasoning_effort + " (author uses xhigh)",
            "MuJoCo rather than Isaac Sim",
            "converted NPU pi0.5 rather than JAX",
            "JSON service adapter with persistent teacher analysis workspace",
            (
                "Windows native functional run; read confinement not verified"
                if args.backend == "windows"
                else "WSL isolated shell/calculation/image/notes tools; external MCP/browser/network excluded from policy"
            ),
        ],
        source_hashes={
            name: hashlib.sha256((client.ROOT / name).read_bytes()).hexdigest()
            for name in (
                "run_author_hybrid.py",
                "author_protocol.py",
                "author_sandbox.py",
                "author_windows.py",
                "author_tool_log.py",
                "client.py",
                "harness/control.py",
                "harness/simsvc.py",
                "assets/x5/dual_x5_scene.xml",
            )
        },
    )

    def checkpoint() -> None:
        save(root / "report.json", report)
        report_html(root, report)

    print("EXPERIMENT_REPORT " + str(root / "report.html"), flush=True)
    checkpoint()
    try:
        for key, endpoint in (("sim_health", client.SIM), ("pi_health", client.PI)):
            report[key] = client.http(endpoint + "/health")
            if not report[key].get("ok"):
                raise RuntimeError("Unhealthy component: " + key)
        current = client.http(client.SIM + "/status")
        if current.get("session_id") and not current.get("done"):
            raise RuntimeError("Another active simulation exists")
        # 18 steps guarantees at least two teacher turns even when choosing 15.
        episode(
            root, report["pilot"], 18, checkpoint, args.backend, args.reasoning_effort
        )
        if not args.pilot_only:
            report["state"] = "running"
            for number in range(1, args.trials + 1):
                trial = dict(number=number, state="pending")
                report["trials"].append(trial)
                episode(
                    root, trial, 700, checkpoint, args.backend, args.reasoning_effort
                )
        report["state"] = "pilot_completed" if args.pilot_only else "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            state="interrupted",
            error=f"{type(exc).__name__}: {exc}",
            stopped_utc=stamp(),
        )
        active = report["trials"][-1] if report["trials"] else report["pilot"]
        active["state"] = "interrupted"
        checkpoint()
        # Only finish our own session, never an unrelated active simulation.
        if active.get("session_id"):
            try:
                status = client.http(client.SIM + "/status")
                if status.get("session_id") == active["session_id"]:
                    active["result"] = client.execute(
                        active["session_id"],
                        status["t"],
                        "finish",
                        {},
                        "Interrupted author-style trial",
                    )
            except Exception as cleanup:
                report["cleanup_error"] = str(cleanup)
        checkpoint()
        raise
    checkpoint()


if __name__ == "__main__":
    main()
