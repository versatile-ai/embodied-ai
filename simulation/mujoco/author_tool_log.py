"""Extract image-tool evidence omitted from Codex code-mode compact JSONL.

Do not export private reasoning or raw session/image contents.
"""

import json
import os
from pathlib import Path
import re
import uuid
from author_sandbox import DISTRO, LINUX_HOME


def image_evidence(rows: list[dict], turn_id: str | None = None) -> list[dict]:
    current = None
    calls = {}
    outputs = set()
    for row in rows:
        payload = row.get("payload", {})
        if row.get("type") == "turn_context":
            current = payload.get("turn_id")
        if row.get("type") != "response_item":
            continue
        if payload.get("type") == "custom_tool_call" and "view_image" in payload.get(
            "input", ""
        ):
            source = payload["input"]
            if source.count("view_image") != 1:
                raise ValueError(
                    "Multiple image calls in one exec cannot be attributed safely"
                )
            match = re.search(
                r'tools\.view_image\s*\(\s*\{\s*["\x27]?path["\x27]?\s*:\s*("(?:[^"\\]|\\.)*")',
                source,
            )
            if not match:
                raise ValueError("Unresolved image tool path; cannot audit")
            calls[payload["call_id"]] = dict(
                path=json.loads(match[1]), call_id=payload["call_id"], turn_id=current
            )
        if payload.get("type") == "custom_tool_call_output":
            output = payload.get("output", [])
            if isinstance(output, list) and any(
                x.get("type") == "input_image" for x in output if isinstance(x, dict)
            ):
                if payload["call_id"] not in calls:
                    raise ValueError("Image returned from an unrecognized tool call")
                outputs.add(payload["call_id"])
    selected = current if turn_id is None else turn_id
    return [
        dict(c, returned_image=key in outputs)
        for key, c in calls.items()
        if c["turn_id"] == selected
    ]


def session_images(
    thread_id: str, turn_id: str | None = None, *, backend: str = "wsl"
) -> list[dict]:
    return image_evidence(session_rows(thread_id, backend=backend), turn_id)


def session_rows(thread_id: str, *, backend: str = "wsl") -> list[dict]:
    """Read a unique native transcript locally; never expose its contents."""
    uuid.UUID(thread_id)
    base = Path("//wsl.localhost") / DISTRO / LINUX_HOME.lstrip("/") / ".codex/sessions"
    if backend == "windows":
        base = (
            Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "sessions"
        )
    elif backend != "wsl":
        raise ValueError("Unknown teacher backend")
    matches = list(base.glob(f"*/*/*/*{thread_id}.jsonl"))
    if len(matches) != 1:
        raise RuntimeError("Expected exact unique teacher session transcript")
    # Parse locally, retaining only call/return metadata in the returned artifact.
    with matches[0].open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows


def startup_only(thread_id: str, *, backend: str = "windows") -> bool:
    """Conservative first-turn retry gate, rejecting any assistant/tool activity."""
    try:
        rows = session_rows(thread_id, backend=backend)
    except (OSError, ValueError, RuntimeError):
        return False
    if not any(row.get("type") == "turn_context" for row in rows):
        return False
    for row in rows:
        kind = row.get("type")
        payload = row.get("payload", {})
        if kind in ("session_meta", "turn_context"):
            continue
        if (
            kind == "response_item"
            and payload.get("type") == "message"
            and payload.get("role") in ("user", "developer", "system")
        ):
            continue
        if kind == "event_msg" and payload.get("type") in (
            "task_started",
            "user_message",
        ):
            continue
        return False
    return True
