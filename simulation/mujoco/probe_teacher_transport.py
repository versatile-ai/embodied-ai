"""Diagnostic first-turn replay; tools share the original workspace, no direct actions."""

import argparse
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    source = args.directory.resolve()
    output = source / ("http_probe_" + datetime.now().strftime("%H%M%S"))
    output.mkdir(exist_ok=False)
    record = json.loads((source / "brain_call.json").read_text(encoding="utf-8"))
    command = record["command"]
    if "resume" in command:
        raise ValueError("Do not replay into an existing teacher conversation")
    command[command.index("-o") + 1] = str(output / "decision.json")
    command[2:2] = [
        "-c",
        'model_provider="author_http"',
        "-c",
        'model_providers.author_http.name="OpenAI HTTP diagnostic"',
        "-c",
        'model_providers.author_http.base_url="https://chatgpt.com/backend-api/codex"',
        "-c",
        "model_providers.author_http.requires_openai_auth=true",
        "-c",
        "model_providers.author_http.supports_websockets=false",
    ]
    (output / "command.json").write_text(json.dumps(command), encoding="utf-8")
    began = time.monotonic()
    result = {}
    try:
        with (output / "events.jsonl").open("w", encoding="utf-8") as out, (
            output / "stderr.txt"
        ).open("w", encoding="utf-8") as err:
            completed = subprocess.run(
                command,
                input=(source / "brain_input.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
                stdout=out,
                stderr=err,
                timeout=180,
            )
        events = [
            json.loads(line)
            for line in (output / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        result = dict(
            returncode=completed.returncode,
            turn_completed=any(e.get("type") == "turn.completed" for e in events),
        )
    except subprocess.TimeoutExpired:
        result = dict(error="timeout_180s", turn_completed=False)
    result["elapsed_seconds"] = time.monotonic() - began
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    if not result["turn_completed"] or result.get("returncode") != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
