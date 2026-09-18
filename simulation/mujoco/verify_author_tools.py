"""Real two-turn GPT tool-capability check. No simulator/inference requests."""

from datetime import datetime
import argparse
import json
from pathlib import Path

import client
import numpy as np
from PIL import Image
from author_protocol import RobotFK
from run_author_hybrid import Teacher
from run_gpt6_hybrid import save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("wsl", "windows"), default="wsl")
    args = parser.parse_args()
    root = (
        client.ROOT
        / "runs"
        / ("author_tools_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    root.mkdir()
    teacher = Teacher(root / "teacher", args.backend)
    fk = RobotFK()
    row = [0.0] * 14
    poses = fk.poses(row)
    images = client.ROOT / "runs/6683a8d35438/observations/000024"
    report = dict(
        state="running", workspace=str(teacher.root), sandbox=teacher.probe_result
    )
    save(root / "report.json", report)
    print("TOOL_PREFLIGHT " + str(root), flush=True)
    for i in range(2):
        directory = root / f"{i:04d}"
        directory.mkdir()
        instruction = (
            "This is an offline TOOL CAPABILITY TEST, not a robot episode. No action will execute. "
            "Use the supplied Python interpreter via the shell tool, actually run NumPy to compute "
            "sum(arange(10)) and write result 45 to NOTES.md. Open the supplied cam_base.png "
            "with Pillow, crop its central half and save crop.png in the workspace, then use the "
            "image viewing tool to inspect the crop. Return a schema-valid student decision with "
            "steps=1; no correction. Do not claim tests passed without doing them."
            if i == 0
            else "This is the second offline TOOL CAPABILITY TEST. Use the shell to read NOTES.md "
            "from the preceding turn, verify it contains 45, and append resumed_ok. Reopen crop.png "
            "with the image tool. Return student steps=1. No robot action will execute."
        )
        obs = dict(
            observation_id=f"offline:{i}",
            t=i,
            remaining_steps=2 - i,
            state=row,
            current_eef=poses,
            instruction=instruction,
        )
        proposal = dict(
            observation_id=obs["observation_id"],
            actions=[row] * 50,
            link6_trajectory=[poses] * 50,
        )
        try:
            decision = teacher.decide(directory, obs, proposal, images, None, poses)
            report[f"turn_{i}"] = dict(thread_id=teacher.thread_id, decision=decision)
            save(root / "report.json", report)
        except Exception as exc:
            report.update(state="failed", error=str(exc))
            save(root / "report.json", report)
            raise
    try:
        notes = (teacher.root / "NOTES.md").read_text()
        if "45" not in notes or "resumed_ok" not in notes:
            raise RuntimeError("Tool-created notes or resumed verification missing")
        with Image.open(images / "cam_base.png") as source, Image.open(
            teacher.root / "crop.png"
        ) as crop:
            width, height = source.size
            expected = source.crop(
                (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
            )
            if not np.array_equal(np.asarray(expected), np.asarray(crop)):
                raise RuntimeError("Crop does not match source evidence")
        if report["turn_0"]["thread_id"] != report["turn_1"]["thread_id"]:
            raise RuntimeError("Teacher conversation was not resumed")
        for i in range(2):
            events = [
                json.loads(line)
                for line in (root / f"{i:04d}" / "brain_events.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            kinds = {
                e.get("item", {}).get("type")
                for e in events
                if e.get("type") == "item.completed"
            }
            views = json.loads(
                (root / f"{i:04d}" / "image_tool_evidence.json").read_text()
            )["views"]
            if "command_execution" not in kinds or not any(
                v["returned_image"] for v in views
            ):
                raise RuntimeError("Actual shell/image-view execution evidence missing")
    except Exception as exc:
        report.update(state="failed", error=str(exc))
        save(root / "report.json", report)
        raise
    report["state"] = "passed"
    save(root / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
