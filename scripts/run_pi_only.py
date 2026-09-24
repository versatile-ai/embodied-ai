#!/usr/bin/env python3
"""Fresh, serial π0.5-only RoboDojo episode with local timing evidence."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_rollout.robodojo.pi05_server.http_client import Pi05HTTPClient
from hybrid_rollout.robodojo.robodojo_server.protocol import RPCClient


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sim = None
    student = None
    started = time.monotonic()
    decisions = []
    step = 0
    episode = None
    try:
        student = Pi05HTTPClient(args.student_url, allow_unverified=True,
                                 camera_schema="aloha")
        sim = RPCClient("127.0.0.1", args.sim_port)
        metadata = sim.request("metadata")
        if metadata["task"] != "put_bottles_into_dustbin":
            raise RuntimeError(f"Unexpected task: {metadata['task']}")
        if metadata["action_dim"] != 14 or metadata["action_horizon"] != 50:
            raise RuntimeError("Unexpected native action shape")
        save(output / "metadata.json", metadata)
        reset = sim.request("reset", seed=args.seed, source="student",
                            policy_version="pi05_npu_http_serial_v1")
        save(output / "reset.json", reset)
        episode = reset["episode_id"]
        max_steps = min(args.max_steps, metadata["max_episode_steps"])
        ended = False
        while step < max_steps and not ended:
            observation = sim.request("teacher_observation", episode_id=episode,
                                      step_id=step)
            prediction_file = output / f"prediction_{len(decisions):03d}.npz"
            t0 = time.monotonic()
            actions, prediction = student.infer(observation, prediction_file)
            inference_wall = time.monotonic() - t0
            executed = 0
            execution_wall = 0.0
            planned = min(50, max_steps - step)
            for offset in range(0, planned, 15):
                batch = actions[offset:min(offset + 15, planned)]
                t1 = time.monotonic()
                response = sim.request("chunk_step", episode_id=episode,
                                       step_id=step, actions=batch)
                execution_wall += time.monotonic() - t1
                rows = response["steps"]
                if not rows:
                    raise RuntimeError("Simulator ACK contained no executed actions")
                step = response["step_id"]
                executed += len(rows)
                ended = bool(rows[-1]["terminated"] or rows[-1]["truncated"]
                             or rows[-1]["success"])
                if ended:
                    break
            record = dict(decision=len(decisions), step_id=step,
                          executed_steps=executed, prediction=prediction,
                          inference_wall_seconds=inference_wall,
                          execution_wall_seconds=execution_wall)
            decisions.append(record)
            save(output / "decisions.json", decisions)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        finish = sim.request("finish_pilot", episode_id=episode, step_id=step,
                             reason="terminal" if ended else "step_budget")
        result = dict(task=metadata["task"], seed=args.seed, episode_id=episode,
                      step_id=step, decisions=len(decisions), finished=ended,
                      native_success=finish["success"],
                      native_terminated=finish["terminated"],
                      native_truncated=finish["truncated"],
                      wall_seconds=time.monotonic() - started,
                      inference_roundtrip_seconds=sum(d["inference_wall_seconds"] for d in decisions),
                      execution_seconds=sum(d["execution_wall_seconds"] for d in decisions),
                      finish=finish)
        save(output / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as error:
        save(output / "failure.json", dict(error=str(error), episode_id=episode,
                                          step_id=step, completed=False))
        raise
    finally:
        if sim is not None:
            sim.close()
        if student is not None:
            student.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sim-port", type=int, default=19113)
    parser.add_argument("--student-url", default="http://127.0.0.1:8642")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=700)
    run(parser.parse_args())
