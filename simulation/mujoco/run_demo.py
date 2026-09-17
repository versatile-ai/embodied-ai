"""Bounded pi0.5 simulation demo; use the existing SSH tunnel or run on server."""

import argparse
import json
import math
import time

import client


def run_session(
    session_id: str, steps: int = 30, batch: int = 3, pause: float = 1.0
) -> dict:
    """Run additional steps, then encode videos; never retry uncertain actions."""
    if not 1 <= steps <= 700 or not 1 <= batch <= 15:
        raise ValueError("steps must be 1..700 and batch must be 1..15")
    if not math.isfinite(pause) or pause < 0:
        raise ValueError("pause must be finite and nonnegative")
    observed, _ = client.observe(session_id)
    start = expected = observed["t"]
    while not observed["done"] and expected - start < steps:
        count = min(batch, steps - (expected - start), observed["remaining_steps"])
        if count <= 0:
            break
        _, proposal = client.infer(session_id, observed)
        if proposal["observation_id"] != observed["observation_id"]:
            raise ValueError("Stale proposal; no action sent")
        began = time.monotonic()
        result = client.execute(
            session_id,
            expected,
            "act",
            {"joints": proposal["actions"][:count]},
            f"Bounded pi0.5 demo: up to {steps} additional steps",
        )
        if not expected < result["t"] <= expected + count:
            raise RuntimeError("Unexpected step advancement; stopping without retry")
        expected = result["t"]
        print(
            json.dumps(
                {
                    "session_id": session_id,
                    "step": expected,
                    "executed": expected - start,
                    "score": result["score"],
                    "done": result["done"],
                    "batch_seconds": round(time.monotonic() - began, 2),
                }
            ),
            flush=True,
        )
        time.sleep(pause)  # Let the single-threaded service serve live-page requests.
        observed, _ = client.observe(session_id)
        if observed["t"] != expected:
            raise RuntimeError("Another controller advanced this session; stopping")
    result = client.execute(session_id, expected, "finish", {}, "Bounded demo complete")
    print(json.dumps({"session_id": session_id, "result": result}), flush=True)
    return result


def main() -> None:
    """Create a bottle scene unless an explicit existing session was supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="Explicit active session to continue")
    parser.add_argument(
        "--steps", type=int, default=30, help="Additional control steps (1..700)"
    )
    parser.add_argument("--batch", type=int, default=3, choices=range(1, 16))
    parser.add_argument(
        "--pause", type=float, default=1.0, help="Seconds between batches"
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 700 or not math.isfinite(args.pause) or args.pause < 0:
        parser.error("steps must be 1..700; pause must be finite and nonnegative")
    session_id = args.session
    try:
        if session_id is None:
            layout = json.loads(
                (client.ROOT / "layouts/put_bottles_into_dustbin_0.json").read_text()
            )
            created = client.http(
                client.SIM + "/session",
                {
                    "layout": layout,
                    "task": "put_bottles",
                    "instruction": "Pick up the bottles and throw them into the dustbin, using handover when needed.",
                },
            )
            session_id = created["session_id"]
        print(
            json.dumps(
                {
                    "session_id": session_id,
                    "steps": args.steps,
                    "live": client.SIM + "/live",
                }
            ),
            flush=True,
        )
        run_session(session_id, args.steps, args.batch, args.pause)
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(
            1,
            f"Stopped: {type(exc).__name__}: {exc}\n"
            f"Session: {session_id}. No additional retry or finish will be sent. "
            "Check /status before continuing; an in-flight action may still complete.\n",
        )


if __name__ == "__main__":
    main()
