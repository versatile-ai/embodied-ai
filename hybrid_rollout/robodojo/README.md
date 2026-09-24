# robodojo

See the [public project guide](../../README.md) for setup, scope, dependencies, tests and licensing.

## Evaluation evidence

- [π0.5 Layout 0 evaluation report](reports/pi05_layout0_20260919/pi05_test_report.html) — performance, trajectory analysis, completion review, parameters, raw logs and the full four-camera recording.
- [GPT-6 Astra Layout 0 evaluation report](reports/gpt6_layout0_20260919/gpt6_test_report.html) — per-decision and per-control-step token accounting, latency, behavior analysis, comparison with π0.5, embedded raw data and the full four-camera recording.

## Frozen GPU simulation entry point

Use `run_gpu_sim_server.sh` for the validated four-bottle scene. Its defaults
are the reproducible baseline: `put_bottles_into_dustbin`, Layout 0, one
environment, GPU 0, TCP port 19113, 250 Hz physics (`dt=0.004`) and 25 Hz
policy control. The server keeps the three official policy cameras and adds a
fourth overview camera only for monitoring and video.

```bash
hybrid_rollout/robodojo/run_gpu_sim_server.sh /path/to/run-output
```

The launcher pins the headless/EULA variables, CUDA device, direct RGB capture,
scene setup behavior and three post-reset RTX warmup renders. The server writes
both `resolved_config.json` (the complete resolved RoboDojo configuration) and
`adapter_runtime.json` (the small set of adapter decisions) into every run.

The adapter contains two required compatibility fixes: it assigns RoboDojo's
nested simulation config to `cuda:0`, and it aligns the static geometry origin
with the CPU pose returned by USD before native reward checks. The latter stays
inside the adapter; the official RoboDojo checkout is not edited.
