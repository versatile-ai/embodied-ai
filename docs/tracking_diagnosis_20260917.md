# Tracking diagnosis — experiment stopped

## Status

User requested stopping the experiment and switching to author source. Run
`author_hybrid_20260917_225148`, session `030fd4325941`, was stopped at 271 steps:
0 bottles, score 0, `stopped_by_operator`. Evidence and three camera videos are
preserved. No new formal trial was started. Tracking is **not fixed**.

Author checkout: `external/GPT-as-Policy`, commit
`8f3d362b077d8efb77e2a7274d5b2c20e2243846`.
Its SOURCE.json pins RoboDojo `ee67a1468510da7624a089164402359f2afc72c8`.
The checkout is not wired into the current runner yet.

## Reproduction and ablations

Run from `simulation/mujoco`:

```powershell
rtk proxy .venv/Scripts/python.exe diagnose_policy_tracking.py --assert-tracking
```

This is intentionally a red-capable **diagnostic**, not a portable acceptance
test. It requires the preserved run, checks 42 consecutive single-frame requests
from decision directories 0000–0002, replays real MuJoCo physics without RPC or
rendering, then holds the last command for 150 control ticks (6 seconds).
It does not establish that the requested trajectory is physically feasible.

Fixture SHA256: `b3f23199312fe6e4501f61a7a0f91695e9211f0120539594909d537dcd8d3fca`.
simsvc.py SHA256: `6a33f503f071c2bdca79fcdcaa9824d33f807c6b19bb7b0bf3c7899e0affd5fe`.

| Offline variant | Maximum arm error (rad) | Applied target error (rad) |
| --- | ---: | ---: |
| Current physics | 1.274423 | 0 |
| No slew / low-pass filter | 1.274423 | 0 |
| All contacts disabled, diagnostic only | 0.005206 | 0 |

The final left joint2 target is -1.581611, actual -0.307188. Penetrating contacts
include table/left_link2, left_base_link/left_link2, left_link1/left_link3 and
left_link2/left_link4. These observations implicate contact constraints for this
replay; they do **not** prove those contacts are erroneous or justify removing
collision protection. No production collision or servo changes were made.

## Verified author configuration

[Pinned native robot configuration](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env/robot_manager/robot_config/x5.py):
arm stiffness 4400, damping 40, armature .01, effort limit 100, velocity limit 5;
gravity disabled; initial arm positions zero and both gripper joints .044.
Local arm position gains are 100/10 and initial grippers are zero.
These are real differences, not yet established as the cause of the bad target.

Official assets were inspected at Hugging Face revision
`91f76c28d93dd20c5fa46ce6a5a1d96a4f384acd` (current asset revision, not proven to be
the historical experiment's asset revision):

- ARX.usd SHA256 `5723a835bf5627a935a8dabb3a44447725bd1465243ce791e221b3e990121d63`.
- ARX_physics.usd SHA256 `96369690dd2a0bdc583f21e7a985e2a0c0d9a50e0f4c6d41d8dc4d9c918df53b`.

Important correction: ARX_physics.usd alone limits joint2/3 to 0–pi, but the
actual standalone ARX.usd uses approximately -10–10 radians. Therefore the
earlier suspicion of mismatched arm limits was rejected. Arm joint axes and
joint-frame rotations also match the local URDF-derived model to rounding.
Do not flip signs or impose the obsolete layer limits as a purported fix.

## Next decision / remaining work

Native author experiments use Isaac Sim, not MuJoCo. Local hardware inspection
found Intel Arc 130T; the recorded 192 deployment uses Ascend 910B3/ARM64.
Neither is the NVIDIA RTX setup required by
[Isaac Sim 5.1](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
An eligible host must be provided before committing to native reproduction.
Keeping MuJoCo and porting the author's controller is a separate, explicitly
non-native experiment and needs the user's choice.

Remaining diagnosis: compare the original policy adapter, normalizer and commanded
vs measured proprioception; then replay native reference actions with matched
initial gripper state and physical configuration. Add a feasible-trajectory
regression test before changing dynamics. Recheck bilateral grasp stability for
any collision/actuator changes. Do not restart the three formal trials yet.
