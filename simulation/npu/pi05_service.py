#!/usr/bin/env python3
"""π0.5 inference service (RoboDojo finetuned weights) on Ascend NPU.

HTTP JSON API for a single experiment caller. Concurrent inference is not supported:

  POST /infer  {"images": {"cam_high": <b64 uint8 HWC>, "cam_left_wrist": ...,
                            "cam_right_wrist": ...},
                "state": [14 floats],          # [L6, Lgrip, R6, Rgrip]
                "prompt": "task instruction"}
  ->  {"actions": [[14 floats] x 50], "ms": float}   # unnormalized robot units

  GET /health -> {"ok": true, "device": "npu"}

Uses openpi's production preprocessing chain (resize/tokenize/norm-stats) so the
inputs match training exactly; repack is identity on canonical camera keys.
"""

import base64
import json
import pathlib
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, "/data/pi05_hybrid/openpi/src")

from openpi import transforms  # noqa: E402
from openpi.models import pi0_config  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.training import checkpoints as ocp_checkpoints  # noqa: E402
from openpi.training import config as train_config_mod  # noqa: E402

CKPT_DIR = "/data/pi05_hybrid/weights/robodojo_pi05_pt"
ASSETS_DIR = (
    "/data/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/"
    "RoboDojo-sim-arx_x5-joint-0/59999/assets"
)
ASSET_ID = "arx_x5_sim"
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "npu"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8642

CAM_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def build_policy():
    data_cfg = train_config_mod.LeRobotAlohaDataConfig(
        repo_id="arx_x5_sim",
        assets=train_config_mod.AssetsConfig(assets_dir=ASSETS_DIR, asset_id=ASSET_ID),
        base_config=train_config_mod.DataConfig(prompt_from_task=True),
        # RoboDojo ARX X5 seed0 was trained without standard Aloha geometry
        # conversion. Generic openpi defaults flip shoulder/elbow signs and
        # remap gripper 0 to ~0.553, breaking this checkpoint's contract.
        adapt_to_pi=False,
        use_delta_joint_actions=True,
    )
    train_cfg = train_config_mod.TrainConfig(
        name="robodojo_pi05_service",
        model=pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=None),
        data=data_cfg,
    )
    norm_stats = ocp_checkpoints.load_norm_stats(pathlib.Path(ASSETS_DIR), ASSET_ID)
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {k: f"images/{k}" for k in CAM_KEYS},
                    "state": "state",
                    "prompt": "prompt",
                }
            )
        ]
    )
    return policy_config.create_trained_policy(
        train_cfg,
        CKPT_DIR,
        repack_transforms=repack,
        norm_stats=norm_stats,
        pytorch_device=DEVICE,
    )


POLICY = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(
                {
                    "ok": True,
                    "device": DEVICE,
                    "contract": "robodojo-arx-x5-v2",
                    "adapt_to_pi": False,
                    "use_delta_joint_actions": True,
                    "asset_id": ASSET_ID,
                }
            )
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/infer":
            self._send({"error": "not found"}, 404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            obs = {
                "images": {
                    k: np.frombuffer(base64.b64decode(req["images"][k]), np.uint8)
                    .reshape(req["shapes"][k])
                    .transpose(2, 0, 1)
                    for k in CAM_KEYS
                },
                "state": np.asarray(req["state"], np.float32),
                "prompt": req.get("prompt", ""),
            }
            t0 = time.perf_counter()
            out = POLICY.infer(obs)
            ms = (time.perf_counter() - t0) * 1000.0
            acts = np.asarray(out["actions"])
            if acts.ndim == 3:
                acts = acts[0]
            self._send({"actions": acts.tolist(), "ms": round(ms, 1)})
        except Exception as e:  # noqa: BLE001
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


if __name__ == "__main__":
    print("loading policy...", flush=True)
    POLICY = build_policy()
    print(f"serving on :{PORT} device={DEVICE}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
