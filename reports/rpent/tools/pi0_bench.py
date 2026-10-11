#!/usr/bin/env python3
"""π0.5 VLA 推理延迟基准（155 NPU 上 vla.predict 直连测量）。
用法（在 155 容器内）:
    python3 pi0_bench.py <flywheel_episode_dir> [frame] [n]
依赖: rpent 包（HttpRpcClient / Pi05VLAClient）、VLA 服务 (127.0.0.1:18803)。
输出: chunk 推理 p50/min/max、摊销动作延迟、chunk频率、pi0_pick 耗时构成推算。
"""
import sys, time, glob
import numpy as np

sys.path.insert(0, "/data/rpent2/rpent")
from rpent.utils.rpc.http_rpc import HttpRpcClient
from rpent.robots.components.pi05_vla_client import Pi05VLAClient

ep_dir = sys.argv[1]
frame = int(sys.argv[2]) if len(sys.argv) > 2 else 50
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10

d = np.load(glob.glob(ep_dir + "/transitions.npz")[0])
env_obs = {
    "main_images": d["main_images"][frame],
    "wrist_images": d["wrist_images"][frame],
    "states": d["states"][frame].astype(np.float32),
    "task_descriptions": "pick up the target object",
}

client = Pi05VLAClient(HttpRpcClient("http://127.0.0.1:18803"), embodiment="libero")

print("预热 2 次...")
for _ in range(2):
    t0 = time.perf_counter()
    a = client.predict(env_obs, options={"mode": "eval"})
    print(f"  warmup: {time.perf_counter()-t0:.3f}s, action shape: {a.shape}")

ts = []
for i in range(n):
    t0 = time.perf_counter()
    a = client.predict(env_obs, options={"mode": "eval"})
    ts.append(time.perf_counter() - t0)
    print(f"  #{i+1}: {ts[-1]*1000:.0f}ms")

import statistics as st
med = st.median(ts)
print(f"\nπ0.5 单次推理(chunk): p50={med*1000:.0f}ms | min={min(ts)*1000:.0f}ms | max={max(ts)*1000:.0f}ms")
print(f"动作chunk: {a.shape[0]} 步 × {a.shape[1]} 维")
print(f"摊到每步动作: {med/a.shape[0]*1000:.1f}ms/action")
print(f"chunk 频率: {1/med:.2f} Hz")

# pi0_pick 耗时构成推算（若 flywheel 数据含 vla_chunk_id）
vla_ids = d.get("vla_chunk_id")
if vla_ids is not None and (vla_ids >= 0).any():
    n_chunks = len(np.unique(vla_ids[vla_ids >= 0]))
    total_steps = len(vla_ids)
    print(f"\nVLA chunk 数: {n_chunks} | 总仿真步数: {total_steps}")
    print(f"推理耗时估算: {n_chunks} × {med:.2f}s = {n_chunks*med:.1f}s")
