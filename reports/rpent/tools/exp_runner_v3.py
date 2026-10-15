#!/usr/bin/env python3
"""Instrumented experiment runner — corrected version.
Fixes: approach phase before grasp, re-observe on failure, real DS plan for B,
multi-frame V-JEPA encoding, full per-step instrumentation + frame capture.
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np

sys.path.insert(0, "/data/rpent2/rpent")
from rpent.utils.rpc.http_rpc import HttpRpcClient
from rpent.robots.components.pi05_vla_client import Pi05VLAClient

DS_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
JEV_KEY = os.environ.get("JEV_KEY", "")
JEV_PROXY = os.environ.get("JEV_PROXY", "")
JEV_URL = "https://api.typesafe.ai/v1/systemone"
OUTDIR = os.environ.get("EXP_OUT", "/data/rpent2/logs/exp")


# ── Jev ─────────────────────────────────────────────────────
def jev_decide(state, options, criteria):
    handlers = []
    if JEV_PROXY:
        handlers.append(urllib.request.ProxyHandler({"http": JEV_PROXY, "https": JEV_PROXY}))
    opener = urllib.request.build_opener(*handlers)
    body = {"model": "jev-latest", "state": state[:3000],
            "questions": {"next": {"type": "choice", "options": list(options), "criteria": criteria}}}
    req = urllib.request.Request(JEV_URL, data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + JEV_KEY, "Content-Type": "application/json"})
    resp = json.loads(opener.open(req, timeout=30).read())
    ans = resp["answers"]["next"]
    return ans["choice"], ans.get("confidence", 0)


# ── V-JEPA ──────────────────────────────────────────────────
_vjepa = None
_mean_t = None; _std_t = None
def get_vjepa():
    global _vjepa, _mean_t, _std_t
    if _vjepa is None:
        import torch, torch_npu
        from transformers import VJEPA2Model
        _vjepa = VJEPA2Model.from_pretrained(
            "/data/rpent2/checkpoints/vjepa2", dtype=torch.float16).to("npu").eval()
        _mean_t = torch.tensor([0.485,0.456,0.406]).view(1,1,3,1,1).to("npu").half()
        _std_t = torch.tensor([0.229,0.224,0.225]).view(1,1,3,1,1).to("npu").half()
    return _vjepa

def vjepa_encode(frames_list):
    """多帧真实视频编码（修复冻结帧缺陷）。frames_list: list of (H,W,3) uint8."""
    import torch
    model = get_vjepa()
    n = min(len(frames_list), 8)  # 最多8帧
    if n == 0: return None
    # 均匀采样到4帧（tubelet_size=2的倍数）
    idx = np.linspace(0, len(frames_list)-1, max(4, n)).astype(int)
    frames = [frames_list[i] for i in idx]
    ts = np.stack([np.asarray(f).astype(np.float32)/255.0 for f in frames])  # (T,H,W,3)
    t = torch.from_numpy(ts).permute(3,0,1,2).unsqueeze(0)  # (1,3,T,H,W)
    t = t.permute(0,2,1,3,4)  # → (1,T,3,H,W) = (B,T,C,H,W)
    t = (t.to("npu").half() - _mean_t) / _std_t
    with torch.no_grad():
        out = model(pixel_values_videos=t)
    latent = out.last_hidden_state
    return latent.mean(dim=1).cpu().float().numpy().flatten()

def vjepa_dist(a, b):
    return float(1.0 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))


# ── Instrumented Runner ─────────────────────────────────────
class Runner:
    def __init__(self, exp, task, max_loops=6, max_chunks=20, use_approach=True):
        self.exp = exp
        self.task = task
        self.max_loops = max_loops
        self.max_chunks = max_chunks
        self.env = HttpRpcClient("http://127.0.0.1:18900")
        self.vla = Pi05VLAClient(HttpRpcClient("http://127.0.0.1:18803"), embodiment="libero")
        self.terminated = False
        self.t0 = time.perf_counter()
        self.steps = []       # 每步记录
        self.frames = []      # agentview帧
        self.wrist_frames = []  # 腕部帧
        self.obs = None
        self.jev_ok = 0; self.jev_fail = 0
        self.jev_lat = []
        self.vla_lat = []     # 每次VLA predict
        self.env_lat = []     # 每次chunk_step
        self.plan_text = ""   # DS生成的计划（B组）
        self.approached = False
        self.use_approach = use_approach
        self.obj_z_log = []
        self.vjepa_dists = []
        self.raw = {}

        os.makedirs(f"{OUTDIR}/{exp}", exist_ok=True)

    def record(self, skill, wall_ms, vla_ms=0, env_ms=0, jev_ms=0,
               result="", terminated=None, extra=None):
        rec = {"step": len(self.steps)+1, "t": round(time.perf_counter()-self.t0,2),
               "skill": skill, "wall_ms": round(wall_ms,1), "vla_ms": round(vla_ms,1),
               "env_ms": round(env_ms,1), "jev_ms": round(jev_ms,1),
               "result": result[:200], "terminated": self.terminated}
        if extra: rec.update(extra)
        self.steps.append(rec)

    def save_frame(self, obs):
        """从观测保存双视角视频帧"""
        if not isinstance(obs, dict): return
        for k in ["main_images", "agentview_image"]:
            if k in obs:
                self.frames.append(np.asarray(obs[k])); break
        for k in ["wrist_images", "robot0_eye_in_hand_image"]:
            if k in obs:
                self.wrist_frames.append(np.asarray(obs[k])); break

    # ── 底层技能 ────────────────────────────────────────────

    def observe(self):
        """raw_obs仅用于特权状态查询（terminated/物体位置/EEF）。
        self.obs（VLA格式8-dim states）由reset/step/chunk_step的返回值维护，不用raw_obs覆盖。"""
        t0 = time.perf_counter()
        self.raw = self.env.call("env.raw_obs")
        ms = (time.perf_counter()-t0)*1000
        term = self.raw.get("terminated", False) if isinstance(self.raw, dict) else False
        if term: self.terminated = True
        self.record("observe", ms, env_ms=ms, result=f"terminated={term}")
        return self.raw

    def get_obj_pos(self, obj_name="salad_dressing"):
        """从env state读取目标物体位置（特权信息，仅用于approach定位）"""
        if not isinstance(self.raw, dict): return None
        for k in self.raw.keys():
            if obj_name.split("_")[0] in k.lower() and k.endswith("_pos"):
                return np.array(self.raw[k])
        return None

    def get_eef_pos(self):
        if not isinstance(self.raw, dict): return np.array([0,0,0])
        return np.array(self.raw.get("robot0_eef_pos", [0,0,0]))

    def servo_to(self, target_xyz, tol=0.02, max_steps=80):
        """比例servo到目标点。action语义与RPent move_to一致:
        action[:3]∈[-1,1]，env按action_scale=0.05缩放（action=1→5cm/步）。
        gripper=-1张开。"""
        t0 = time.perf_counter()
        env_total = 0
        ACTION_SCALE = 0.05
        STEP_CLIP = 0.025
        for i in range(max_steps):
            self.raw = self.env.call("env.raw_obs")
            eef = self.get_eef_pos()
            err = target_xyz - eef
            if np.linalg.norm(err) < tol: break
            step_dxyz = np.clip(err, -STEP_CLIP, STEP_CLIP)
            act = np.zeros(7)
            act[:3] = np.clip(step_dxyz / ACTION_SCALE, -1.0, 1.0)
            act[6] = -1.0  # 张开夹爪
            t1 = time.perf_counter()
            result = self.env.call("env.step", (act.tolist(),))
            env_total += time.perf_counter()-t1
            self.obs = result[0] if isinstance(result, (list, tuple)) else result
            self.save_frame(self.obs)
            terms = result[2] if isinstance(result, (list, tuple)) and len(result)>2 else False
            if bool(np.any(terms)) if hasattr(terms,'__len__') else bool(terms):
                self.terminated = True; break
        ms = (time.perf_counter()-t0)*1000
        eef = self.get_eef_pos()
        dist = float(np.linalg.norm(target_xyz - eef))
        self.record("servo", ms, env_ms=env_total*1000,
                    result=f"steps={i+1}, dist={dist:.3f}m")
        return dist

    def vla_grasp(self):
        """pi0_pick：VLA闭环 + 逐chunk插桩"""
        t0 = time.perf_counter()
        for ci in range(self.max_chunks):
            t1 = time.perf_counter()
            # 直接用env返回的观测dict（含env原始task_descriptions，
            # π0.5对prompt措辞敏感，不能改写——这是之前全败的根因）
            obs_vla = dict(self.obs)
            if "task_descriptions" not in obs_vla:
                obs_vla["task_descriptions"] = self.task
            actions = self.vla.predict(obs_vla, options={"mode":"eval"})
            vla_ms = (time.perf_counter()-t1)*1000
            self.vla_lat.append(vla_ms)

            alist = np.asarray(actions).tolist()
            if np.array(alist).ndim == 3: alist = alist[0]
            t2 = time.perf_counter()
            result = self.env.call("env.chunk_step", (alist,), {"return_all_frames": True})
            env_ms = (time.perf_counter()-t2)*1000
            self.env_lat.append(env_ms)

            # result[0] = 逐步obs列表（稠密帧），最后一个是下一轮的输入
            step_obs_list = result[0]
            if isinstance(step_obs_list, list):
                for so in step_obs_list:
                    self.save_frame(so)
                self.obs = step_obs_list[-1]
            else:
                self.obs = step_obs_list
                self.save_frame(self.obs)
            terms = result[2]
            term = bool(np.any(terms)) if hasattr(terms,'__len__') else bool(terms)
            if term:
                self.terminated = True
                ms = (time.perf_counter()-t0)*1000
                self.record("pi0_pick", ms, vla_ms=vla_ms, env_ms=env_ms,
                            result=f"chunks={ci+1}, terminated=True", extra={"chunks": ci+1})
                return self.obs, True
        ms = (time.perf_counter()-t0)*1000
        self.record("pi0_pick", ms, vla_ms=vla_ms, env_ms=env_ms,
                    result=f"chunks={self.max_chunks}, terminated=False")
        return self.obs, False

    def verify(self):
        t0 = time.perf_counter()
        self.obs = self.env.call("env.raw_obs")
        term = self.obs.get("terminated", False) if isinstance(self.obs, dict) else False
        if term: self.terminated = True
        ms = (time.perf_counter()-t0)*1000
        self.record("verify", ms, env_ms=ms, result=f"terminated={term}")
        return term

    # ── DS 计划生成（B组用）──────────────────────────────────

    def ds_plan(self):
        """调DS Flash生成计划骨架（B组的真DS调用）"""
        t0 = time.perf_counter()
        body = {"model": "deepseek-flash", "max_completion_tokens": 800,
                "messages": [
                    {"role":"system","content":"你是具身任务规划器。给定任务描述，输出JSON格式的步骤列表。每步含skill和expect字段。只输出JSON。"},
                    {"role":"user","content":f"任务: {self.task}\n可用技能: observe,approach,grasp,grip,verify,finish\n输出格式: {{\"steps\": [{{\"skill\":\"...\",\"expect\":\"...\"}},...]}}"}
                ]}
        req = urllib.request.Request("http://127.0.0.1:8788/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type":"application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=15))
        ms = (time.perf_counter()-t0)*1000
        text = resp["choices"][0]["message"]["content"]
        self.plan_text = text[:500]
        self.record("ds_plan", ms, result=f"DS计划: {text[:100]}")
        return text

    # ── 实验流程 ────────────────────────────────────────────

    def run(self):
        print(f"\n{'='*56}\n实验 {self.exp}: {self.task}\n{'='*56}")

        # 阶段0: reset + 初始观测
        reset_obs = self.env.call("env.reset")
        self.obs = reset_obs[0] if isinstance(reset_obs, list) else reset_obs
        self.save_frame(self.obs)
        self.observe()

        # 阶段1: B组先调DS生成计划
        if self.exp == "B":
            try: self.ds_plan()
            except Exception as e: print(f"DS计划失败(非致命): {str(e)[:60]}")

        # 阶段2: 主循环
        orig_obj_pos = None
        for loop in range(1, self.max_loops+1):
            if self.terminated: break

            # 2a. 首次approach；后续仅当物体被撞离原位才重新approach
            cur_obj_pos = self.get_obj_pos()
            if orig_obj_pos is None and cur_obj_pos is not None:
                orig_obj_pos = cur_obj_pos.copy()
            if cur_obj_pos is not None:
                self.obj_z_log.append(round(float(cur_obj_pos[2]), 3))

            need_approach = False
            if not self.use_approach:
                pass
            elif not self.approached:
                need_approach = True
            elif (cur_obj_pos is not None and orig_obj_pos is not None
                  and np.linalg.norm(cur_obj_pos - orig_obj_pos) > 0.03
                  and cur_obj_pos[2] < 0.15):
                print(f"  物体被移动{np.linalg.norm(cur_obj_pos-orig_obj_pos):.3f}m (z={cur_obj_pos[2]:.3f})，重新approach")
                need_approach = True
            elif (cur_obj_pos is not None and orig_obj_pos is not None
                  and np.linalg.norm(cur_obj_pos - orig_obj_pos) > 0.03):
                print(f"  物体z={cur_obj_pos[2]:.3f}m（可能在夹爪中），不servo")

            if need_approach and cur_obj_pos is not None:
                target = cur_obj_pos + np.array([0, 0, 0.14])
                dist = self.servo_to(target)
                self.approached = dist < 0.05
                print(f"  approach: dist={dist:.3f}m, ok={self.approached}")

            # 2b. VLA闭环抓取+放置（中途不servo，保持闭环连续性）
            print(f"\n[Loop {loop}] VLA执行...")
            self.obs, done = self.vla_grasp()
            if done:
                break

            # 2c. 失败后只刷新观测（供下一轮VLA），不重置机械臂位置
            print(f"  未完成，刷新观测继续...")
            self.observe()

        # 阶段3: 验证
        if not self.terminated:
            self.verify()

        # 阶段4: V-JEPA验证（E/F/G组）
        vjepa_dist = None
        if self.exp in ("E", "F", "G") and len(self.frames) >= 4:
            try:
                latent_start = vjepa_encode(self.frames[:4])
                latent_end = vjepa_encode(self.frames[-4:])
                vjepa_dist = vjepa_dist(latent_start, latent_end)
                self.record("vjepa_verify", 0, result=f"dist={vjepa_dist:.4f}",
                            extra={"vjepa_dist": round(vjepa_dist,4)})
            except Exception as e:
                print(f"V-JEPA验证失败: {str(e)[:60]}")

        wall = time.perf_counter() - self.t0
        result = {
            "exp": self.exp, "terminated": self.terminated,
            "wall_s": round(wall, 1), "loops": loop if 'loop' in dir() else 0,
            "vla_calls": len(self.vla_lat),
            "vla_avg_ms": round(np.mean(self.vla_lat),1) if self.vla_lat else 0,
            "env_avg_ms": round(np.mean(self.env_lat),1) if self.env_lat else 0,
            "jev_ok": self.jev_ok, "jev_fail": self.jev_fail,
            "vjepa_dist": vjepa_dist,
            "obj_z_log": self.obj_z_log,
            "plan": self.plan_text[:200] if self.plan_text else "",
            "steps": self.steps,
        }

        # 保存结果和帧
        with open(f"{OUTDIR}/{self.exp}/result.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        np.savez(f"{OUTDIR}/{self.exp}/frames.npz",
                 main_images=np.array(self.frames),
                 wrist_images=np.array(self.wrist_frames) if self.wrist_frames else np.array([]))

        print(f"\n结果: terminated={self.terminated}, {wall:.1f}s, "
              f"VLA={len(self.vla_lat)}次(avg{result['vla_avg_ms']}ms), "
              f"env={len(self.env_lat)}次(avg{result['env_avg_ms']}ms)")
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["B","C","E","F","G"])
    ap.add_argument("--task", default="pick up the salad dressing and place it in the basket")
    ap.add_argument("--max-loops", type=int, default=5)
    ap.add_argument("--no-approach", action="store_true")
    a = ap.parse_args()

    r = Runner(a.exp, a.task, a.max_loops, use_approach=not a.no_approach)
    r.run()


if __name__ == "__main__":
    main()
