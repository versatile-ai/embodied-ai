#!/usr/bin/env python3
"""Experiment J: DS Flash directly controls the arm (NO π0.5).
Each step: image + positions → DS outputs JSON delta action → env.step.
Full instrumentation: per-step DS latency, action, distances; dense dual-view frames.
"""
import argparse, base64, io, json, os, re, sys, time, urllib.request
import numpy as np

sys.path.insert(0, "/data/rpent2/rpent")
from rpent.utils.rpc.http_rpc import HttpRpcClient

PROXY = "http://127.0.0.1:8788"
OUTDIR = os.environ.get("EXP_OUT", "/data/rpent2/logs/exp_dense")
TASK = "Pick the salad dressing and place it in the basket"

SYS = """你是Franka Panda机械臂控制器。每轮你会看到当前相机图像和状态，输出一步动作JSON。

动作语义:
- dx, dy, dz: 末端执行器在世界坐标系的位移（毫米）。x左右, y前后, z上下。每步范围±25mm，会被截断。
- gripper: "open"(-1张开) | "close"(+1闭合) | "hold"(保持)
- 物体被夹住后随末端移动。把物体放进篮子（basket）内部即任务完成。

策略提示: 先在物体上方对准→下降到物体高度→闭合夹爪→抬升→移动到篮子上方→下降→张开。
只输出JSON: {"dx":0,"dy":0,"dz":0,"gripper":"hold","note":"一句话意图"}"""

def b64_jpeg(img):
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()

def ds_action(msgs):
    body = {"model": "deepseek-flash", "max_completion_tokens": 1500,
            "messages": msgs, "stream": False}
    req = urllib.request.Request(PROXY + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=90))
    content = resp["choices"][0]["message"].get("content")
    if content is None:
        raise RuntimeError("DS返回空content(thinking耗尽配额)")
    return content, resp.get("usage", {})

def parse_action(text):
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m: return None
    try:
        a = json.loads(m.group(0))
        g = {"open": -1.0, "close": 1.0, "hold": 0.0}.get(str(a.get("gripper","hold")).lower(), 0.0)
        return {
            "dxyz": [float(a.get("dx",0))/1000.0, float(a.get("dy",0))/1000.0, float(a.get("dz",0))/1000.0],
            "gripper": g, "note": str(a.get("note",""))[:80],
        }
    except Exception:
        return None

def run(max_steps=100, max_wall_s=900):
    env = HttpRpcClient("http://127.0.0.1:18900")
    os.makedirs(f"{OUTDIR}/J", exist_ok=True)

    t0 = time.perf_counter()
    obs = env.call("env.reset")
    if isinstance(obs, list): obs = obs[0]
    raw = env.call("env.raw_obs")
    frames, wrist = [], []
    steps_log = []
    msgs = [{"role": "system", "content": SYS}]
    gripper = -1.0
    terminated = False
    total_in_tok = total_out_tok = 0

    def fresh_state():
        r = env.call("env.raw_obs")
        eef = np.array(r.get("robot0_eef_pos", [0,0,0]))
        # 精确匹配绝对位置键（避免匹配到 ..._to_robot0_eef_pos）
        obj = np.array(r["salad_dressing_1_pos"]) if "salad_dressing_1_pos" in r else None
        basket = np.array(r["basket_1_pos"]) if "basket_1_pos" in r else None
        # env自带的真值相对向量
        obj_rel = np.array(r["salad_dressing_1_to_robot0_eef_pos"]) if "salad_dressing_1_to_robot0_eef_pos" in r else None
        return r, eef, obj, basket, obj_rel

    for step in range(1, max_steps+1):
        if time.perf_counter()-t0 > max_wall_s:
            print("TIMEOUT"); break
        raw, eef, obj, basket, obj_rel = fresh_state()
        img = None
        for k in ["main_images","agentview_image"]:
            if k in raw: img = np.asarray(raw[k]); break
        if img is not None:
            frames.append(img)
            w = raw.get("robot0_eye_in_hand_image")
            if w is not None: wrist.append(np.asarray(w))

        dist_obj = float(np.linalg.norm(eef-obj)) if obj is not None else -1
        dist_basket = float(np.linalg.norm(obj-basket)) if (obj is not None and basket is not None) else -1
        obj_rel_str = f"[{obj_rel[0]:+.3f}, {obj_rel[1]:+.3f}, {obj_rel[2]:+.3f}]" if obj_rel is not None else "?"
        basket_rel = (basket - eef) if (basket is not None) else None
        basket_rel_str = f"[{basket_rel[0]:+.3f}, {basket_rel[1]:+.3f}, {basket_rel[2]:+.3f}]" if basket_rel is not None else "?"
        state_txt = (f"▶ 物体相对EEF的向量(米): {obj_rel_str}\n"
                     f"  含义: 向量[i]>0 → 应向轴i正方向移动来接近物体。例: [-0.05,+0.03,-0.10]表示物体在EEF的-x方向5cm、+y方向3cm、下方10cm。\n"
                     f"▶ 篮子相对EEF的向量(米): {basket_rel_str}（同样语义）\n"
                     f"夹爪: {'闭合' if gripper>0 else '张开'}\n"
                     f"EEF到物体距离: {dist_obj:.3f}m | 物体到篮子距离: {dist_basket:.3f}m\n"
                     f"已执行步数: {step}")

        um = {"role": "user", "content": [
            {"type": "text", "text": f"任务: {TASK}\n{state_txt}"}]}
        if img is not None:
            um["content"].append({"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + b64_jpeg(img)}})
        msgs.append(um)

        t1 = time.perf_counter()
        try:
            text, usage = ds_action(msgs)
            ds_ms = (time.perf_counter()-t1)*1000
        except Exception as e:
            print(f"[{step}] DS调用失败: {str(e)[:60]}")
            msgs.pop()
            continue
        total_in_tok += usage.get("prompt_tokens",0)
        total_out_tok += usage.get("completion_tokens",0)
        msgs.append({"role": "assistant", "content": text})

        act = parse_action(text)
        if act is None:
            print(f"[{step}] JSON解析失败: {text[:60]}")
            continue

        # 限幅并执行
        d = np.clip(act["dxyz"], -0.025, 0.025)
        env_act = np.zeros(7)
        env_act[:3] = np.clip(d/0.05, -1.0, 1.0)
        if act["gripper"] != 0.0:
            gripper = act["gripper"]
        env_act[6] = gripper

        t2 = time.perf_counter()
        result = env.call("env.step", (env_act.tolist(),))
        env_ms = (time.perf_counter()-t2)*1000
        obs = result[0]
        terms = result[2]
        terminated = bool(np.any(terms)) if hasattr(terms,'__len__') else bool(terms)

        steps_log.append({"step": step, "eef": [round(float(v),3) for v in eef],
                          "obj": [round(float(v),3) for v in obj] if obj is not None else None,
                          "ds_ms": round(ds_ms,1), "env_ms": round(env_ms,1),
                          "dxyz_mm": [round(x*1000,1) for x in d], "gripper": gripper,
                          "dist_obj": round(dist_obj,3), "dist_obj_basket": round(dist_basket,3),
                          "note": act["note"], "terminated": terminated,
                          "prompt_tok": usage.get("prompt_tokens",0),
                          "out_tok": usage.get("completion_tokens",0)})
        print(f"[{step}] ds={ds_ms/1000:.1f}s d={[round(x*1000) for x in d]} g={gripper} "
              f"d_obj={dist_obj:.3f} d_basket={dist_basket:.3f} {act['note'][:40]}")

        # 控制上下文长度：保留system + 最近20轮
        if len(msgs) > 42:
            msgs = [msgs[0]] + msgs[-40:]

        if terminated:
            print(f"\n*** TERMINATED at step {step}! ***")
            break

    wall = time.perf_counter()-t0
    result = {"exp": "J", "terminated": terminated, "wall_s": round(wall,1),
              "steps": steps_log, "n_steps": len(steps_log),
              "total_in_tok": total_in_tok, "total_out_tok": total_out_tok,
              "note": "DS直接控制（无VLA），视觉+特权位置"}
    json.dump(result, open(f"{OUTDIR}/J/result.json","w"), ensure_ascii=False, indent=2, default=str)
    np.savez(f"{OUTDIR}/J/frames.npz",
             main_images=np.array(frames) if frames else np.array([]),
             wrist_images=np.array(wrist) if wrist else np.array([]))
    print(f"\nJ结果: terminated={terminated}, {wall:.1f}s, {len(steps_log)}步, "
          f"tok in={total_in_tok:,} out={total_out_tok:,}")
    return result

if __name__ == "__main__":
    run()
