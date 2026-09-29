#!/usr/bin/env python3
"""Build a self-contained, auditable RoboDojo decision report from archived data."""

import base64
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "hybrid_rollout/robodojo/reports"
OUTPUT = ROOT / "docs/robodojo-decision-report-20260929.html"
INLINE_VIDEO = False


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compact_number(value):
    return round(float(value), 6)


def sanitized_request(raw):
    # The original requests contain machine-specific local paths; keep the
    # decision fields and camera names, but do not publish broken path strings.
    fields = ("task", "instruction", "step_id", "remaining_steps", "current_state",
              "current_eef", "previous_result", "previous_observation", "counters",
              "student_eef_trajectory", "action_diagnostics", "fk_check",
              "gate_policy", "gate_instruction", "task_context")
    result = {key: raw[key] for key in fields if key in raw}
    if "images" in raw:
        result["images"] = [entry.get("camera") for entry in raw["images"]]
    return sanitize_paths(result)


def sanitize_paths(value):
    if isinstance(value, dict):
        return {key: sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_paths(item) for item in value]
    if isinstance(value, str) and value.startswith(("/Users/max/", "/root/")):
        return "<historical-host-path>/" + Path(value).name
    return value


def media(path):
    blob = path.read_bytes()
    video_url = ("data:video/mp4;base64," + base64.b64encode(blob).decode()) if INLINE_VIDEO else \
                "../" + str(path.relative_to(ROOT))
    return {"data": video_url,
            "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
            "source": str(path.relative_to(ROOT))}


def gpt_report_metrics():
    path = REPORTS / "gpt6_layout0_20260919/gpt6_test_report.html"
    source = path.read_text()
    start = source.index("const DATA=") + len("const DATA=")
    data, _ = json.JSONDecoder().raw_decode(source[start:])
    executed = {row["decision"]: row for row in data["inferences"]
                if row.get("status") == "executed" and row.get("decision") is not None}
    return data["decisions"], executed


def gpt_episode(name, directory, video_path, visual, metrics=None, waits=None):
    path = REPORTS / directory
    run = read_json(path / "run.json")
    result = read_json(path / "result.json")
    history = read_json(path / "history.json")
    tokens = read_json(path / "token_usage.json")["usage"]["total"]
    decisions = []
    for index, row in enumerate(history):
        request_path = path / f"request_{index:03d}.json"
        response_path = path / f"response_{index:03d}.json"
        execution_path = path / f"execution_{index:03d}.npz"
        request = read_json(request_path)
        response = read_json(response_path)
        with np.load(execution_path) as z:
            actions = z["actions"].round(6).tolist()
            states = z["states"].round(6).tolist()
        detail = metrics[0][index] if isinstance(metrics, tuple) else {}
        first = metrics[1].get(index, {}) if isinstance(metrics, tuple) else {}
        latency = {
            "gpt_wait_ms": detail.get("model_wait_ms"),
            "sim_execute_ms": detail.get("execution_ms"),
            "input_tokens": detail.get("input_tokens"),
            "cached_input_tokens": detail.get("cached_input_tokens"),
            "output_tokens": detail.get("output_tokens"),
            "reasoning_tokens": detail.get("reasoning_tokens"),
            "ttft_proxy_ms": first.get("ttft_proxy_ms"),
            "tpot_proxy_ms": first.get("tpot_proxy_ms_per_output_token"),
        }
        if waits is not None:
            latency["gpt_wait_ms"] = round(waits[index] * 1000, 3)
        decisions.append({
            "index": index, "start": row["start_tick"], "end": row["end_tick"],
            "mode": response.get("mode", "eef"), "reason": response.get("reason", ""),
            "input": sanitized_request(request), "output": response,
            "actions": actions, "states": states, "latency": latency,
            "sources": [str(p.relative_to(ROOT)) for p in
                        (request_path, response_path, execution_path)],
        })
    return {
        "name": name, "route": "GPT Direct", "protocol": "GPT-6 Astra / xhigh · EEF",
        "visual": visual, "native": result["success"], "native_valid": True,
        "steps": result["step_id"], "wall_s": result["wall_seconds"],
        "tokens": tokens, "run": {"seed": run["seed"], "task": run["task"],
                                  "model": run["teacher_model"],
                                  "effort": run["teacher_reasoning_effort"]},
        "decisions": decisions, "video": media(REPORTS / video_path),
        "source": str(path.relative_to(ROOT)),
    }


def hybrid_episode(name, directory, video_path, visual, summary):
    path = REPORTS / directory
    run = read_json(path / "run.json")
    result = read_json(path / "result.json")
    history = read_json(path / "history.json")
    tokens = read_json(path / "token_usage.json")["usage"]["total"]
    by_decision = {row["decision"]: row for row in summary["decisions"]}
    decisions = []
    for index, row in enumerate(history):
        request_path = path / f"request_{index:03d}.json"
        response_path = path / f"response_{index:03d}.json"
        execution_path = path / f"execution_{index:03d}.npz"
        request = read_json(request_path)
        response = read_json(response_path)
        with np.load(execution_path) as z:
            actions = z["actions"].round(6).tolist()
            states = z["states"].round(6).tolist()
        measured = by_decision[index]
        decisions.append({
            "index": index, "start": row["start_tick"], "end": row["end_tick"],
            "mode": response.get("mode", "student"), "reason": response.get("reason", ""),
            "input": sanitized_request(request), "output": response,
            "actions": actions, "states": states,
            "discarded": measured.get("discarded_steps"),
            "latency": {"pi_server_ms": measured.get("pi_server_ms"),
                        "pi_roundtrip_ms": measured.get("pi_roundtrip_ms"),
                        "gpt_wait_ms": measured.get("gpt_review_ms"),
                        "sim_execute_ms": measured.get("execute_ms")},
            "sources": [str(p.relative_to(ROOT)) for p in
                        (request_path, response_path, execution_path)],
        })
    return {
        "name": name, "route": "GPT + π0.5", "protocol": "π 50×14 候选 · GPT-6 Astra / xhigh 审核",
        "visual": visual, "native": result["success"], "native_valid": True,
        "steps": result["step_id"], "wall_s": result["wall_seconds"],
        "tokens": tokens, "run": {"seed": run["seed"], "task": run["task"],
                                  "model": run["teacher_model"],
                                  "effort": run["teacher_reasoning_effort"]},
        "decisions": decisions, "video": media(REPORTS / video_path),
        "source": str(path.relative_to(ROOT)),
    }


def pi_episode(name, base, visual):
    path = REPORTS / "pi05_layout0_20260919/artifacts"
    summary = read_json(path / f"{base}_summary.json")
    events = read_jsonl(path / f"{base}.jsonl")
    performance = [row for row in read_jsonl(path / f"{base}_performance.jsonl")
                   if "decision" in row]
    executed = read_jsonl(path / f"{base}_executed.jsonl")
    observations = {row["index"]: row for row in events if row.get("event") == "observation"}
    inferences = {row["index"]: row for row in events if row.get("event") == "inference"}
    steps = {}
    for row in executed:
        steps.setdefault(row["decision"], []).append(row)
    decisions = []
    for index, timing in enumerate(performance):
        prediction = inferences[index]
        rows = steps[index]
        actions = prediction["actions"][:len(rows)]
        state = []
        for row in rows:
            measured = row.get("measured", {})
            left = measured.get("left_arm_joint_state", [None] * 6)
            right = measured.get("right_arm_joint_state", [None] * 6)
            state.append([*left, None, *right, None])
        decisions.append({
            "index": index, "start": index * 50,
            "end": timing["total_steps"], "mode": "student",
            "reason": "π0.5 无文字理由；执行的是本轮 50×14 连续动作。",
            "input": {"instruction": "Pick up the bottles and throw them into the dustbin, using handover when needed.",
                      "state": observations[index]["state"],
                      "cameras": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                      "request_bytes": prediction.get("request_bytes"),
                      "images_archived": False},
            "output": {"actions": prediction["actions"], "chunk_size": prediction["chunk_size"]},
            "actions": actions, "states": state,
            "latency": {"observation_ms": timing.get("obs_ms"),
                        "pi_server_ms": prediction.get("npu_infer_ms"),
                        "pi_roundtrip_ms": timing.get("infer_roundtrip_ms"),
                        "sim_execute_ms": timing.get("chunk_execute_ms")},
            "sources": [str((path / f"{base}{suffix}").relative_to(ROOT)) for suffix in
                        (".jsonl", "_performance.jsonl", "_executed.jsonl")],
        })
    return {
        "name": name, "route": "π0.5", "protocol": "NPU · 50×14 joint action chunk",
        "visual": visual, "native": None, "native_valid": False,
        "steps": 700, "wall_s": summary["result"]["wall_time_s"],
        "tokens": None, "run": {"seed": 0, "task": "put_bottles_into_dustbin",
                                  "model": "π0.5", "effort": None},
        "decisions": decisions, "video": media(path / f"{base}_four_views.mp4"),
        "source": str((path / f"{base}_summary.json").relative_to(ROOT)),
    }


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboDojo · 四瓶入桶逐轮决策验证报告</title>
<style>
:root{color-scheme:light;--ink:#173139;--muted:#53686d;--line:#cbd9d9;--accent:#006e72;--bg:#f5f7f4;--paper:#fff;--soft:#eaf0ed}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,-apple-system,sans-serif}main{max-width:1370px;margin:auto;padding:32px 28px 90px}h1{font-size:34px;line-height:1.23;margin:9px 0 13px;letter-spacing:-.6px}h2{font-size:22px;margin:34px 0 12px;padding-top:20px;border-top:1px solid var(--line)}h3{font-size:16px;margin:22px 0 8px}p{max-width:1150px}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--accent)}.note{color:var(--muted);font-size:13px}.nav{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}.nav a{padding:7px 12px;background:var(--paper);border:1px solid var(--line);border-radius:4px;text-decoration:none;color:var(--accent)}.controls{display:flex;gap:11px;align-items:center;flex-wrap:wrap;padding:13px 0;position:sticky;top:0;background:#f5f7f4f5;z-index:3;border-block:1px solid var(--line)}select,button{font:inherit;padding:7px 11px;border:1px solid #a7bdbd;border-radius:4px;background:var(--paper);color:var(--ink)}button{cursor:pointer}.metrics{display:flex;flex-wrap:wrap;margin:18px 0}.metric{padding:3px 22px 3px 0;margin-right:22px;border-right:1px solid var(--line)}.metric b{display:block;font-size:24px;font-weight:650}.metric small{color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}th{background:#e7efeb;position:sticky;top:0;z-index:1}td,th{padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}tbody tr:hover{background:#e6eeeb}tr.selected{background:#d4e8e3}pre{font:12px/1.55 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word;background:var(--soft);padding:14px;max-height:430px;overflow:auto;border-radius:4px}details{margin:13px 0}summary{cursor:pointer;font-weight:600}video{width:100%;max-height:540px;background:#111;border-radius:5px}.status{font-weight:650;color:var(--accent)}a{color:var(--accent)}.strip{height:26px;background:#dfe8e4;display:flex;overflow:hidden;border-radius:3px}.strip span{height:100%;min-width:1px}.strip .gpt{background:#006e72}.strip .pi{background:#6c94ae}.strip .sim{background:#bb874b}.strip .rest{background:#adb9b6}.legend{font-size:12px;color:var(--muted)}.plot{background:white;border:1px solid var(--line);width:100%;height:175px}.callout{background:#e7efeb;border-left:3px solid var(--accent);padding:10px 16px}.keypoint{padding:10px 0;border-bottom:1px solid var(--line)}.keypoint strong{color:var(--accent)}@media(max-width:800px){main{padding:22px 14px}.grid{grid-template-columns:1fr}h1{font-size:27px}.metric{margin-right:12px;padding-right:12px}}
</style></head><body><main>
<div class="eyebrow">DECISION TRACE / ROBODOJO / LAYOUT 0 / 2026-09-29</div>
<h1>四瓶入桶验证：从观测、决策到双臂执行</h1>
<p>基于已归档的同一 RoboDojo 任务，整理 GPT-6 Astra 直接控制、昇腾 NPU π0.5 和混合策略的九个历史回合。报告按 LIBERO 逐轮决策模板组织，包含可切换的每轮输入、输出、耗时、关节动作和完整四视角录像。此页是历史数据再分析，没有重新运行模型或仿真。</p>
<nav class="nav"><a href="#comparison">九次实验</a><a href="#episode">逐轮记录</a><a href="#timing">性能口径</a><a href="#analysis">行为与结论</a><a href="#source">来源和复现</a></nav>
<div class="controls"><label>路线 / 回合 <select id="episode-select"></select></label><label>决策 <select id="decision-select"></select></label><button id="previous">上一轮</button><button id="next">下一轮</button><button id="export">导出本轮 JSON</button></div>
<h2 id="comparison">三路线 · 九次历史回合</h2><div class="scroll"><table><thead><tr><th>路线</th><th>回合</th><th>视觉完成</th><th>native evaluator</th><th>控制步</th><th>决策轮</th><th>墙钟 / 分</th><th>GPT 总 Token</th></tr></thead><tbody id="compare-rows"></tbody></table></div>
<p class="note">视觉完成由原报告的完整录像人工复核；native 列只用原生仿真判定。π 四轮原生奖励注册无效，不写成官方通过。每格仅一个历史样本，且控制策略与采样不同，不构成统计显著的模型能力排名。</p>
<h2 id="episode">当前回合 <span class="status" id="status"></span></h2><div id="episode-metrics" class="metrics"></div><p id="episode-description"></p>
<div class="scroll" style="max-height:340px"><table><thead><tr><th>轮次</th><th>控制步</th><th>模式</th><th>GPT 等待 / s</th><th>π 往返 / s</th><th>π 服务端 / s</th><th>仿真执行 / s</th><th>输入 Token</th><th>缓存 Token</th><th>输出 Token</th><th>TTFT 代理 / s</th><th>TPOT 代理 / ms</th></tr></thead><tbody id="decision-rows"></tbody></table></div>
<p class="note">固定 5 步 GPT 的输入/输出 Token 是该决策轮所有模型调用的合计；TTFT/TPOT 代理来自其中执行动作的调用，两组列不能直接相除。其它路线缺少逐轮 Token 记录时留空。</p>
<h2 id="decision-title">当前决策</h2><div id="round-metrics" class="metrics"></div><p id="reason"></p>
<div class="grid"><section><h3>输入：任务、机器人状态与候选</h3><p class="note">录像帧用于回看现场；历史单张模型输入图未全部归档，不能把录像帧称为原始输入。下方 JSON 只包含已保存的请求字段，主机绝对路径已去除。</p><details open><summary>查看本轮输入 JSON</summary><pre id="input-json"></pre></details></section><section><h3>输出：模型决策与执行解释</h3><p class="note">π 原始输出是 50×14 连续动作，没有文字思维过程。GPT 的 reason / assessment 是对外记录的解释，不代表内部逐字思维过程，也不是任务成功证据。</p><details open><summary>查看本轮输出 JSON</summary><pre id="output-json"></pre></details></section></div>
<h3>整轮可观测耗时</h3><div id="timing-strip" class="strip"></div><p id="timing-note" class="legend"></p>
<h3>执行轨迹：左右臂首关节与夹爪</h3><p class="note">曲线是本轮执行后的实测关节角，单位 rad；π 单独日志未保存逐步实测夹爪开度，因此只画有记录的关节。表中动作是 14D 目标值：左 6 关节＋夹爪、右 6 关节＋夹爪。0 夹爪闭合，1 张开。</p><svg id="joint-plot" class="plot" viewBox="0 0 780 175" role="img" aria-label="本轮左右臂第一关节角度轨迹"></svg><div class="scroll" style="max-height:350px"><table><thead><tr><th>全局控制步</th><th>目标动作 14D</th><th>实测状态 14D</th></tr></thead><tbody id="action-rows"></tbody></table></div>
<h3>四视角完整录像</h3><video id="video" controls preload="metadata" playsinline></video><p id="video-note" class="note"></p><button id="play-segment">播放当前决策片段</button><details><summary>当前决策的归档文件路径</summary><pre id="round-sources"></pre></details>
<h2 id="timing">耗时、Token 与信息边界</h2>
<div class="keypoint"><strong>GPT 固定 5 步。</strong> 700 步 / 140 次决策，墙钟 5471.34 秒；模型等待 4748.12 秒，占 86.8%，仿真执行约 582.18 秒。20.77M 总 Token 中大部分为缓存输入。逐轮 TTFT/TPOT 是客户端“首个可见事件”和后续可见输出间隔的代理，包含网络、排队及不可见内部处理，不能解释为服务端纯 prefill/decode。</div>
<div class="keypoint"><strong>π 单独。</strong> 最快一轮 172.979 秒，14 次 NPU 服务端推理共 7.696 秒，但 HTTP 往返共 42.197 秒，动作执行共 129.508 秒。服务端时间包含在 HTTP 往返内，不得重复相加。单轮请求体字节数见逐轮输入；文本 TTFT/TPOT 不适用于连续动作数组。</div>
<div class="keypoint"><strong>混合。</strong> 两轮分别 1899.11 和 2616.85 秒；GPT 审核累计约 753.74 和 1027.47 秒，π HTTP 往返约 105.39 和 135.56 秒，动作执行约 316.22 和 351.70 秒。墙钟还有未单独埋点的残差，不能全算作网络或模型。每轮 50 个 π 候选只执行 10–15 步，剩余丢弃并重新观测。</div>
<p class="note">时间条仅显示当前轮有独立记录的互斥大项；其中 π 服务端时间是 π 往返的子集，仅列数字、不另叠加。GPT 固定 50、自适应和混合的逐轮 Token / TTFT 未完整归档，本页显示“—”，不按全局均值伪造。</p>
<h2 id="analysis">结果与下一步</h2>
<div class="grid"><section><h3>行为能力</h3><p>固定 5 步 GPT 和两轮混合从视频看完成四瓶入桶；但 GPT 固定 5 步与混合复测的 native success 仍为 false，分数 0.40。混合首轮是本批唯一可确认的 native success。π 四轮视觉完成数为 2、3、3、4，显示动作先验较快，但单靠 50 步块缺少明确的失败诊断与自适应重试。</p></section><section><h3>对比限制</h3><p>固定 50 步 GPT 把 140 次决策降到 14 次，墙钟从 91.19 分钟降至 19.04 分钟，但视觉完成数由 4 降为 1；自适应 5–50 步为 43.06 分钟、视觉 2。速度改进没有保持同等任务效果，不能仅按“每次输出步数”评价控制方案。</p></section></div>
<p>下一轮应先核对逐瓶 object-in-bin、双臂归位和 reward 条件，解决视频与 native 判定分歧；随后在同一环境、同一布局和多随机种子下比较 5/15/50 步与自适应前缀。为分析 TTFT、TPOT、网络和渲染瓶颈，需要服务端打点与同步时钟；现有记录只支持本文标明的客户端代理指标。</p>
<h2 id="source">来源、校验与复现</h2><p>本页由 <code>scripts/build_decision_report.py</code> 从仓库内 JSON / JSONL / NPZ 和 MP4 生成。__MEDIA_NOTE__ <code>MANIFEST.sha256</code>、<code>scripts/verify_archive.py</code> 可复核原始归档。官方 RoboDojo / Isaac Sim、NPU 镜像与权重不在此 HTML 或 Git 中；新实验环境搭建详见 <code>docs/REPRODUCE.md</code>。原始模型输入图片并非每轮完整归档，报告不声称能从此 HTML 精确重放历史模型请求。</p>
<pre id="source-list"></pre>
</main><script id="report-data" type="application/json">__REPORT_JSON__</script>
<script>
const data=JSON.parse(document.getElementById('report-data').textContent),$=id=>document.getElementById(id);let e,r,clipEnd=null,videoEpisode=-1;
const fmt=(x,n=2)=>x==null?'—':Number(x).toFixed(n),tokens=x=>x==null?'—':Number(x).toLocaleString('en-US'),text=x=>x==null?'—':String(x),esc=x=>text(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),pretty=x=>JSON.stringify(x,null,2),sec=ms=>ms==null?'—':fmt(ms/1000),metric=(v,label)=>`<div class="metric"><b>${esc(v)}</b><small>${esc(label)}</small></div>`;
function comparison(){ $('compare-rows').innerHTML=data.episodes.map((x,i)=>`<tr data-i="${i}"><td>${esc(x.route)}</td><td>${esc(x.name)}</td><td>${esc(x.visual)}</td><td>${x.native_valid?x.native?'通过':'未通过':'无效 / N/A'}</td><td>${x.steps}</td><td>${x.decisions.length}</td><td>${fmt(x.wall_s/60)}</td><td>${x.tokens?tokens(x.tokens.totalTokens):'不适用'}</td></tr>`).join('') }
function selectEpisode(){let idx=+$('episode-select').value;e=data.episodes[idx];$('decision-select').innerHTML=e.decisions.map((d,i)=>`<option value="${i}">#${String(i).padStart(3,'0')} · ${d.start}–${d.end} 步 · ${esc(d.mode)}</option>`).join('');$('status').textContent=`${e.name} · 视觉 ${e.visual} · native ${e.native_valid?(e.native?'通过':'未通过'):'无效'}`;$('episode-metrics').innerHTML=metric(e.steps,'控制步')+metric(e.decisions.length,'决策轮')+metric(fmt(e.wall_s/60)+' min','墙钟')+metric(e.tokens?tokens(e.tokens.totalTokens):'不适用','GPT 总 Token');$('episode-description').textContent=e.protocol+'；任务 '+e.run.task+'；Layout/seed '+e.run.seed+'。';$('decision-rows').innerHTML=e.decisions.map((d,i)=>{let t=d.latency;return `<tr data-i="${i}"><td>${i}</td><td>${d.start}–${d.end}</td><td>${esc(d.mode)}</td><td>${sec(t.gpt_wait_ms)}</td><td>${sec(t.pi_roundtrip_ms)}</td><td>${sec(t.pi_server_ms)}</td><td>${sec(t.sim_execute_ms)}</td><td>${tokens(t.input_tokens)}</td><td>${tokens(t.cached_input_tokens)}</td><td>${tokens(t.output_tokens)}</td><td>${sec(t.ttft_proxy_ms)}</td><td>${fmt(t.tpot_proxy_ms)}</td></tr>`}).join('');for(const row of $('decision-rows').children)row.onclick=()=>{$('decision-select').value=row.dataset.i;selectDecision()};for(const row of $('compare-rows').children)row.classList.toggle('selected',+row.dataset.i===idx);if(videoEpisode!==idx){$('video').src=e.video.data;videoEpisode=idx}$('source-list').textContent=pretty({run:e.source,video:e.video.source,video_bytes:e.video.bytes,video_sha256:e.video.sha256,model:e.run.model,reasoning_effort:e.run.effort,archive_commit:data.archive_commit});$('decision-select').value='0';selectDecision()}
function plot(){let s=r.states.filter(x=>x&&x.length>=14),svg=$('joint-plot');if(!s.length){svg.innerHTML='';return}let series=[{n:'左臂 J1',i:0,c:'#006e72'},{n:'右臂 J1',i:7,c:'#bb874b'}],values=series.flatMap(o=>s.map(x=>x[o.i]).filter(x=>x!=null)),lo=Math.min(...values),hi=Math.max(...values);if(hi-lo<.02){hi+=.01;lo-=.01}let path=(idx)=>s.map((x,i)=>{let v=x[idx];return v==null?'':`${i?'L':'M'}${34+i*720/Math.max(1,s.length-1)},${145-(v-lo)*115/(hi-lo)}`}).join(' ');svg.innerHTML=`<line x1="34" y1="145" x2="758" y2="145" stroke="#9db2b2"/><line x1="34" y1="30" x2="34" y2="145" stroke="#9db2b2"/><text x="5" y="21" fill="#53686d" font-size="11">rad ${fmt(hi)}</text><text x="5" y="154" fill="#53686d" font-size="11">${fmt(lo)}</text><text x="640" y="166" fill="#53686d" font-size="11">控制步 / 本轮</text>${series.map(o=>`<path d="${path(o.i)}" fill="none" stroke="${o.c}" stroke-width="2"/>`).join('')}<text x="44" y="18" fill="#006e72" font-size="12">左臂 J1</text><text x="135" y="18" fill="#bb874b" font-size="12">右臂 J1</text>`}
function timing(){let t=r.latency,parts=[['GPT 等待',t.gpt_wait_ms,'gpt'],['π 往返',t.pi_roundtrip_ms,'pi'],['仿真执行',t.sim_execute_ms,'sim']].filter(x=>x[1]!=null),sum=parts.reduce((a,x)=>a+x[1],0);$('timing-strip').innerHTML=parts.map(x=>`<span class="${x[2]}" style="width:${sum?x[1]/sum*100:0}%" title="${x[0]} ${fmt(x[1]/1000)}s"></span>`).join('');$('timing-note').textContent=parts.map(x=>x[0]+' '+fmt(x[1]/1000)+' s').join(' · ')+(t.pi_server_ms!=null?' · π 服务端 '+fmt(t.pi_server_ms/1000)+' s（包含在往返内）':'')+'。此条仅是有记录的阶段，不代表整轮完整墙钟。'}
function selectDecision(){r=e.decisions[+$('decision-select').value];$('decision-title').textContent=`决策 #${r.index} · 控制步 ${r.start}–${r.end}`;$('round-metrics').innerHTML=metric(r.end-r.start,'执行步')+metric(sec(r.latency.gpt_wait_ms)+' s','GPT 等待')+metric(sec(r.latency.pi_roundtrip_ms)+' s','π 往返')+metric(sec(r.latency.sim_execute_ms)+' s','仿真执行');$('reason').textContent=r.reason;$('input-json').textContent=pretty(r.input);$('output-json').textContent=pretty(r.output);$('round-sources').textContent=pretty(r.sources);$('action-rows').innerHTML=r.actions.map((a,i)=>`<tr><td>${r.start+i+1}</td><td>${esc(JSON.stringify(a))}</td><td>${esc(JSON.stringify(r.states[i]))}</td></tr>`).join('');plot();timing();for(const row of $('decision-rows').children)row.classList.toggle('selected',+row.dataset.i===r.index);clipEnd=null;$('video').pause();let seek=r.start*.04;try{$('video').currentTime=seek}catch{}$('video-note').textContent=`当前片段 ${fmt(seek)}–${fmt(r.end*.04)} 秒；录像 25 fps，按仿真控制步编码，不包含模型等待的墙钟时间。`}
comparison();$('episode-select').innerHTML=data.episodes.map((x,i)=>`<option value="${i}">${esc(x.route)} · ${esc(x.name)}</option>`).join('');$('episode-select').onchange=selectEpisode;$('decision-select').onchange=selectDecision;for(const [id,d] of [['previous',-1],['next',1]])$(id).onclick=()=>{$('decision-select').value=Math.max(0,Math.min(e.decisions.length-1,+$('decision-select').value+d));selectDecision()};$('play-segment').onclick=()=>{clipEnd=r.end*.04;$('video').currentTime=r.start*.04;$('video').play()};$('video').ontimeupdate=()=>{if(clipEnd!=null&&$('video').currentTime>=clipEnd){$('video').pause();clipEnd=null}};$('export').onclick=()=>{let blob=new Blob([pretty({episode:e.name,protocol:e.protocol,decision:r})],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`robodojo-${e.route.replace(/\W/g,'_')}-${r.index}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};selectEpisode();
</script></html>'''


def main():
    global INLINE_VIDEO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inline-videos", action="store_true",
                        help="Embed all nine MP4s for a single offline HTML file")
    args = parser.parse_args()
    INLINE_VIDEO = args.inline_videos
    gpt_rows, infer_rows = gpt_report_metrics()
    hybrid_summary = read_json(REPORTS / "hybrid_gpt_pi_layout0_20260919/artifacts/three_policy_comparison_raw.json")
    hybrids = {item["id"]: item for item in hybrid_summary["hybrid_runs"]}
    fixed50 = read_json(REPORTS / "gpt6_layout0_h50_20260919/run_20260919_235529/performance_comparison.json")
    episodes = [
        gpt_episode("固定 5 步", "gpt6_layout0_20260919/run_20260919_120207",
                    "gpt6_layout0_20260919/artifacts/gpt6_layout0_full_four_views.mp4",
                    "4/4", metrics=(gpt_rows, infer_rows)),
        gpt_episode("固定 50 步", "gpt6_layout0_h50_20260919/run_20260919_235529",
                    "gpt6_layout0_20260919/artifacts/gpt6_horizon50_sensors.mp4",
                    "1/4", waits=fixed50["per_decision_model_wait_seconds"]),
        gpt_episode("自适应 5–50 步", "gpt6_layout0_adaptive50_v3_20260920/run_20260920_083346",
                    "gpt6_layout0_20260919/artifacts/gpt6_adaptive50_sensors.mp4", "2/4"),
    ]
    pi_runs = (
        ("第 1 轮", "pi_closed_loop_layout0_700_video_20260919_174926", "2/4"),
        ("第 2 轮", "pi_closed_loop_layout0_700_video_repeat_20260919_180809", "3/4"),
        ("第 3 轮", "pi_closed_loop_layout0_700_video_repeat3_20260919_182547", "3/4"),
        ("第 4 轮", "pi_closed_loop_layout0_700_video_repeat4_20260919_210134", "4/4"),
    )
    episodes.extend(pi_episode(*item) for item in pi_runs)
    episodes.extend((
        hybrid_episode("首轮", "hybrid_gpt_pi_layout0_20260919/run_20260919_152000",
                       "hybrid_gpt_pi_layout0_20260919/run_20260919_152000/remote_sim/sensors.mp4",
                       "4/4", hybrids["run_20260919_152000"]),
        hybrid_episode("复测", "hybrid_gpt_pi_layout0_20260919/run_20260919_212200",
                       "hybrid_gpt_pi_layout0_20260919/artifacts/hybrid_repeat_20260919_212200_four_views.mp4",
                       "4/4", hybrids["run_20260919_212200"]),
    ))
    assert len(episodes) == 9
    assert sum(len(e["decisions"]) for e in episodes) == 3 * 0 + 140 + 14 + 64 + 4 * 14 + 34 + 49
    data = {"report_date": "2026-09-29", "archive_commit": "3b9392594c1177e58b5d231cbf1f1e8f209d2ca4",
            "episodes": episodes}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    output = OUTPUT.with_name(OUTPUT.stem + "-offline.html") if INLINE_VIDEO else OUTPUT
    media_note = ("九个录像已作为 data URL 内嵌，可离线单文件阅读。" if INLINE_VIDEO else
                  "录像使用仓库内相对路径；从仓库根目录运行 python3 -m http.server 8778 后打开此页。")
    output.write_text(HTML.replace("__MEDIA_NOTE__", media_note).replace("__REPORT_JSON__", payload))
    print(f"wrote {output} ({output.stat().st_size / 1024**2:.1f} MiB), "
          f"{sum(len(e['decisions']) for e in episodes)} decisions")


if __name__ == "__main__":
    main()
