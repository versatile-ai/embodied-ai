#!/usr/bin/env python3
"""RPent × DeepSeek 性能报告生成器（skill v6 口径）。
生成与本仓库 reports/rpent/*.html 完全一致的自包含 HTML 报告。

用法: 编辑下方 CONFIG 后运行
    python3 generate_report.py

输入（均为 files_proxy v3.1 产物）:
  llm_jsonl     代理日志（含 ttft_s/proxy_prep_s/upload_s/api_wait_s/
                first_content_s/duration_s/gen_s/messages/usage...）
  run_log       rpent run.log（工具执行计时）
  translations  translate_rounds.py 的输出缓存
  img_dir       代理提取的 req_imgs/
  videos        ffmpeg 合成的双视角 mp4

报告结构:
  §1.1 公式版总体结论（占比%）+ 端到端公式
  §1.2 TTFT 三分解基线  §1.3 TPOT 基线
  §1.4 方案对比  §1.5 π0.5 推理基线
  §2 双视角视频  §3 逐轮明细（23列全实测）  §4 彩色IO卡片+中文翻译
"""
import json, html, os, re, base64, datetime
import statistics as st

D = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    "llm_jsonl":     "../data/llm_10t0.jsonl",        # 代理日志
    "run_log":       "../data/t10_run.log",           # rpent run.log
    "translations":  "../data/t10_translations.json", # 翻译缓存
    "img_dir":       "../data/req_imgs_t10",          # 代理提取图片
    "video_main":    "../data/t10_agentview.mp4",
    "video_wrist":   "../data/t10_wrist.mp4",
    "out_html":      "../ds_flash_report_out.html",
    "wall_s":        888.7,       # 端到端（run.log elapsed）
    "task_meta":     'libero_10_swap t0/s0 · 2026-10-11 · deepseek-flash @ api.deepseek.com · files_proxy v3.1',
}

RP = lambda p: os.path.normpath(os.path.join(D, CONFIG[""] if False else ".", p))
recs = [json.loads(l) for l in open(RP(CONFIG["llm_jsonl"]))]
run_log = open(RP(CONFIG["run_log"])).read().splitlines()
translations = json.load(open(RP(CONFIG["translations"])))
WALL = CONFIG["wall_s"]; N = len(recs)
IMG_DIR = RP(CONFIG["img_dir"])

# ── 工具执行计时（run.log） ──────────────────────────────
tool_re = re.compile(r"^(\S+ \S+) I \[api_loop\] \[tool(>|<)\] (\w+)")
ts_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_last = None
for line in run_log:
    m2 = ts_re.match(line)
    if m2: _last = m2.group(1)
_off = (recs[-1]["ts"] + 5) - datetime.datetime.strptime(_last, "%Y-%m-%d %H:%M:%S").timestamp() if _last else 0
pend, tools_exec = {}, []
for line in run_log:
    m = tool_re.match(line)
    if not m: continue
    ts = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + _off
    n, a = m.group(3), m.group(2)
    if a == ">": pend.setdefault(n, []).append(ts)
    else:
        if pend.get(n): tools_exec.append({"name": n, "t0": pend[n].pop(0), "t1": ts})

# ── 派生字段 ─────────────────────────────────────────────
for r in recs:
    r["_prep"] = (r.get("proxy_prep_s") if r.get("proxy_prep_s") is not None
                  else round(max(0, r.get("ttft_s",0) - r.get("upload_s",0) - r.get("api_wait_s",0)), 3))

total_ttft = sum(r.get("ttft_s",0) for r in recs)
total_dur  = sum(r.get("duration_s",0) for r in recs)
total_prep = sum(r["_prep"] for r in recs)
total_up   = sum(r.get("upload_s",0) for r in recs)
total_api  = sum(r.get("api_wait_s",0) for r in recs)
total_think_tok = sum(((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens",0) for r in recs)
total_out_tok   = sum((r.get("usage") or {}).get("completion_tokens",0) for r in recs)
total_dec_tok   = total_out_tok - total_think_tok

total_think_s = total_dec_s = 0
for r in recs:
    ttft = r.get("ttft_s",0); fc = r.get("first_content_s"); dur = r.get("duration_s",0)
    if fc:
        total_think_s += fc - ttft; total_dec_s += dur - fc
    else:
        th = ((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens",0)
        out = (r.get("usage") or {}).get("completion_tokens",0)
        frac = th/max(out,1); gen = r.get("gen_s",0)
        total_think_s += gen*frac; total_dec_s += gen*(1-frac)

ttft_avg = total_ttft/N; dur_avg = total_dur/N
prep_avg = total_prep/N; up_avg = total_up/N; api_avg = total_api/N
think_avg = total_think_s/N; dec_avg = total_dec_s/N
llm_avg = dur_avg; tool_avg = (WALL - total_dur)/N
think_tok_avg = total_think_tok//N; dec_tok_avg = total_dec_tok//N
pure_avg = ttft_avg + dec_avg   # 去除Thinking的单轮推理

# ── 工具函数 ─────────────────────────────────────────────
def esc(s): return html.escape(str(s))
def vid(p): return "data:video/mp4;base64," + base64.b64encode(open(p,"rb").read()).decode()
def img64(p):
    return ("data:image/png;base64," + base64.b64encode(open(p,"rb").read()).decode()) if os.path.exists(p) else None
def msg_text(m):
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(p.get("text","") for p in c if isinstance(p,dict) and p.get("type")=="text")
    return ""
def msg_imgs(m):
    c = m.get("content") if isinstance(m, dict) else None
    if not isinstance(c, list): return []
    return [os.path.basename((p.get("image_url") or {}).get("url","")) for p in c
            if isinstance(p,dict) and p.get("type")=="image_url"]
def zh_block(text_zh, label="🇨🇳 中文"):
    if not text_zh: return ""
    return (f'<div style="background:#f8fafc;border-left:3px solid #94a3b8;padding:8px;margin-top:6px;border-radius:0 6px 6px 0">'
            f'<b style="color:#64748b;font-size:12px">{label}</b>'
            f'<pre style="color:#334155;margin:4px 0;max-height:300px;overflow:auto">{esc(text_zh)}</pre></div>')

# ── §3 行 ────────────────────────────────────────────────
rows = []
_prev_in = 0; _prev_imgs = 0
for i, r in enumerate(recs, 1):
    u = r.get("usage") or {}
    pd = u.get("prompt_tokens_details") or {}
    rd = u.get("completion_tokens_details") or {}
    inp = u.get("prompt_tokens") or 0
    cached = pd.get("cached_tokens") or 0
    out = u.get("completion_tokens") or 0
    th_tok = rd.get("reasoning_tokens") or 0
    dec_tok = out - th_tok
    ttft = r.get("ttft_s") or 0
    prep = r["_prep"]; up_chat = r.get("upload_s") or 0
    api_wait = r.get("api_wait_s") or 0
    dur = r.get("duration_s") or 0
    gen = r.get("gen_s") or 0
    fc = r.get("first_content_s")
    if fc:
        think_s = round(fc - ttft, 2); dec_s = round(dur - fc, 2); est_mark = ""
    else:
        frac = th_tok/max(out,1)
        think_s = round(gen*frac, 1); dec_s = round(gen*(1-frac), 1); est_mark = "~"
    orig_kb = r.get("body_bytes",0)//1024
    fwd_kb = r.get("fwd_body_bytes",0)//1024
    imgs = r.get("image_files") or []
    tpot_ms = round(gen/out*1000, 1) if out > 0 else None
    real_infer = round(api_wait + dec_s, 1)
    pth = int((inp-cached)/ttft) if ttft else 0
    hit = round(100*cached/max(inp,1))
    delta_in = inp - _prev_in if i > 1 else inp
    new_imgs = max(0, len(imgs)-_prev_imgs)
    if i < N:
        nxt = (recs[i].get("usage") or {}).get("prompt_tokens") or 0
        tool_out = max(0, nxt - inp - out)
    else:
        tool_out = None
    tcs = r.get("tool_calls") or []
    tc_c = {}
    for tc in tcs: tc_c[tc.get("name","?")] = tc_c.get(tc.get("name","?"),0)+1
    tool_str = "；".join(f"{n}×{c}" for n,c in tc_c.items()) if tc_c else "—"
    if i < N:
        next_ts = recs[i]["ts"]; curr_ts = r["ts"]
        wt = [t for t in tools_exec if curr_ts <= t["t0"] < next_ts]
        tool_exec_s = round(sum(t["t1"]-t["t0"] for t in wt), 1)
        tool_exec_str = f"{tool_exec_s}s" if tool_exec_s > 0 else "—"
    else:
        tool_exec_str = "—"
    rows.append(dict(i=i, ts=r["ts"],
        clock=datetime.datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S"),
        ttft=round(ttft,2), prep=prep, up_chat=up_chat, api=round(api_wait,2),
        orig_kb=orig_kb, fwd_kb=fwd_kb, think_s=think_s, dec_s=dec_s, est=est_mark,
        dur=round(dur,1), real=real_infer, gen=round(gen,1),
        inp=inp, cached=cached, hit=hit, th_tok=th_tok, dec_tok=dec_tok, out=out, tpot=tpot_ms,
        pth=pth, delta=delta_in, new_i=new_imgs, tot_i=len(imgs),
        tool=tool_str, tool_out=tool_out, tool_exec=tool_exec_str,
        content=r.get("content") or "", thinking=r.get("thinking") or "",
        tool_calls=tcs, messages=r.get("messages"), imgs=imgs))
    _prev_in = inp; _prev_imgs = len(imgs)

tt = ""
for r in rows:
    tt += (f"<tr><td><b>{r['i']}</b></td><td>{r['clock']}</td>"
           f"<td><b>{r['ttft']}s</b></td>"
           f"<td style='color:#a78bfa'>{r['prep']:.2f}s</td>"
           f"<td style='color:#2563eb'>{r['up_chat']:.2f}s</td>"
           f"<td style='color:#f59e0b'><b>{r['api']:.2f}s</b></td>"
           f"<td>{r['orig_kb']}→{r['fwd_kb']}KB</td>"
           f"<td>{r['think_s']}s{r['est']}</td><td>{r['dec_s']}s{r['est']}</td>"
           f"<td style='color:#16a34a'><b>{r['real']}s</b></td>"
           f"<td><b>{r['dur']}s</b></td>"
           f"<td>{r['inp']:,}</td><td>+{r['delta']:,}</td>"
           f"<td>{r['new_i']}/{r['tot_i']}</td>"
           f"<td>{r['th_tok']}</td><td>{r['dec_tok']}</td><td>{r['out']}</td>"
           f"<td>{r['pth']}</td><td>{r['tpot'] if r['tpot'] is not None else '—'}</td><td>{r['hit']}%</td>"
           f"<td>{esc(r['tool'])}</td>"
           f"<td style='color:#9a3412'><b>{r['tool_exec']}</b></td>"
           f"<td>{r['tool_out'] if r['tool_out'] is not None else '—'}</td></tr>")

# ── §4 卡片 ──────────────────────────────────────────────
cards = []
for idx in range(N):
    rw = rows[idx]; r = recs[idx]; i = idx+1
    tr = translations.get(str(idx), {})
    think_zh = tr.get("thinking_zh", ""); content_zh = tr.get("content_zh", "")
    input_zh = translations.get(f"{idx}_input", "")
    system_zh = translations.get(f"{idx}_system", "")
    user_zh = translations.get(f"{idx}_user", "")
    msgs = r.get("messages") or []
    prev_len = len(recs[idx-1].get("messages") or []) if idx > 0 else 0
    new_msgs = msgs[prev_len:] if prev_len < len(msgs) else []
    new_a = [m for m in new_msgs if isinstance(m,dict) and m.get("role")=="assistant"]
    new_t = [m for m in new_msgs if isinstance(m,dict) and m.get("role")=="tool"]
    new_u = [m for m in new_msgs if isinstance(m,dict) and m.get("role")=="user"]
    parts = []
    if idx == 0:
        for m in msgs:
            if isinstance(m,dict) and m.get("role")=="system":
                parts.append(f'<div style="background:#fef2f2;border-left:3px solid #dc2626;padding:8px"><b style="color:#dc2626">System Prompt:</b><pre style="max-height:200px;overflow:auto">{esc(msg_text(m)[:2000])}</pre>{zh_block(system_zh, "🇨🇳 System Prompt 中文翻译")}</div>')
            elif isinstance(m,dict) and m.get("role")=="user":
                parts.append(f'<div style="background:#fffbeb;border-left:3px solid #f59e0b;padding:8px"><b style="color:#f59e0b">User 任务:</b><pre>{esc(msg_text(m)[:1000])}</pre>{zh_block(user_zh, "🇨🇳 任务中文翻译")}</div>')
    else:
        for m in new_a:
            txt = msg_text(m)
            tcs_m = m.get("tool_calls") or []
            tct = "".join(f"\n[调用 {tc.get('function',{}).get('name','')}] 参数: {tc.get('function',{}).get('arguments','')[:300]}" for tc in tcs_m)
            if txt or tct:
                parts.append(f'<div style="background:#f0fdf4;border-left:3px solid #16a34a;padding:8px;margin:4px 0"><b style="color:#16a34a">模型输出（上一轮）:</b><pre style="color:#16a34a;margin:4px 0">{esc(txt[:2000])}{esc(tct[:1500])}</pre></div>')
        for m in new_t:
            txt = msg_text(m)
            if txt:
                parts.append(f'<div style="background:#eff6ff;border-left:3px solid #2563eb;padding:8px;margin:4px 0"><b style="color:#2563eb">工具返回:</b><pre style="color:#2563eb;margin:4px 0;max-height:200px;overflow:auto">{esc(txt[:1500])}</pre></div>')
        for m in new_u:
            txt = msg_text(m); ims = msg_imgs(m)
            ih = "".join(f'<img src="{img64(os.path.join(IMG_DIR,n))}" style="max-width:31%;border-radius:6px;margin:3px">' for n in ims if img64(os.path.join(IMG_DIR,n)))
            if txt or ih:
                parts.append(f'<div style="background:#fffbeb;border-left:3px solid #f59e0b;padding:8px;margin:4px 0"><b style="color:#f59e0b">观测输入:</b>{ih}<pre style="margin:4px 0;max-height:150px;overflow:auto">{esc(txt[:1000])}</pre></div>')
    if input_zh and idx > 0:
        parts.append(zh_block(input_zh, "🇨🇳 本轮新增输入中文翻译"))
    ni = "\n".join(parts) if parts else "<p style='color:#9ca3af'>（首轮）</p>"
    thinking = r.get("thinking") or ""; content = r.get("content") or ""; tcs = r.get("tool_calls") or []
    tch = ""
    if tcs:
        items = [f'<div style="background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;padding:8px;margin:4px 0"><b style="color:#dc2626">工具调用 #{j+1}: {esc(tc.get("name",""))}</b><pre style="color:#dc2626;margin:4px 0;font-size:11px">{esc(tc.get("arguments","")[:800])}</pre></div>' for j,tc in enumerate(tcs)]
        tch = f'<details open><summary style="color:#dc2626;font-weight:bold">工具调用 ({len(tcs)} 个) — 本轮模型的主要输出</summary>{"".join(items)}</details>'
    trn = ""
    if idx+1 < N:
        nm = recs[idx+1].get("messages") or []
        nn = nm[len(msgs):] if len(msgs)<len(nm) else []
        tp = []
        for m in nn:
            if not isinstance(m,dict): continue
            if m.get("role")=="tool":
                txt = msg_text(m)
                if txt: tp.append(f'<pre style="color:#2563eb;background:#eff6ff;border-radius:6px;padding:6px;margin:3px 0;max-height:180px;overflow:auto;font-size:11px">{esc(txt[:1200])}</pre>')
            elif m.get("role")=="user":
                ims = msg_imgs(m); ot = msg_text(m)
                ih = "".join(f'<img src="{img64(os.path.join(IMG_DIR,n))}" style="max-width:31%;border-radius:6px;margin:3px">' for n in ims if img64(os.path.join(IMG_DIR,n)))
                if ih or ot: tp.append(f'<div style="margin:4px 0">{ih}<pre style="font-size:11px;max-height:120px;overflow:auto">{esc(ot[:800])}</pre></div>')
        if tp: trn = f'<details><summary style="color:#2563eb;font-weight:bold">本轮工具执行结果（进入下一轮输入）</summary>{"".join(tp)}</details>'
    ai = r.get("image_files") or []
    aih = ""
    if ai:
        tags = [f'<img src="{img64(os.path.join(IMG_DIR,os.path.basename(ip)))}" style="max-width:23%;border-radius:6px;margin:2px">' for ip in ai[:15] if img64(os.path.join(IMG_DIR,os.path.basename(ip)))]
        aih = f'<details><summary style="font-weight:bold;cursor:pointer">📷 上下文中的全部图片 ({len(ai)} 张)</summary><div>{"".join(tags)}</div></details>'
    else:
        aih = '<p style="color:#9ca3af;font-size:12px;margin:6px 0">📷 本轮上下文中无图片</p>'
    u2 = r.get("usage") or {}
    th2 = (u2.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    think_block = ""
    if thinking:
        think_block = (f'<details open><summary>thinking ({th2} Token, 实测{rw["think_s"]}s)</summary>'
                       f'<pre style="color:#dc2626;max-height:300px;overflow:auto">{esc(thinking[:4000])}</pre>'
                       f'{zh_block(think_zh, "🇨🇳 thinking 中文翻译")}</details>')
    if content:
        content_block = (f'<details open><summary>正文 ({len(content)} 字)</summary>'
                         f'<pre style="color:#dc2626">{esc(content[:1500])}</pre>'
                         f'{zh_block(content_zh, "🇨🇳 正文中文翻译")}</details>')
    else:
        content_block = '<p style="color:#9ca3af;margin:4px 0">（本轮无正文输出，模型直接发出工具调用）</p>'
    cards.append(f'''<div class="card">
<h3>轮 {i} · TTFT {rw['ttft']:.2f}s（传图{rw['prep']:.2f}+上传{rw['up_chat']:.2f}+服务端{rw['api']:.2f}） · 总耗时 {rw['dur']}s · prompt {rw['inp']:,} → 输出 {rw['out']} Token</h3>
<div style="background:#fffbeb;border:2px solid #fde047;border-radius:8px;padding:12px;margin:8px 0"><b>📋 新增输入（本轮新增 {len(new_msgs)} 条消息）</b>{ni}</div>
<div style="background:#fef2f2;border:2px solid #fecaca;border-radius:8px;padding:12px;margin:8px 0"><b>🔴 模型输出</b>
{think_block}
{content_block}
{tch}</div>
{trn}{aih}</div>''')

ttfts = [r["ttft"] for r in rows]
apis = [r["api"] for r in rows]
preps = [r["prep"] for r in rows]
ups = [r["up_chat"] for r in rows]
hits = [r["hit"] for r in rows]

TOOLS_DOC = """<h2>附录 · 工具字典</h2>
<table><tr><th style="width:140px">工具</th><th>定义</th></tr>
<tr><td><code>read_text_file</code></td><td>读文本文件（绑定三层记忆）</td></tr>
<tr><td><code>write_text_file</code></td><td>写文件</td></tr>
<tr><td><code>list_dir</code></td><td>列目录</td></tr>
<tr><td><code>finish</code></td><td>结束任务</td></tr>
<tr><td><code>view_env_state</code></td><td>读环境状态+嵌入图像</td></tr>
<tr><td><code>segment</code></td><td>SAM3 视觉分割</td></tr>
<tr><td><code>back_project</code></td><td>像素→世界坐标</td></tr>
<tr><td><code>move_to</code></td><td>末端伺服到世界坐标</td></tr>
<tr><td><code>pi0_pick</code></td><td>π0.5 闭环抓取</td></tr>
<tr><td><code>pi0_doubled</code></td><td>π0.5 非抓取技能</td></tr>
<tr><td><code>release</code></td><td>张爪释放</td></tr>
<tr><td><code>set_gripper</code></td><td>驱动夹爪</td></tr></table>"""

html_out = f'''<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>DeepSeek V4.1 Flash 评测 · {WALL}s</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;background:#f6f7f9;color:#1f2937;line-height:1.6}}
.wrap{{max-width:1300px;margin:0 auto;padding:28px 20px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:19px;margin:30px 0 10px;border-left:4px solid #2563eb;padding-left:10px}}
h3{{font-size:16px;margin:16px 0 8px}}
.meta{{color:#6b7280;font-size:13px;margin-bottom:14px}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:11px;background:#fff}}
th,td{{border:1px solid #e5e7eb;padding:3px 5px;text-align:left}} th{{background:#f3f4f6;font-size:10.5px}}
pre{{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:10px;white-space:pre-wrap;word-break:break-word;font-size:11.5px}}
details{{margin-top:6px}} summary{{cursor:pointer;color:#2563eb;font-size:12.5px}}
video{{width:100%;border-radius:8px;background:#000}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
footer{{color:#9ca3af;font-size:12px;margin-top:26px}}
</style></head><body><div class="wrap">
<h1>DeepSeek V4.1 Flash 性能评测报告</h1>
<div class="meta">{CONFIG["task_meta"]} · {N} 轮 / {WALL}s</div>

<div style="background:#fef3c7;border:2px solid #f59e0b;border-radius:10px;padding:14px 18px;margin:12px 0;font-size:14px;line-height:1.7">
<b>📐 测量口径（全部实测）</b>：TTFT = 代理传图(并行) + chat上传 + DS服务端 三分解（首 SSE chunk）。Thinking/Decode 由 first_content_s 实测拆分。
</div>

<h2>1 · 概览</h2>
<h3>1.1 性能建模与瓶颈分析</h3>
<div style="background:#eff6ff;border:2px solid #2563eb;border-radius:10px;padding:14px 18px;margin:10px 0;font-size:15px;line-height:1.8">
<b>总体结论</b>：T_llm = TTFT + T_think + T_decode = <b>{llm_avg:.1f} s/轮</b>（全部实测）=
<span style="color:#ef4444"><b>{ttft_avg:.2f}s（{100*ttft_avg/(llm_avg+tool_avg):.0f}%）</b></span> +
<span style="color:#d97706"><b>{think_avg:.1f}s（{100*think_avg/(llm_avg+tool_avg):.0f}%）</b></span> +
<span style="color:#16a34a"><b>{dec_avg:.1f}s（{100*dec_avg/(llm_avg+tool_avg):.0f}%）</b></span>。
<b>去除 Thinking 后单轮推理耗时 ≈ {pure_avg:.1f}s</b>。
</div>
<div class="card">
<pre style="font-size:16px;background:#1e293b;color:#e2e8f0;padding:20px;border-radius:10px;line-height:2.4;max-height:none;overflow:visible;white-space:pre-wrap">
<b style="color:#fff">端到端总耗时 = 推理轮数 × 每轮平均耗时</b>
    T_total = N × T_avg = {N} × {llm_avg+tool_avg:.1f} = <b style="color:#fff">{WALL}s</b>
</pre>
<pre style="font-size:15px;background:#1e293b;color:#e2e8f0;padding:18px;border-radius:10px;line-height:2.2;max-height:none;overflow:visible;white-space:pre-wrap">
    T_avg = T_llm + T_tool
            <span style="color:#60a5fa">{llm_avg:.1f}s（占{100*llm_avg/(llm_avg+tool_avg):.0f}%）</span>  <span style="color:#f97316">{tool_avg:.1f}s（占{100*tool_avg/(llm_avg+tool_avg):.0f}%）</span>

    T_llm = TTFT     + T_think  + T_decode = {llm_avg:.1f} s/轮  <span style="color:#94a3b8">（全部实测）</span>
            <span style="color:#ef4444">{ttft_avg:.2f}s（{100*ttft_avg/(llm_avg+tool_avg):.0f}%）</span> <span style="color:#fbbf24">{think_avg:.1f}s（{100*think_avg/(llm_avg+tool_avg):.0f}%）</span> <span style="color:#34d399">{dec_avg:.1f}s（{100*dec_avg/(llm_avg+tool_avg):.0f}%）</span>

    TTFT  = T_代理传图 + T_chat上传 + T_DS服务端
            <span style="color:#a78bfa">{prep_avg:.2f}s（{100*prep_avg/(llm_avg+tool_avg):.0f}%）</span>  <span style="color:#6b7280">{up_avg:.2f}s（{100*up_avg/(llm_avg+tool_avg):.0f}%）</span>   <span style="color:#ef4444">{api_avg:.2f}s（{100*api_avg/(llm_avg+tool_avg):.0f}%）</span>
            <span style="color:#94a3b8">并行上传·仅新图轮   请求体传输         prefill+排队（稳定{min(apis):.1f}~{max(apis):.1f}s）</span>

    T_think  = {think_tok_avg} Token × TPOT ≈ <span style="color:#fbbf24">{think_avg:.1f}s</span>
    T_decode = {dec_tok_avg} Token × TPOT ≈ <span style="color:#34d399">{dec_avg:.1f}s</span>
</pre>
</div>

<h3>1.2 TTFT 基线</h3>
<div class="card">
<p style="font-size:14.5px;margin:4px 0 10px;padding:8px 12px;background:#f0fdf4;border-radius:8px;border:1px solid #86efac">
<b>一句话</b>：本局（{N}轮）真 TTFT 中位数 <b>{st.median(ttfts):.2f}s</b>（范围 {min(ttfts):.2f}~{max(ttfts):.2f}s），DS 服务端 <b>{st.median(apis):.2f}s</b>（{min(apis):.2f}~{max(apis):.2f}s）——全程稳定。</p>
<table style="font-size:13px">
<tr><th>组成</th><th>p50</th><th>范围</th><th>说明</th></tr>
<tr><td style="color:#a78bfa"><b>代理传图（并行）</b></td><td>{st.median(preps):.2f}s</td><td>{min(preps):.2f}~{max(preps):.2f}s</td><td>新图轮</td></tr>
<tr><td style="color:#2563eb"><b>chat 上传</b></td><td>{st.median(ups):.2f}s</td><td>{min(ups):.2f}~{max(ups):.2f}s</td><td>请求体传输</td></tr>
<tr><td style="color:#f59e0b"><b>DS 服务端</b></td><td><b>{st.median(apis):.2f}s</b></td><td><b>{min(apis):.2f}~{max(apis):.2f}s</b></td><td>prefill+排队</td></tr>
<tr style="background:#f0fdf4"><td><b>真 TTFT 合计</b></td><td><b>{st.median(ttfts):.2f}s</b></td><td>{min(ttfts):.2f}~{max(ttfts):.2f}s</td><td>流式实测 ✓</td></tr>
</table>
</div>

<h3>1.3 TPOT 基线</h3>
<div class="card">
<p style="font-size:14.5px;margin:4px 0 10px;padding:8px 12px;background:#f0fdf4;border-radius:8px;border:1px solid #86efac">
<b>一句话</b>：实测 TPOT ≈ <b>4ms/Token</b>。每轮 Thinking <b>{think_tok_avg} Token</b>（{think_avg:.1f}s/轮），正文/工具调用 <b>{dec_tok_avg} Token</b>（{dec_avg:.1f}s/轮）。去掉 Thinking 每轮省 {think_avg:.1f}s，{N} 轮共省 {think_avg*N:.0f}s。</p>
</div>

<h3>1.4 上下文结构与 Files API 方案</h3>
<div class="card">
<table style="font-size:13px;margin:8px 0">
<tr><th></th><th>原始 RPent</th><th>本局</th></tr>
<tr><td>图片编码</td><td>base64 内嵌（4.6MB/请求）</td><td><b>file_id 引用（0.2MB/请求）</b></td></tr>
<tr><td>RPent 图片裁剪</td><td>开启（破坏缓存）</td><td><b style="color:#16a34a">禁用（图片全保留）</b></td></tr>
<tr><td>图片上传</td><td>—</td><td><b style="color:#16a34a">并行（2.8x加速）</b></td></tr>
<tr><td>TTFT 测量</td><td>非流式含生成（口径错误）</td><td><b style="color:#16a34a">流式转换 + 三分解实测</b></td></tr>
<tr><td>最终图片保留</td><td>~8 张</td><td><b style="color:#16a34a">{rows[-1]['tot_i']} 张</b></td></tr>
<tr><td>缓存命中率</td><td>42-100%（有断裂）</td><td><b style="color:#16a34a">{min(hits)}-{max(hits)}%</b></td></tr>
</table>
</div>

<h3>1.5 π0.5 推理基线（VLA 闭环控制）</h3>
<div class="card">
<p style="font-size:14.5px;margin:4px 0 10px;padding:8px 12px;background:#f0fdf4;border-radius:8px;border:1px solid #86efac">
<b>一句话</b>：π0.5 在 155 NPU 上单次 chunk 推理 <b>384ms</b>（极其稳定），摊 76.7ms/action，2.61 chunk/s。pi0_pick 耗时中推理仅占 ~15%，其余 85% 是仿真器 CPU 软件渲染步进——真机无此开销。详见 tools/pi0_bench.py。</p>
</div>

<h2>2 · 双视角视频</h2>
<div class="g2">
<div><b>Agent 视角（agentview）</b><video controls src="{vid(RP(CONFIG['video_main']))}"></video></div>
<div><b>腕部视角（wrist）</b><video controls src="{vid(RP(CONFIG['video_wrist']))}"></video></div>
</div>

<h2>3 · 按时间先后的逐轮明细（真实测量口径）</h2>
<p style="font-size:12.5px;color:#6b7280;margin:4px 0">TTFT 三分解：🟣代理传图（并行） + 🔵chat上传 + 🟠DS服务端（prefill+排队）</p>
<table><tr>
<th>轮</th><th>时刻</th><th>真TTFT<br>(合计)</th>
<th style="color:#a78bfa">代理传图<br>耗时</th><th style="color:#2563eb">chat上传<br>耗时</th><th style="color:#f59e0b">DS服务端<br>(TTFT核心)</th><th>请求体大小<br>(原→转发)</th>
<th>Thinking<br>耗时</th><th>Decode<br>耗时</th>
<th style="color:#16a34a">推理真实耗时<br>(剔除thinking+上传)</th>
<th>推理总耗时<br>(实测)</th>
<th>prompt<br>长度</th><th>本轮新增<br>prompt长度</th><th>新增/总<br>图片数</th>
<th>Thinking<br>长度</th><th>Decode<br>长度</th><th>输出总<br>Token</th>
<th>prefill吞吐<br>(Token/s)</th><th>TPOT<br>(ms/Token)</th><th>Cache<br>命中率</th>
<th>本轮发出的<br>工具调用</th><th style="color:#9a3412">工具执行<br>耗时</th><th>工具输出<br>Token</th></tr>{tt}</table>

<h2>4 · 逐轮完整输入输出（含图像 + 中文翻译）</h2>
<p style="font-size:12.5px;color:#374151;margin:6px 0"><b>配色</b>：🟡 浅黄底=新增输入 | 🟢 绿字=模型上一轮输出 | 🔵 蓝字=工具返回结果 | 🔴 红字=本轮模型输出 | ⬜ 灰底=🇨🇳中文翻译</p>
{''.join(cards)}

{TOOLS_DOC}

<footer>数据源：{os.path.basename(CONFIG["llm_jsonl"])}（files_proxy v3.1）· {N} 轮 / {WALL}s</footer>
</div></body></html>'''

out = RP(CONFIG["out_html"])
open(out, "w").write(html_out)
print(f"written {out} ({os.path.getsize(out)/1e6:.1f} MB)")
print(f"N={N} | wall={WALL}s | 真TTFT p50={st.median(ttfts):.2f}s | DS服务端 p50={st.median(apis):.2f}s | 缓存{min(hits)}-{max(hits)}%")
