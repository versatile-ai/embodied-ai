#!/usr/bin/env python3
"""逐轮轨迹翻译（thinking + 正文 + 输入上下文）→ 缓存 JSON。
用法: python3 translate_rounds.py <llm.jsonl> <out_translations.json>
依赖: DEEPSEEK_API_KEY 环境变量。
幂等：已翻译的轮次跳过（断点续翻）。
注意：报告生成前必须确认缓存已含全部 {i}_system/{i}_user/{i}_input 键，
     否则轮1翻译会空（生成与翻译并发时曾踩坑）。
"""
import json, os, sys, time, urllib.request

llm_path = sys.argv[1] if len(sys.argv) > 1 else "llm.jsonl"
cache = sys.argv[2] if len(sys.argv) > 2 else "translations.json"

KEY = os.environ.get("DEEPSEEK_API_KEY")
BASE = "https://api.deepseek.com/v1/chat/completions"
if not KEY:
    sys.exit("need DEEPSEEK_API_KEY")

recs = [json.loads(l) for l in open(llm_path)]

def msg_text(m):
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    return ""

def translate(text, kind):
    if not text or len(text.strip()) < 5: return ""
    body = {"model": "deepseek-flash", "max_completion_tokens": 2000,
            "messages": [
                {"role": "system",
                 "content": "你是技术翻译。把下面的具身智能Agent的" + kind +
                            "翻译成简洁的中文。保留工具名、坐标、文件名、数值、JSON结构不译。直接输出译文，不要解释。"},
                {"role": "user", "content": text[:3200]}]}
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": "application/json"})
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=120))
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[翻译失败: {e}]"

translations = {}
if os.path.exists(cache):
    translations = json.load(open(cache))

for i, r in enumerate(recs):
    if str(i) in translations:
        continue
    thinking = (r.get("thinking") or "")[:3200]
    content = (r.get("content") or "")[:1500]
    translations[str(i)] = {
        "thinking_zh": translate(thinking, "推理过程(thinking)") if thinking else "",
        "content_zh": translate(content, "输出正文") if content else "",
    }
    msgs = r.get("messages") or []
    prev_len = len(recs[i-1].get("messages") or []) if i > 0 else 0
    new_msgs = msgs[prev_len:] if prev_len < len(msgs) else []
    if i == 0:
        sys_text = ""; user_text = ""
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system":
                sys_text = msg_text(m)[:2500]
            elif isinstance(m, dict) and m.get("role") == "user":
                user_text = msg_text(m)[:1000]
        translations[f"{i}_system"] = translate(sys_text, "系统提示词")
        translations[f"{i}_user"] = translate(user_text, "用户任务指令")
    else:
        texts = []
        for m in new_msgs:
            if isinstance(m, dict) and m.get("role") == "assistant":
                t = msg_text(m)
                if t.strip(): texts.append(t[:2000])
        for m in new_msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                t = msg_text(m)
                if t.strip() and len(t.strip()) > 10: texts.append(t[:800])
        combined = "\n---\n".join(texts)
        translations[f"{i}_input"] = translate(combined, "输入上下文（上一轮输出+观测）") if combined else ""
    print(f"轮{i+1}/{len(recs)}", end=" ", flush=True)
    json.dump(translations, open(cache, "w"), ensure_ascii=False)
    time.sleep(0.25)

print(f"\n翻译完成: {len(translations)} 条 → {cache}")
