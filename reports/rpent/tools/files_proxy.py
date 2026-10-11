#!/usr/bin/env python3
"""Files API proxy: uploads images to DS Files API, replaces base64 with file_id refs.
Combined with complete_proxy features: full IO recording + phase timing + strip."""
import argparse, base64, hashlib, json, os, sys, threading, time, io
import urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_IMG_SEQ = [0]
_FILE_CACHE = {}  # {md5_hash: file_id}
_UPLOAD_COUNT = [0]


def save_request_images(messages, outdir="/data/rpent2/req_imgs"):
    files = []
    if not isinstance(messages, list): return files, messages
    os.makedirs(outdir, exist_ok=True)
    for m in messages:
        if not isinstance(m, dict): continue
        c = m.get("content")
        if not isinstance(c, list): continue
        for p in c:
            if not isinstance(p, dict): continue
            url = (p.get("image_url") or {}).get("url", "") if p.get("type") == "image_url" else ""
            if not (url and url.startswith("data:")): continue
            try:
                head, b64 = url.split(",", 1)
                _IMG_SEQ[0] += 1
                path = "%s/req%04d_img%d.png" % (outdir, _IMG_SEQ[0], len(files))
                with open(path, "wb") as f: f.write(base64.b64decode(b64))
                files.append(path)
                p["image_url"]["url"] = "file:" + path
            except Exception: pass
    return files, messages


def upload_image_to_files(img_data, api_key, upstream):
    """Upload image bytes to DS Files API, return file_id."""
    boundary = "----ProxyBoundary" + str(time.time_ns())
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="purpose"\r\n\r\n')
    body.write(b"user_data\r\n")
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="img.png"\r\n')
    body.write(b"Content-Type: image/png\r\n\r\n")
    body.write(img_data)
    body.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(upstream.rstrip("/") + "/files",
        data=body.getvalue(),
        headers={"Authorization": "Bearer " + api_key,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=60))
        fid = resp.get("id")
        if fid: _UPLOAD_COUNT[0] += 1
        return fid
    except Exception as e:
        print(f"[files] upload failed: {e}", file=sys.stderr)
        return None


def replace_images_with_file_ids(messages, api_key, upstream):
    """Replace base64 image_url blocks with file_id references.
    并行上传新图（实测串行1.12s→并行0.40s，2.8x加速）。
    Returns (modified_messages, n_replaced, n_new_uploads)."""
    if not isinstance(messages, list): return messages, 0, 0

    # 第一遍：收集需要上传的新图（hash 去重后）
    pending = []  # [(img_data, hash)]
    locations = []  # [(content_list, index, hash)]
    seen = set()
    for m in messages:
        if not isinstance(m, dict): continue
        c = m.get("content")
        if not isinstance(c, list): continue
        for j, p in enumerate(c):
            if not isinstance(p, dict) or p.get("type") != "image_url": continue
            url = (p.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"): continue
            try:
                _, b64 = url.split(",", 1)
                img_data = base64.b64decode(b64)
            except Exception: continue
            h = hashlib.md5(img_data).hexdigest()
            locations.append((c, j, h))
            if h not in _FILE_CACHE and h not in seen:
                seen.add(h)
                pending.append((img_data, h))

    # 并行上传所有新图
    if pending:
        def _up(item):
            data, h = item
            fid = upload_image_to_files(data, api_key, upstream)
            if fid: _FILE_CACHE[h] = fid
        threads = [threading.Thread(target=_up, args=(it,)) for it in pending]
        for t in threads: t.start()
        for t in threads: t.join()

    # 第二遍：替换为 file_id
    n_replaced = 0
    for c, j, h in locations:
        if h in _FILE_CACHE:
            c[j] = {"type": "file", "file_id": _FILE_CACHE[h]}
            n_replaced += 1
    return messages, n_replaced, len(pending)


def strip_history_images(messages):
    if not isinstance(messages, list): return 0
    last_img_msg = None
    for m in reversed(messages):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in ("image_url", "file") for p in c):
            last_img_msg = m; break
    n = 0
    for m in messages:
        if m is last_img_msg: continue
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list): continue
        for j in range(len(c)):
            if isinstance(c[j], dict) and c[j].get("type") in ("image_url", "file"):
                c[j] = {"type": "text", "text": "[img]"}
                n += 1
    return n



def restructure_observation(messages, max_images=3):
    """每轮重构：旧图全丢，只保留最新N张放在末尾的观测消息中。
    消除跨模态attention开销 + 保持前缀缓存稳定。"""
    if not isinstance(messages, list): return messages, 0
    
    # 1. 收集所有 file_id（按出现顺序）
    all_images = []
    for m in messages:
        if not isinstance(m, dict): continue
        c = m.get("content")
        if not isinstance(c, list): continue
        for j, p in enumerate(c):
            if isinstance(p, dict) and p.get("type") == "file":
                all_images.append(p["file_id"])
                # 替换为占位符（保持消息结构不变）
                c[j] = {"type": "text", "text": "[img]"}
    
    if not all_images:
        return messages, 0
    
    # 2. 只保留最新的 N 张
    keep = all_images[-max_images:]
    n_dropped = len(all_images) - len(keep)
    
    # 3. 追加观测消息到末尾
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": "Current visual observation:"}] +
                  [{"type": "file", "file_id": fid} for fid in keep]
    })
    
    return messages, n_dropped


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = None; api_key = None; log_path = None
    strip_history = False; use_files_api = True; restructure = False; reasoning_effort = None

    def log_message(self, *a): pass

    def _reply(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        t0 = time.perf_counter()
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)

        try: req_obj = json.loads(raw)
        except Exception: return self._reply(400, {"error": "bad json"})
        model = req_obj.get("model", "")
        stream = bool(req_obj.get("stream"))

        # Image extraction for logging (on deepcopy)
        import copy as _cp
        img_files = []
        try:
            msgs_copy = _cp.deepcopy(req_obj.get("messages"))
            img_files, msgs_safe = save_request_images(msgs_copy)
        except Exception:
            msgs_safe = req_obj.get("messages")

        # Files API: replace base64 with file_id (reduces body 17x)
        n_file_replaced = 0; n_new_uploads = 0
        stripped = 0
        if self.use_files_api:
            msgs, n_file_replaced, n_new_uploads = replace_images_with_file_ids(
                req_obj.get("messages"), self.api_key, self.upstream)
            req_obj["messages"] = msgs

        n_dropped = 0
        if self.restructure:
            msgs, n_dropped = restructure_observation(req_obj.get("messages"))
            req_obj["messages"] = msgs

        if self.strip_history:
            stripped = strip_history_images(req_obj.get("messages"))

        # 注入 reasoning_effort（压缩 thinking）
        if Handler.reasoning_effort:
            req_obj["reasoning_effort"] = Handler.reasoning_effort

        # 非流式请求 → 内部转流式（测真实TTFT：首chunk时间），
        # 收完chunks后组装成标准非流式响应返回给client。
        # 原因：非流式时无法分离 prefill 与 generation，ttft会被生成耗时污染。
        converted_stream = False
        if not stream:
            req_obj["stream"] = True
            req_obj["stream_options"] = {"include_usage": True}
            converted_stream = True

        fwd = json.dumps(req_obj).encode() if (n_file_replaced or stripped or converted_stream) else raw

        headers = {"Content-Type": "application/json",
                   "Authorization": "Bearer " + self.api_key,
                   "Accept": "text/event-stream"}
        url = self.upstream.rstrip("/") + self.path

        t_pre_send = time.perf_counter()
        logrec = {"ts": time.time(), "model": model, "stream": stream,
                  "proxy_prep_s": round(t_pre_send - t0, 3),
                  "messages": msgs_safe, "image_files": img_files,
                  "body_bytes": len(raw),
                  "fwd_body_bytes": len(fwd),
                  "stripped_images": stripped,
                  "file_replaced": n_file_replaced,
                  "file_new_uploads": n_new_uploads, "obs_dropped": n_dropped}

        outbound = urllib.request.Request(url, data=fwd, headers=headers)
        try:
            resp = urllib.request.urlopen(outbound, timeout=300)
        except urllib.error.HTTPError as e:
            err = e.read().decode(errors="replace")
            logrec.update({"status": e.code, "error": err[:4000],
                          "upload_s": round(time.perf_counter() - t_pre_send, 3)})
            self._flush(logrec)
            return self._reply(e.code, err)
        except Exception as e:
            logrec.update({"status": "transport_error", "error": str(e),
                          "upload_s": round(time.perf_counter() - t_pre_send, 3)})
            self._flush(logrec)
            return self._reply(502, {"error": str(e)})

        t_post_send = time.perf_counter()
        upload_s = round(t_post_send - t_pre_send, 3)

        if not stream:
            # 流式接收上游，测真实TTFT，然后组装非流式响应
            ttft = None; ttct = None; usage = None
            thinking = ""; content = ""; tc_acc = {}
            resp_id = ""; resp_created = 0; resp_model = model
            finish_reason = "stop"

            buf = b""
            while True:
                b = resp.read(1)
                if not b: break
                buf += b
                if buf.endswith(b"\n\n"):
                    for line in buf.decode(errors="replace").splitlines():
                        if not line.startswith("data: "): continue
                        d = line[6:]
                        if d.strip() == "[DONE]": continue
                        try: o = json.loads(d)
                        except Exception: continue
                        if o.get("id"): resp_id = o["id"]
                        if o.get("created"): resp_created = o["created"]
                        if o.get("model"): resp_model = o["model"]
                        if o.get("usage"): usage = o["usage"]
                        for c in o.get("choices", []):
                            if c.get("finish_reason"): finish_reason = c["finish_reason"]
                            delta = c.get("delta") or {}
                            now = time.perf_counter()
                            for tc in (delta.get("tool_calls") or []):
                                idx = tc.get("index", 0)
                                slot = tc_acc.setdefault(idx, {"id":"","name":"","arguments":""})
                                if tc.get("id"): slot["id"] += tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"): slot["name"] += fn["name"]
                                if fn.get("arguments"): slot["arguments"] += fn["arguments"]
                            if delta.get("reasoning_content"):
                                thinking += delta["reasoning_content"]
                                if ttft is None: ttft = now
                            if delta.get("content"):
                                content += delta["content"]
                                if ttft is None: ttft = now
                                if ttct is None: ttct = now
                            if delta.get("tool_calls"):
                                if ttft is None: ttft = now
                                if ttct is None: ttct = now
                    buf = b""

            t_end = time.perf_counter()
            t_first = ttft or t_end

            # 组装标准非流式 chat.completion 响应
            msg = {"role": "assistant", "content": content if content else None}
            if thinking: msg["reasoning_content"] = thinking
            if tc_acc:
                msg["tool_calls"] = [
                    {"id": tc_acc[k]["id"] or f"call_{k}",
                     "type": "function",
                     "function": {"name": tc_acc[k]["name"],
                                  "arguments": tc_acc[k]["arguments"]}}
                    for k in sorted(tc_acc.keys())]
            assembled = {
                "id": resp_id or "chatcmpl-proxy",
                "object": "chat.completion",
                "created": resp_created or int(time.time()),
                "model": resp_model,
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": finish_reason}],
                "usage": usage or {},
            }
            self._reply(resp.status, assembled)

            u = usage or {}
            rd = u.get("completion_tokens_details") or {}
            pd = u.get("prompt_tokens_details") or {}
            logrec.update({
                "status": resp.status, "upload_s": upload_s,
                "ttft_s": round(t_first - t0, 3),
                "api_wait_s": round(t_first - t_post_send, 3),
                "first_content_s": round(ttct - t0, 3) if ttct else None,
                "duration_s": round(t_end - t0, 3),
                "gen_s": round(t_end - t_first, 3),
                "converted_stream": True,
                "thinking": thinking[:8000],
                "content": content[:2000],
                "tool_calls": [{"name": v["name"], "arguments": v["arguments"][:1500]}
                               for k, v in sorted(tc_acc.items())],
                "usage": usage,
                "thinking_tok": rd.get("reasoning_tokens"),
                "cached_tok": pd.get("cached_tokens"),
            })
            self._flush(logrec)
            return

        # Streaming path
        self.send_response(resp.status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        ttft = None; ttct = None; usage = None
        thinking = ""; content = ""; tc_acc = {}
        client_ok = True

        def send(data):
            nonlocal client_ok
            if not client_ok: return
            try:
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                client_ok = False

        buf = b""
        while True:
            b = resp.read(1)
            if not b: break
            buf += b
            if buf.endswith(b"\n\n"):
                send(buf)
                for line in buf.decode(errors="replace").splitlines():
                    if not line.startswith("data: "): continue
                    d = line[6:]
                    if d.strip() == "[DONE]": continue
                    try: o = json.loads(d)
                    except Exception: continue
                    if o.get("usage"): usage = o["usage"]
                    for c in o.get("choices", []):
                        delta = c.get("delta") or {}
                        for tc in (delta.get("tool_calls") or []):
                            idx = tc.get("index", 0)
                            slot = tc_acc.setdefault(idx, {"name":"","arguments":""})
                            fn = tc.get("function") or {}
                            if fn.get("name"): slot["name"] += fn["name"]
                            if fn.get("arguments"): slot["arguments"] += fn["arguments"]
                        now = time.perf_counter()
                        if delta.get("reasoning_content"):
                            thinking += delta["reasoning_content"]
                            if ttft is None: ttft = now
                        if delta.get("content"):
                            content += delta["content"]
                            if ttft is None: ttft = now
                            if ttct is None: ttct = now
                        if (delta.get("tool_calls") and ttct is None):
                            ttct = now
                        if delta.get("tool_calls") and ttft is None:
                            ttft = now
                buf = b""

        if client_ok:
            try: self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError): client_ok = False

        t_end = time.perf_counter()
        u = usage or {}
        rd = u.get("completion_tokens_details") or {}
        pd = u.get("prompt_tokens_details") or {}
        tcs_list = [tc_acc[k] for k in sorted(tc_acc.keys())] if tc_acc else []

        logrec.update({
            "status": resp.status, "upload_s": upload_s,
            "api_wait_s": round((ttft or t_end) - t_post_send, 3) if ttft else None,
            "ttft_s": round((ttft or t_end) - t0, 3),
            "first_content_s": round(ttct - t0, 3) if ttct else None,
            "duration_s": round(t_end - t0, 3),
            "thinking": thinking[:8000],
            "content": content[:2000],
            "tool_calls": tcs_list,
            "usage": usage,
            "thinking_tok": rd.get("reasoning_tokens"),
            "cached_tok": pd.get("cached_tokens"),
        })
        self._flush(logrec)

    def _flush(self, rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, {"ok": True, "upstream": self.upstream,
                           "files_cached": len(_FILE_CACHE),
                           "uploads": _UPLOAD_COUNT[0]})
        else:
            self._reply(404, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--strip-history-images", action="store_true")
    ap.add_argument("--no-files-api", action="store_true")
    ap.add_argument("--restructure-obs", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low","medium","high","minimal"])
    a = ap.parse_args()
    Handler.upstream = a.upstream.rstrip("/")
    Handler.api_key = (os.environ.get("DEEPSEEK_API_KEY")
                       or os.environ.get("HUOSHAN_API_KEY")
                       or os.environ.get("ARK_API_KEY") or "")
    Handler.log_path = a.log
    Handler.strip_history = a.strip_history_images
    Handler.use_files_api = not a.no_files_api; Handler.restructure = a.restructure_obs; Handler.reasoning_effort = a.reasoning_effort
    if not Handler.api_key: sys.exit("need API key")
    print(f"files-api proxy on 127.0.0.1:{a.port} -> {Handler.upstream} "
          f"log={a.log} strip={Handler.strip_history} files_api={Handler.use_files_api}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
