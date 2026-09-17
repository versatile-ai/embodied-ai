# RoboDojo 混合架构评测操作手册

> 本文档供 AI 代理（或人类操作员）独立执行完整的 RoboDojo 混合架构评测。
> 全部命令在 Mac 终端执行，不需要 SSH 到远程服务器（服务已在运行）。
> 大脑模型通过 volcengine 网关调用 Qwen3.8-Max。

## 0 前置条件

| 项 | 检查命令 | 预期输出 |
|---|---|---|
| 仿真服务 | curl -s http://127.0.0.1:8663/health | {"ok": true} |
| π0.5 推理服务 | curl -s http://127.0.0.1:8642/health | {"ok": true, "device": "npu"} |
| 大脑网关 | source ~/.qwen_env 后 curl chat/completions | 包含 "content" |
| Live 页面 | curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8665/ | 200 |

## 1 测试要求（三条铁律）

1. 每次测试前恢复环境到初始化状态——每次 POST /session 创建新 Session，simsvc 自动 mj_resetData + 按 layout JSON 精确放置所有物体（自检门 A1/A2 验证 ≤2mm）。
2. 记录大脑模型每次输入和输出——llm_brain.py 自动写入 BRAIN_IO_DIR（默认 /tmp/brain_io/），每个决策点一个 JSON 文件，包含：指令、历史、图像元数据、候选 EE 轨迹、原始响应、解析决策、延迟。
3. 录制完整机械臂视频——orchestrator 在每个决策边界保存 base/head + wrist 帧（PPM），episode 结束后自动 ffmpeg 合成 MP4 到 /tmp/ep_video_*/。

## 2 可用测试任务

| 任务 | 布局文件 | 类型 | step_lim | 评分 |
|---|---|---|---|---|
| put_bottles_into_dustbin | layouts/put_bottles_into_dustbin_{0..4}.json | 接触类 | 700 | 1瓶=10, 2瓶=25, 3瓶=40, 4瓶=100 |
| classify_objects | layouts/classify_objects_{0..4}.json | 语义类 | 1100 | 1篮=15, 2篮=40, 3篮=100 |

## 3 三臂定义

| 臂 | 命令 | 大脑 | 说明 |
|---|---|---|---|
| π0.5-only | python3 harness/run_pi05_only.py <layout> <task> | 无 | VLA 独立执行，纯基线 |
| Hybrid | python3 harness/run_hybrid.py <layout> <task> | π0.5 提案 + LLM 审批 | 原文核心架构 |
| Direct | python3 harness/run_direct.py <layout> <task> | 仅 LLM（无 π0.5） | 纯大脑遥控 |

## 4 执行评测

### 4.1 单跑
```bash
cd /Users/max/code/ai/pi05_hybrid
source ~/.qwen_env
python3 harness/run_hybrid.py layouts/put_bottles_into_dustbin_0.json put_bottles
```

### 4.2 全矩阵
```bash
for lay in layouts/put_bottles_into_dustbin_0.json layouts/classify_objects_0.json; do
  for arm in pi05_only hybrid direct; do
    task=$(basename $lay | sed 's/_0\.json//' | sed 's/into_dustbin//')
    echo "=== $arm × $task"
    case $arm in
      pi05_only) python3 harness/run_pi05_only.py "$lay" "$task" ;;
      hybrid)    python3 harness/run_hybrid.py    "$lay" "$task" ;;
      direct)    python3 harness/run_direct.py    "$lay" "$task" ;;
    esac
  done
done
```

## 5 结果收集
- FINAL 行：score/success/steps
- 大脑 I/O：/tmp/brain_io/*.json
- 视频：/tmp/ep_video_*/cam_base.mp4
- 干预率 = correct_steps / (follow_steps + correct_steps)

## 6 故障排查

| 症状 | 修复 |
|---|---|
| π0.5 health 拒连 | pkill tunnel → 重启 tunnel_8642.sh |
| simsvc session 崩 | tail /tmp/simsvc_mac.log |
| brain 429 | 等 60s 重试 |
| brain 返回空 | 确认 max_tokens ≥ 2048 |
| 直播不动 | episode 结束，等下一局 |
| MuJoCo NaN | 检查物体 spawn 穿透 |
