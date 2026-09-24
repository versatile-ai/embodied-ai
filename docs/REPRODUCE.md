# 复现指南

## 1. 先复核归档

在本仓库根目录执行 `python3 scripts/verify_archive.py`。再运行 `python3 -m http.server 8778`，通过 `http://127.0.0.1:8778/` 打开根 README 的三份 HTML 报告；直接双击 `file://` 页面时浏览器可能拦截动态 JSON 读取。校验脚本检查各报告引用的本地资源、归档哈希和五个 GPT/混合结果文件；只需 Python 标准库。优先核对录像、`result.json`、逐轮 `history.json` 和 π 四轮摘要。`result.success` 是 native evaluator 结果；视觉判断在报告中另列。

## 2. 环境和外部资料

实验采用三台角色：Mac 运行 GPT-6 Astra 控制器；NVIDIA GPU 服务器运行官方 RoboDojo / Isaac Sim 仿真；昇腾 NPU 服务器运行 π0.5 服务。Mac 通过 SSH 本地转发连接 GPU 的 `19113` 和 NPU 的 `8642`。真实地址与登录凭据不随仓库分发。

- GPU：官方 RoboDojo checkout 及 Isaac Sim 5.1、CuRobo、机器人场景资源。原运行时记录的 RoboDojo commit 为 `ee67a1468510da7624a089164402359f2afc72c8`。仓库地址与依赖安装请以团队有权限的上游镜像核对；此处 commit 是本实验溯源点，不保证公网仓库仍提供它。
- OpenPI 源码记录的 commit 为 `432f82b1758c5b1202e42a3dfe014546dbc50871`；NPU 镜像内部依赖应与此兼容。
- NPU 镜像：`/data/pi05_hybrid/pi05-hybrid-v1.tar.gz`，约 8.9 GB，含代码与依赖，不含权重。
- NPU 权重：`/data/pi05_hybrid/weights/robodojo_pi05_pt/`，约 6.8 GB。服务还读取 `/data/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets` 中的归一化统计。42 GB 原始 JAX checkpoint 本身仅在重新转换时需要，但上述 `assets` 必须存在。团队应核对模型和统计文件版本，镜像服务的 `/health` 目前不证明权重哈希。
- 布局定义通常在 `/data/pi05_hybrid/layouts/`。本档案只比较 Layout 0 / seed 0，不包括完整官方评测集。

NPU 用户需要 Docker 权限与 `/data` 共享盘读权限；GPU 用户需要可运行官方 Isaac Sim 的 NVIDIA 环境。Mac 需 `python3`、本仓库 `requirements-tools.txt` 的轻量依赖，以及可访问 GPT-6 Astra / xhigh 的已登录 Codex CLI。不要将个人 token、SSH 密码或服务器地址写入仓库。

## 3. 启动 NPU π0.5 服务

以下命令在 NPU 主机的本仓库根目录运行：

```bash
docker load < /data/pi05_hybrid/pi05-hybrid-v1.tar.gz
docker run -d --name pi05-service --privileged --network host \
  -e ASCEND_VISIBLE_DEVICES=0 -v /data:/data \
  -v "$PWD/npu/pi05_service.py:/data/pi05_hybrid/scripts/pi05_service.py:ro" \
  pi05-hybrid:v1 bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && python3 /data/pi05_hybrid/scripts/pi05_service.py npu 8642'
curl -fsS http://127.0.0.1:8642/health
```

如果宿主机驱动没有通过镜像默认 Ascend runtime 暴露，按本机安装方式额外挂载驱动目录。`/health` 预期返回 `{"ok":true,"device":"npu"}`，随后做一次真实推理才算模型可用。服务期望相机键 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`；Mac 端必须选 `--student-camera-schema aloha`。

## 4. 启动官方 GPU 仿真

在 GPU 主机同时准备官方 RoboDojo checkout 和本仓库 checkout，安装 RoboDojo / Isaac Sim 上游说明中的环境。在本仓库根目录设置路径后运行：

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export ROBODOJO_ADAPTER_ROOT="$PWD"
export ROBODOJO_PYTHON=/path/to/robodojo-env/bin/python
hybrid_rollout/robodojo/run_gpu_sim_server.sh /path/to/new-sim-run put_bottles_into_dustbin 19113 0
```

`ROBODOJO_PYTHON` 必须指向有官方环境、Isaac Sim、CuRobo 的可执行 Python；实际部署也可为 Anaconda 环境中的 `bin/python`。启动器锁定单环境、GPU 0、布局 0、三次 RTX 重置预热与四视角录像；运行目录应出现 `resolved_config.json`、`adapter_runtime.json`。一个服务只接受一个新的 episode；每次正式测试重新启动服务、重置机器人和瓶子，不接续旧 episode。

Mac 端开两个 SSH 隧道（替换成团队自己的别名）：

```bash
ssh -N -L 19113:127.0.0.1:19113 gpu-host-alias
ssh -N -L 8642:127.0.0.1:8642 npu-host-alias
```

这两个命令分别保持在终端中；Mac 上的控制器连接 `127.0.0.1`。监控图片由 GPU 服务器写入 `ROBODOJO_LIVE_DIR`，不等于 Mac 本地 8766 页面；如需本地持续监看，须另外部署并转发监控 HTTP 服务。即使浏览器监控中断，GPU 服务器也会在仿真运行目录写 `sensors.mp4`。

## 5. 新跑一次同任务

Mac 在本仓库根目录。归档控制器默认的 `galbot` 配置有意使用无效示例网关；用团队自己的 GPT-6 Astra 服务时，必须配置真实网关与密钥。若通过已授权的 Codex/ChatGPT 登录，可用隔离的 `codex_a` profile（每个新实验沿用同一授权账户，但创建全新模型会话）：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-tools.txt
export ROLLOUT_AUTH_PROFILE=codex_a
export ROLLOUT_SHARED_ROOT="$PWD/.rollout-private"
python3 -m hybrid_rollout.robodojo.codex_backend.profiles login codex_a --codex "$(command -v codex)"
export CODEX_HOME="$ROLLOUT_SHARED_ROOT/private/auth_profiles/codex_a/codex_home"
python3 -m hybrid_rollout.robodojo.skill.run \
  --output runs/gpt-layout0-new --task put_bottles_into_dustbin \
  --evaluation-method gpt_only --codex "$(command -v codex)" \
  --sim-port 19113 --seed 0 --max-decisions 180
```

如果该 profile 已登录，跳过 `login`，直接设置同一个 `CODEX_HOME`。先执行 `codex login status` 检查身份；不要将 `.rollout-private` 提交到 Git。企业 LiteLLM 路径需自行替换示例网关并按服务策略设置 `OPENAI_API_KEY`，示例配置不能直接推理。

混合实验先重新启动仿真服务，再运行：

```bash
python3 -m hybrid_rollout.robodojo.skill.run \
  --output runs/hybrid-layout0-new --task put_bottles_into_dustbin \
  --evaluation-method pi05_plus_gpt --codex "$(command -v codex)" \
  --sim-port 19113 --seed 0 --max-decisions 180 \
  --student-backend npu-http --student-url http://127.0.0.1:8642 \
  --student-camera-schema aloha --allow-unverified-student
```

π 单独实验也需新的仿真服务，再运行：

```bash
python3 scripts/run_pi_only.py --output runs/pi-layout0-new \
  --sim-port 19113 --student-url http://127.0.0.1:8642 \
  --seed 0 --max-steps 700
```

控制器每次用新的输出目录与新模型会话。当前 GPT 直接控制代码采用自适应步长上限 50；历史固定 5 步、固定 50 步实验用的是当时的控制逻辑。要精确重跑历史配置，需对应当时源码和模型服务状态；本分支保证可检查历史数据，并可跑相同任务的新实验，**不保证逐位复现历史轨迹**。模型采样、RTX 渲染、网络和服务器负载也会改变耗时与动作。

## 6. 验收与限制

检查 `reset.json`、`native_evaluation.json`、`initial_observation_fingerprint.json`、`evaluation_outcome.json`、`summary.json`、完整 `sensors.mp4` 和控制器 `result.json` / `history.json`。成功判定以 native evaluator 为官方口径，视频人工复核另记；控制步不足 700 且未触发 native success 的中断运行标为未完成。π 服务返回的 `ms` 是服务端推理耗时；Mac 的 HTTP 往返包含编码、传输和排队，仿真 chunk 执行又是另一段时间。历史报告只包含这些已有埋点，不能反推服务端真实 GPT TTFT、TPOT 或每段网络时延。
