# Astra 双臂仿真环境搭建与集成文档

本文档描述当前可交付给同事复现的独立仿真环境：组件、目录、安装方式、服务协议、GPT/π 推理接入、验收标准和已知边界。

## 1. 环境目标

这套环境用于在 Mac 上复现双 X5 机械臂的视觉控制评测，当前优先支持 `put_bottles_into_dustbin`，同时保留 `classify_objects` 场景。仿真服务与上游 `pi05_hybrid` 代码和服务隔离，使用本目录中的场景、资产、控制器和会话录像。

当前实现是 **MuJoCo 移植环境**，不是 RoboDojo/Isaac 的原生运行时。它适合做控制协议、视觉输入、动作执行、环境回馈和策略调试；不能直接把这里的分数当成官方 benchmark 分数。

## 2. 系统组成

部署关系和数据流如下：

```text
┌──────────────────────── Mac ────────────────────────┐
│ Astra 控制端 / client.py                            │
│       │ observe                                      │
│       ▼                                              │
│ simsvc.py :8763 ◄────动作──── Astra                  │
│   MuJoCo / 相机 / IK / 评分 / 录像                   │
└───────────────┬─────────────────────────────────────┘
                │ images + state + instruction
                │ HTTP :8642（SSH tunnel 或内网）
                ▼
┌──────────────────── NPU 主机 ───────────────────────┐
│ pi05-hybrid 容器                                     │
│ π0.5 / openpi / CANN runtime / model.safetensors     │
│                         POST /infer :8642            │
└──────────────────────────────────────────────────────┘
```

用户实际操作的是 Mac 上的 Astra。Mac 负责仿真、观测、动作执行、评分和页面；NPU 主机只负责 π0.5 推理。权重和 NPU runtime 不需要安装到 Mac，也不需要在 NPU 主机运行 MuJoCo。

| 组件 | 位置 | 作用 |
|---|---|---|
| MuJoCo 场景服务 | `harness/simsvc.py` | 创建会话、加载场景、执行物理、渲染、评分、记录 |
| 6D 阻尼 IK | `harness/control.py` | 根据末端位置和姿态目标计算双臂关节目标 |
| 双 X5 场景 | `assets/x5/dual_x5_scene.xml` | 机器人、执行器、相机、桌面和基础材质 |
| OBJ 与碰撞资产 | `assets/meshes/` | 瓶子、篮筐、垃圾桶等视觉网格和 bbox |
| 任务布局 | `layouts/*.json` | 物体初始位姿、缩放、类别和任务配置 |
| HTTP 客户端 | `client.py` | 观测保存、动作提交、推理服务适配、录像流程 |
| 直播页面 | `harness/live.html` | 显示主相机、左右腕相机和当前会话状态 |
| 视频编码器 | `encode_video.swift` | 使用 macOS AVFoundation 将 PNG 帧编码为 MP4 |
| 启动器 | `start_server.py` | 后台启动服务、检查端口和记录 PID/日志 |
| 回归与集成测试 | `test_service.py`、`check_render.py`、`verify_http.py` | 环境、渲染、协议和录像验收 |

推理服务不包含在仿真包内。`client.py` 可以调用兼容的 `/infer` HTTP 服务；GPT 或 π 的 checkpoint、NPU 服务和 token 需要由测试方单独配置。π0.5 通过 `pi05-hybrid` 容器镜像在 NPU 主机启动，Mac 仿真仓库只通过 HTTP 调用它。

## 3.1 权重与推理服务路径

当前仓库没有提交模型权重。π0.5 的交付包已在 NPU 共享盘准备完成，权重不放进 Docker 镜像，而是在运行时从 `/data` 挂载加载：

```text
/data/pi05_hybrid/pi05-hybrid-v1.tar.gz                         # Docker 镜像，约 8.9G，代码+依赖
/data/pi05_hybrid/weights/robodojo_pi05_pt/model.safetensors
/data/pi05_hybrid/weights/robodojo_pi05_pt/                      # 权重目录，约 6.8G
/data/pi05_hybrid/robodojo_ckpt/                                 # 原始 JAX checkpoint，可选，约 42G
/data/pi05_hybrid/layouts/                                       # 评测布局 JSON，<1M
```

对应的 openpi 源码和归一化资产为：

```text
/data/pi05_hybrid/openpi/src
/data/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets
asset_id = arx_x5_sim
```

镜像内服务入口是 `/data/pi05_hybrid/scripts/pi05_service.py`，默认监听 `:8642`，通过 `ASTRA_PI05` 接入本机客户端。Mac 与 NPU 主机之间可以使用上游 `harness/tunnel_8642.sh` 的 SSH 隧道。权重路径、部署方式和私有配置详见本 README 的“权重与推理服务路径”章节。

GPT 大脑适配器是上游 `harness/llm_brain.py`，使用 OpenAI-compatible `/chat/completions`，通过 `QWEN_BASE_URL`、`QWEN_API_KEY`、`QWEN_MODEL` 配置。API key 只能放在本机 secret 文件或密码管理器，不能提交 GitHub。

## 3.2 π0.5 NPU 容器部署

当前已交付固定镜像文件 `/data/pi05_hybrid/pi05-hybrid-v1.tar.gz`。镜像包含代码、Python 依赖、openpi 补丁和 NPU/CANN 运行环境，但**不包含 6.8G 模型权重**；同事必须同时具备 Docker 权限和 `/data` 共享盘读权限。如果服务需要在共享盘写缓存，还需对应写权限。无论通过 tar 包还是仓库分发，都应记录镜像 tag 和 digest：

```text
<registry>/<namespace>/pi05-hybrid:<tag>@sha256:<digest>
```

如果权重不在镜像中，容器内应保持以下路径：

```text
/data/pi05_hybrid/weights/robodojo_pi05_pt/model.safetensors
/data/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets
```

容器启动参数中的 NPU 设备映射取决于 NPU 主机安装的 Docker/CANN runtime；不要直接照抄其他机器的 `--device` 列表。当前交付包使用下面这组已验证的启动参数：

```sh
docker load < /data/pi05_hybrid/pi05-hybrid-v1.tar.gz

docker run -d --name pi05-service \
  --network host \
  --privileged \
  -e ASCEND_VISIBLE_DEVICES=0 \
  -v /data:/data \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  pi05-hybrid:v1 \
  bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
           python3 /data/pi05_hybrid/scripts/pi05_service.py npu 8642"
```

如果 NPU 主机的 CANN/驱动路径不同，需要按该主机实际路径调整 driver 挂载和 `set_env.sh`；不要删除 `/data` 挂载。服务必须在容器内监听 `0.0.0.0:8642`，并提供：

```sh
curl http://127.0.0.1:8642/health
# {"ok": true, "device": "npu"}
```

Mac 仿真端接入方式：

```sh
# 同一台机器或已做端口转发
export ASTRA_PI05=http://127.0.0.1:8642

# NPU 在远程主机时，可在 Mac 建 SSH 隧道
ssh -N -L 8642:127.0.0.1:8642 <npu-user>@<npu-host>
export ASTRA_PI05=http://127.0.0.1:8642
```

先验证 `/health`，再执行 `client.py infer`。第一次推理还应检查 `/infer` 返回 `(50,14)` 动作、`ms` 延迟和输入相机键映射。容器日志和镜像 digest 应随每次策略评测一并记录。

## 3.3 同事实际操作顺序

### NPU 主机

```sh
docker load < /data/pi05_hybrid/pi05-hybrid-v1.tar.gz
docker run -d --name pi05-service --privileged --network host \
  -e ASCEND_VISIBLE_DEVICES=0 -v /data:/data \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  pi05-hybrid:v1 bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  python3 /data/pi05_hybrid/scripts/pi05_service.py npu 8642"
curl http://127.0.0.1:8642/health
```

### Mac

```sh
git clone <astra-repo-url>
cd astra_eval
./setup_colleague.sh
python3 start_server.py
ssh -N -L 8642:127.0.0.1:8642 <npu-user>@<npu-host>
```

另开一个 Mac 终端：

```sh
cd astra_eval
source .venv/bin/activate
export ASTRA_PI05=http://127.0.0.1:8642
python3 client.py start --task put_bottles --layout 0
python3 client.py observe --session <session_id>
python3 client.py infer --session <session_id>
python3 client.py follow --session <session_id> --steps 1 --reason 'Astra reviewed action'
open http://127.0.0.1:8763/live
```

已知镜像环境版本：Python 3.11、PyTorch 2.7.1、torch_npu 2.7.1.post2、CANN 8.5.2、transformers 4.53.2、MuJoCo 3.11.0；首次加载权重约 4 分钟，单个 50 步 action chunk 延迟约 570ms。输入图像要求 HWC `uint8`，服务端会转 CHW 并 resize 到 224。

如果 NPU 与 Mac 在同一局域网且端口已安全开放，可以把 `ASTRA_PI05` 设置为 `http://<npu-host>:8642`，不必建立 SSH 隧道。推荐优先使用隧道，避免把推理服务直接暴露到公网。

## 3.4 官方场景对比审计（2026-09-17）

本次核对以仓库内的 `docs/robodojo_task_port_spec.md`、`docs/alignment_audit.md`、官方布局 JSON 结构以及当前 `harness/simsvc.py`/`assets/x5/dual_x5_scene.xml` 为准。结论是：**当前两个场景可用于 Astra/π0.5 的接口和控制回归，但还不是官方 RoboDojo 的等价仿真环境**。因此目前的分数只能作为本地回归分数，不能与官方 benchmark 分数直接横向比较。

控制接口保持 25 Hz；当前 MuJoCo 为稳定双指接触改用 1000 Hz 物理步（每个动作 40 步），不再声称物理步长与官方相同。其余移植部分包括：桌面、地面摩擦和弹性；双 X5 根位姿、6+1 关节/夹爪结构；头部相机外参和头部/腕部相机视场角；布局中的固定物体位姿、类别和质量；两个任务的步数上限、主要成功条件和录像/HTTP 协议。`put_bottles_into_dustbin` 的 4 瓶布局和 `classify_objects` 的 3 类物体+3 个篮筐也能从当前 JSON 复现。

| 对比项 | 当前实现 | 与官方的影响 | 优先级 |
|---|---|---|---|
| 任务覆盖 | 仅 `put_bottles`、`classify_objects` | 官方规格列出 10 个任务，缺少 build_tower、make_kong、fold_clothes、classify_objects_by_language、arrange_largest_number、organize_table、imitate_sorting_sequence、pack_objects_into_box | P0 |
| 桌面布局 | 两个当前任务的布局均为 `[1.4,1.1,0.05]` | 已按官方公共规格统一；桶相对桌边界保留布局 JSON 的固定位姿 | 已修复 |
| 物理引擎 | MuJoCo 3.3.7 | 官方是 Isaac Sim/PhysX/RTX；接触、摩擦、堆叠稳定性和渲染结果不等价 | P0 |
| 资产碰撞 | 视觉网格 + bbox/盒体碰撞壳 | 瓶子、手办、分类物体的真实轮廓和接触点被简化，抓取可行性与官方不同 | P0 |
| 机器人自碰撞 | 为解决相邻链节初始自穿透，当前排除了相邻链节碰撞；官方配置为 `robot_self_collision: true` | 双臂/手臂互碰和障碍规避结果可能偏乐观 | P1 |
| 抓取模型 | 默认开启 grasp assist，并对夹爪/物体使用简化接触 | 便于稳定回归，但不代表官方纯物理抓取；会改变失败率和动作分数 | P1 |
| 夹爪行程 | XML 与归一化均为 `joint7 ∈ [-0.01, 0.044]` | 已按官方 affine scale 修正 | 已修复 |
| 相机外参 | 头部相机已改为 `[0,-0.41,1.308]`、Rx30°；腕相机仍为 EE 相对近似挂载 | 头部构图已对齐；腕相机安装链仍需用官方 URDF 做像素级标定 | P1 |
| 房间/光照 | 五个平面+两盏 MuJoCo 灯 | 缺少 `Simple_Room_nolight` 房间网格、HDR 背景和官方材质；颜色/阴影/反射不同 | P1 |
| 装饰与元数据 | `camera_stand` 已加入可见的非碰撞代理几何；布局仍只取位姿/质量等字段 | 支架外观已有占位，官方 mesh/材质和完整元数据仍缺失 | P2 |
| 功能点/支撑圆 | 目前主要用 bbox、XY 范围和篮筐包围盒判定 | 官方 `functional`、`support`、checkpoint 等语义未完整移植；堆塔、装箱、折叠等任务无法等价实现 | P0 |
| 柔性/铰接/支援臂 | 未实现 PBD 布料、抽屉铰接体、Franka 支援臂与演示轨迹 | `fold_clothes`、`organize_table`、`imitate_sorting_sequence` 等任务无法复现 | P0 |
| 多环境 | 当前单环境逐局运行 | 官方布局有 `num_envs=10`；不影响单局正确性，但缺少并行吞吐与批量评测 | P2 |

`classify_objects` 的分类分组问题已修复：评分现在按布局字段 `category` 聚合，同类的不同 `category_idx` 变体会进入同一个篮筐；`category_idx` 仍只用于选择外观/资产变体。修复位置为 `harness/simsvc.py::_build_classify`。

建议的修复顺序是：先把头部/腕部相机外参和真实 mesh/bbox 碰撞校准，再关闭默认 grasp assist 做一组纯物理基线；随后移植功能点/support metadata 和官方评分状态机。完成这些后，再决定是否投入 PhysX/Isaac 原生环境和剩余 8 个任务。当前最小可交付边界应明确写成“两个刚体 pick-place 场景的 MuJoCo 复现”。

可复核依据：`docs/alignment_audit.md` 记录了已对齐与近似项，`docs/robodojo_task_port_spec.md` 记录了官方 10 任务、相机链、功能点/support metadata 和物理参数；运行时差异可直接在 `harness/simsvc.py` 与 `assets/x5/dual_x5_scene.xml` 中检查。

## 3. 已验证的运行基线

当前在 macOS 上验证：

- Python 3.9.6
- MuJoCo 3.3.7
- NumPy 2.0.2
- Pillow 11.3.0
- Xcode Command Line Tools（提供 `swiftc` 和 AVFoundation 编码依赖）
- MuJoCo 物理频率 1000 Hz，控制频率 25 Hz
- 本地仿真端口 `127.0.0.1:8763`

离屏渲染依赖 macOS 图形上下文。若在无图形会话的 SSH、沙箱或 CI 进程中直接运行渲染测试，可能出现 `invalid CoreGraphics connection`；应在有图形上下文的 macOS 用户会话中运行，或为 CI 单独配置 MuJoCo EGL/OSMesa 后端。

## 4. 目录结构

```text
astra_eval/
├── assets/
│   ├── x5/                 # 双 X5 XML、URDF、相机和材质
│   └── meshes/             # OBJ + bbox.json + axis_fix.json
├── layouts/                # 任务布局 JSON
├── harness/
│   ├── simsvc.py           # 仿真 HTTP 服务
│   ├── control.py          # 6D 阻尼 IK
│   └── live.html           # 直播页面
├── client.py               # 统一客户端与推理适配
├── start_server.py         # 后台启动器
├── test_service.py         # 10 项回归测试
├── check_render.py         # 场景/相机检查
├── verify_http.py          # HTTP 集成检查
├── encode_video.swift      # macOS 视频编码源代码
├── setup_colleague.sh      # 创建 venv、装依赖、编译编码器
├── validate_colleague.sh   # 一键验收
├── requirements.txt        # 固定 Python 依赖
├── docs/                   # 上游评测协议、任务和对齐审查文档
└── runs/                   # 运行时生成；不需要随源码交付
```

## 5. 从零安装

### 5.1 获取源码包

解压源码到任意目录即可，不需要原始 `pi05_hybrid` 路径。源码包含完整场景和任务资产，不包含历史 `runs/` 和机器相关的已编译二进制。

### 5.2 准备系统依赖

在 macOS 安装 Xcode Command Line Tools：

```sh
xcode-select --install
```

确认 Python 和 Swift 编译器可用：

```sh
python3 --version
swiftc --version
```

### 5.3 创建 Python 环境

```sh
cd /path/to/astra_eval
./setup_colleague.sh
```

脚本会：

1. 创建 `.venv`；
2. 安装 `requirements.txt` 中的固定版本；
3. 从 `encode_video.swift` 编译 `encode_video`；
4. 对核心 Python 文件执行语法检查。

如需指定 Python 或虚拟环境位置：

```sh
PYTHON_BIN=/opt/homebrew/bin/python3 VENV_DIR=$PWD/.venv ./setup_colleague.sh
```

## 6. 启动服务和打开页面

```sh
cd /path/to/astra_eval
source .venv/bin/activate
python3 start_server.py
```

启动器会：

- 检查 `127.0.0.1:8763` 是否已经是 Astra 服务；
- 必要时编译视频编码器；
- 后台启动 `harness/simsvc.py`；
- 将 PID 写入 `runs/server.pid`；
- 将服务输出写入 `runs/server.log`。

浏览器打开：

```text
http://127.0.0.1:8763/live
```

健康检查：

```sh
curl http://127.0.0.1:8763/health
```

正常响应应包含：

```json
{
  "ok": true,
  "physics_hz": 1000,
  "control_hz": 25,
  "eef": "jaw_center",
  "grasp_assist": true,
  "arm_ctrl_step": 0.028,
  "grip_ctrl_step": 0.008,
  "ctrl_smooth": 0.85
}
```

## 7. 一键验收

安装完成后运行：

```sh
./validate_colleague.sh
```

该脚本依次执行：

```sh
python3 test_service.py
python3 check_render.py
python3 start_server.py
python3 verify_http.py
```

通过标准：

- `test_service.py` 显示 `Ran 10 tests` 和 `OK`；
- `check_render.py` 显示 `nbody=28 nq=51 finite=True`；
- `verify_http.py` 显示 `idempotent replay PASS` 和 `stale rejection PASS`；
- 三路 MP4 出现在对应的 `runs/<session_id>/` 目录；
- 页面能显示三路相机和当前会话状态。

当前基线已经通过以上验收。最近一次单瓶 Direct 回放为 `runs/2b4504b82db5`，在控制步 588 前完成首瓶抓取、搬运、入桶和释放。

## 8. 仿真服务协议

服务默认监听 `127.0.0.1:8763`。

### 8.1 创建会话

```http
POST /session
Content-Type: application/json
```

请求：

```json
{
  "layout": {"Rigid": {}, "Geometry": {}, "Table": {}},
  "instruction": "Pick up the bottles and throw them into the dustbin, using handover when needed.",
  "task": "put_bottles"
}
```

`layout` 也可以直接读取 `layouts/put_bottles_into_dustbin_0.json`。`task` 支持 `put_bottles` 和 `classify_objects`。

响应包含 `session_id`、`record_dir` 和初始状态。每次创建会话都会重新加载布局，不复用上一局物体状态。

### 8.2 获取观测

```http
GET /session/<id>/observe
GET /session/<id>/state
GET /session/<id>/result
```

`observe` 返回：

- `images`：三路 base64 图像；
- `shapes`：每路图像的数组形状；
- `state`：14 维状态；
- `ee`：左右末端实测 `xyz`、`quat_wxyz` 和夹爪；
- `instruction`、`observation_id`、`t`、`done`、`score`、`bottles_in`。

相机键为：

```text
cam_base        主相机
left_cam_wrist  左腕相机
right_cam_wrist 右腕相机
```

推理服务使用的兼容键是 `cam_high`、`cam_left_wrist`、`cam_right_wrist`；`client.py` 会自动完成映射。

### 8.3 提交关节动作

```http
POST /session/<id>/act
```

请求字段：

```json
{
  "request_id": "unique-request-id",
  "expected_step": 0,
  "joints": [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]],
  "decision_summary": "short reason"
}
```

`joints` 是 `K x 14`，`K` 为 1–50。顺序为：

```text
left_joint1..left_joint6, left_gripper,
right_joint1..right_joint6, right_gripper
```

前 12 个值是弧度，夹爪是 0–1 归一化值。每个动作必须带 `request_id` 和与当前观测匹配的 `expected_step`。同一个 `request_id` 重复提交同一 payload 会返回原结果；相同步骤的过期请求返回 409。

### 8.4 提交末端目标

```http
POST /session/<id>/eef
```

请求示例：

```json
{
  "request_id": "unique-request-id",
  "expected_step": 0,
  "steps": 1,
  "goals": {
    "left": {
      "xyz": [-0.30, -0.26, 0.92],
      "quat_wxyz": [0.7071068, 0, 0, 0.7071068],
      "gripper": 1.0
    },
    "right": {
      "xyz": [0.30, -0.26, 0.92],
      "quat_wxyz": [0.7071068, 0, 0, 0.7071068],
      "gripper": 1.0
    }
  },
  "decision_summary": "move above bottle"
}
```

每次目标相对实测末端最多 5 cm / 0.35 rad，`steps` 为 1–5。服务每个真实控制步重新求解 IK。`left_ee/right_ee` 定义为两指中间的真实夹持中心，不是腕部原点。

### 8.5 结束并导出录像

```http
POST /session/<id>/finish
```

服务会将会话记录的 PNG 帧编码成：

```text
runs/<session_id>/cam_base.mp4
runs/<session_id>/left_cam_wrist.mp4
runs/<session_id>/right_cam_wrist.mp4
```

每个真实控制步记录一帧，视频输出为 25 fps。原始状态、请求、响应分别记录在 `states.jsonl`、`requests.jsonl`、`responses.jsonl`。

## 9. 控制与环境行为

### 9.1 物理和控制

- 物理步长为 1/1000 秒；每个 25Hz 控制步包含 40 个物理子步。
- 目标在前 32 个物理子步线性插值，后 8 个子步保持。
- 6D 阻尼 IK 同时控制末端位置和姿态，避免腕部姿态漂移。
- 服务端对关节目标做限速和低通，默认值为 `SIM_ARM_CTRL_STEP=0.028`、`SIM_GRIP_CTRL_STEP=0.008`、`SIM_CTRL_SMOOTH=0.85`。
- 夹爪指令增加滞回：`0.20` 以下才确认闭合、`0.80` 以上才确认打开，中间区间保持上一次明确指令；可用 `SIM_GRIP_CLOSE_CMD`、`SIM_GRIP_OPEN_CMD` 调整。这避免策略小幅噪声导致夹爪来回抖动。
- `ASTRA_DIRECT_STEP` 是 `client.py move` 的笛卡尔 waypoint 步长，默认 0.018 m。

### 9.2 抓取辅助

默认 `SIM_GRASP_ASSIST=1`。当同一侧两根手指同时接触同一瓶子且夹爪闭合时，服务建立稳定的抓取约束；夹爪打开后释放。该辅助机制使用 MuJoCo weld equality，在真实双指接触时记录相对位姿、启用约束，明确打开时停用；不再逐步重写物体位置或清零关节速度。它仍是辅助抓取，不代表纯接触物理或官方等价性。

如需审计原始接触物理：

```sh
SIM_GRASP_ASSIST=0 python3 harness/simsvc.py
```

### 9.3 评分

`put_bottles` 的过渡分数为 1/2/3/4 个瓶子入桶对应 10/25/40/100，但评分还要求双夹爪打开。`success` 还要求 4 个瓶子入桶且双臂回到 home。因此单臂单瓶验证可能看到 `bottles_in=1` 但 `score=0`，这是评分条件导致的，不是入桶失败。

## 10. GPT/π 推理服务接入

### 10.1 推理输入

`client.py infer` 从最新观测生成：

```json
{
  "images": {
    "cam_high": "base64 bytes",
    "cam_left_wrist": "base64 bytes",
    "cam_right_wrist": "base64 bytes"
  },
  "shapes": {"cam_high": [480, 640, 3]},
  "state": [14 floats],
  "prompt": "task instruction"
}
```

相机默认使用 `SIM_CAMERA_PROFILE=wide` 的直播视角：主相机覆盖整张桌面，腕相机采用夹爪近景：参考姿态下位于夹爪中心后方 4 cm、上方 18 cm，朝向中心前方 2.5 cm、下方 2.5 cm；垂直视场角为 70°，显示双指及夹持区域，减少机械臂本体入镜。相机固定在对应腕部坐标系，随腕运动而不自动追踪。该配置也影响提供给策略的腕部图像。需要对齐官方 π0.5 图像分布时设置 `SIM_CAMERA_PROFILE=official`。

默认推理地址为 `http://127.0.0.1:8642/infer`，可通过环境变量替换：

```sh
ASTRA_PI05=http://127.0.0.1:8642 python3 client.py infer --session <id>
```

直接调用 π0.5 服务时，图像应为 HWC `uint8` 数组，状态为 14 个浮点数；服务返回 50×14 的绝对关节目标：

```python
import base64, json, urllib.request
import numpy as np

def pi05_infer(images, state, prompt, endpoint="http://127.0.0.1:8642"):
    body = {
        "images": {k: base64.b64encode(v.tobytes()).decode() for k, v in images.items()},
        "shapes": {k: list(v.shape) for k, v in images.items()},
        "state": list(state), "prompt": prompt,
    }
    req = urllib.request.Request(endpoint + "/infer", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        actions = np.asarray(json.load(resp)["actions"])
    assert actions.shape == (50, 14)
    return actions
```

### 10.2 推理输出

`/infer` 必须返回 `actions`，形状为 `(50, 14)` 或 `(1, 50, 14)`，且所有值为有限数。客户端会检查形状后保存 `proposal_<step>.json`。

执行前应重新获取观测并确认 `observation_id` / `expected_step` 没有过期：

```sh
python3 client.py follow --session <id> --steps 1 --reason 'reviewed action'
```

不要在推理服务或客户端里隐式 fallback 到另一个策略；失败应保留请求、响应和观测，方便复盘。

### 10.3 端到端最小流程

```sh
python3 client.py start --task put_bottles --layout 0
# 记录输出的 session_id
python3 client.py observe --session <session_id>
python3 client.py infer --session <session_id>
python3 client.py follow --session <session_id> --steps 1 --reason 'accept first action'
python3 client.py observe --session <session_id>
python3 client.py finish --session <session_id>
```

完整策略测试应由同事固定模型版本、checkpoint、预处理、推理端点和随机种子后再比较结果。

## 11. 已做的环境修复

当前代码包含以下关键修复：

- 场景重建后统一使用新 `MjModel` / `MjData`，避免相机和物体引用错位；
- 修正瓶子和篮筐网格的轴向变换，记录在 `assets/meshes/axis_fix.json`；
- 将 EEF site 移到两指真实夹持中心；
- 用完整 6D 阻尼 IK 固定腕部姿态；
- 统一夹爪归一化与物理长度转换；
- 关闭非任务连杆碰撞，只保留两指任务碰撞；
- 增加可配置的接触触发抓取约束；
- 增加目标限速、低通和 `ASTRA_DIRECT_STEP` 路径限速，降低画面抖动；
- 增加状态、请求、响应和三路相机逐步记录；
- 增加 `request_id` / `expected_step`，防止重复动作和过期动作；
- 评分不再把分数 100 直接视为成功；
- 页面显示真实会话状态，并在等待策略时保持静止。

## 12. 交付内容建议

GitHub 交付时提交源码、资产、布局、文档和测试脚本。不要提交整个 `runs/` 历史目录；它包含大量逐帧 PNG，当前约 11GB。不要把 API token、NPU 凭据、个人绝对路径或私有 checkpoint 放入仓库。完整提交命令见本文档末尾的“GitHub 提交清单”。

## 13. 故障排查

| 现象 | 检查方式 | 处理 |
|---|---|---|
| `swiftc is required` | `swiftc --version` | 安装 Xcode Command Line Tools |
| `address already in use` | `curl http://127.0.0.1:8763/health` | 若是本服务，直接复用；若是其他程序，停止占用者后再启动 |
| 页面显示服务不可达 | 查看 `runs/server.log` 和 `/health` | 重启 `python3 start_server.py` |
| `invalid CoreGraphics connection` | 直接运行 `check_render.py` | 在有图形上下文的 macOS 用户会话中运行 |
| `stale observation` | 检查 `expected_step` | 重新 `observe`，使用最新 `t` |
| `request_id payload conflict` | 同 ID 提交了不同 body | 每个逻辑动作生成新的 UUID |
| `Target exceeds 5 cm` | EEF 目标跳跃过大 | 将目标拆成 waypoint，每次使用实测反馈 |
| 抖动明显 | 查看主相机和腕相机分别表现 | 确认使用当前平滑参数；腕相机的运动会放大末端动作 |
| 推理接口失败 | `curl <ASTRA_PI05>/health` 或检查 `/infer` | 确认端点、输入图像键和 `(50,14)` 输出格式 |

## 14. 复现完成定义

同事完成以下步骤即可认为环境复现成功：

```text
安装依赖成功
→ /health 返回 ok=true
→ 10 项回归测试通过
→ 三路相机可渲染
→ verify_http 幂等和过期请求检查通过
→ 页面显示新会话
→ 至少完成一局单瓶抓取或策略动作回放
→ 生成 MP4、states.jsonl、requests.jsonl、responses.jsonl
```

环境复现成功后，再单独固定 GPT/π 推理服务和模型版本进行策略测试。两者应分别记录：环境版本、推理服务地址、模型/checkpoint、随机种子、layout 编号和会话目录。

## GitHub 提交清单

### 应提交

```text
README.md
requirements.txt
setup_colleague.sh
validate_colleague.sh
start_server.py
client.py
test_service.py
check_render.py
verify_http.py
bridge.py
probe_inference.py
fix_mesh_axes.py
harness/
assets/
layouts/
docs/
encode_video.swift
.gitignore
```

### 不应提交

```text
runs/                  # 运行缓存和逐帧 PNG，当前约 11 GB
dist/
.venv/
encode_video           # 本机编译产物，会由 setup_colleague.sh 生成
*.safetensors *.ckpt *.pth *.pt *.bin *.onnx *.gguf
.env、API token、SSH 私钥、NPU 凭据、私有 checkpoint
```

仓库内的 `.gitignore` 已排除这些内容。首次提交：

```sh
cd /path/to/astra_eval
git init
git add README.md requirements.txt setup_colleague.sh validate_colleague.sh \
  start_server.py client.py test_service.py check_render.py verify_http.py \
  bridge.py probe_inference.py fix_mesh_axes.py harness assets layouts docs \
  encode_video.swift .gitignore
git diff --cached --stat
git commit -m "Add reproducible Astra MuJoCo evaluation environment"
```

公开仓库前请确认 X5、RoboDojo 资产和上游文档的许可证允许再分发。π0.5 镜像应使用完整的 tag 和 digest 记录；模型权重和 API key 不进入 GitHub。

### 夹爪抖动回归

运行 `python3 test_gripper_stability.py`，无需图形环境；保留真实物理、控制与接触，只替换渲染和 HTTP 传输。检查两个夹指（joint7、joint8），而不是只检查策略接口中的 joint7：全程同步误差 <0.3 mm，抬升/搬运每帧位移 <0.5 mm、开度漂移 <1 mm，静止开爪 2 秒峰峰值 <0.1 mm，并要求真实双指接触后抓取、保持、释放、瓶子入桶。

双指加入对称位置驱动、被动阻尼和更紧的 mimic equality；物理积分使用 1 ms 步长。14 维外部动作接口不变，新增内部 follower 驱动不对策略暴露。`states.jsonl` 的 `fingers` 记录两指实际位置和目标。基础测试在 1.12 m 高度水平搬运并在桶口上方释放，避免旧路线碰撞 bottle3 和桌沿。
