# 192 服务器部署记录

2026-09-18 连接更新：当前跳板机改为 `root@121.91.172.53`。已实测通过
`ssh -J root@121.91.172.53 root@192.168.0.192` 登录目标机，主机名为
`aura-0916-3.novalocal`。下文旧 IP 保留于历史部署记录，不再用于新连接。

部署日期：2026-09-17。目标：`192.168.0.192`（`aura-0916-3.novalocal`），经 `root@159.138.11.11` 跳转。

## 当前生效方案（同日最终更新）

按用户批准，仿真已切回 Windows 本地：物理、Intel Arc GPU 渲染和录像在本地 `simulation/mujoco` 运行，监听 `127.0.0.1:8763`，不再将此端口转发到服务器。192 的 π0.5 服务保持原部署；远端 MuJoCo 容器保留作备用，未启动新的远端实验。下文服务器渲染配置是此前部署记录，不是当前直播的数据来源。

本地启动：在 `simulation/mujoco` 执行 `.\.venv\Scripts\python.exe start_server.py`。远程推理只需转发 8642：

```powershell
ssh -N -o ConnectTimeout=8 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:8642:192.168.0.192:8642 root@121.91.172.53
```

切换后已运行 `verify_http.py`：会话 `6c687d6cda72`，抬臂 5 步、保持 20 步，幂等及过期请求保护通过。26 条状态（t=0..25）全部有限，三路相机均更新；三路 MP4 全量解码通过，每路 26 帧。输出在本地 `simulation/mujoco/runs/6c687d6cda72/`。测试已正常结束，页面保留最后一帧；未启动正式实验。

本轮最初经跳板读取 π0.5 `/health` 成功，但随后 SSH 连接多次超时，8642 转发未建立。因此本地基础用例通过，远程 π0.5 闭环仍待网络恢复后验收，不能将本轮结果称为完整闭环通过。

## 部署边界

初始部署按照 `pi05_colleague_guide.md` 的分工，192 只承载 π0.5 昇腾推理服务。2026-09-17 后续按用户要求增加独立 MuJoCo 仿真容器：物理、三路相机渲染和录像也在 192 执行，PC 只显示页面和发送控制请求。下文 π0.5 容器配置保持不变；仿真部署见文末。

## 环境

| 项目 | 实测值 |
|---|---|
| 系统 | Huawei Cloud EulerOS 2.0 / ARM64 |
| CPU / 内存 | Kunpeng 920，192 核 / 1.5 TiB |
| NPU | 8 × Ascend 910B3，64 GiB/卡 |
| 驱动 / CANN | 25.5.1 / 8.5.2 |
| Docker | 27.2.0，ascend runtime |
| Python | 3.11.15（容器内） |
| PyTorch / torch_npu | 2.7.1+cpu / 2.7.1.post2 |
| transformers | 4.53.2，openpi SigLIP 补丁检查通过 |
| NumPy / JAX / flax | 2.3.5 / 0.5.3 / 0.10.2 |
| orbax-checkpoint / jaxtyping | 0.11.13 / 0.2.36 |
| MuJoCo | 3.11.0（镜像内依赖；并非 Mac 仿真的 3.3.7 基线） |

空闲卡 HBM 基线约 3203 MiB，而非 32 GB。`npu-smi` 在窄终端换行时可能重复显示边界字符。

## 交付资产与启动

共享存储 `/data` 已挂载；以下交付资产保留原路径：

- 镜像包：`/data/pi05_hybrid/pi05-hybrid-v1.tar.gz`
- 镜像：`pi05-hybrid:v1`
- 本地镜像 ID：`sha256:fec054512e0e2b9acbaa5fa79a843611fdf30d2c5999b30f6135b40194859c66`（本地 image ID，不是 registry manifest digest）
- 权重：`/data/pi05_hybrid/weights/robodojo_pi05_pt/model.safetensors`
- 归一化统计：`/data/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets/arx_x5_sim/norm_stats.json`
- 服务源码：`/data/pi05_hybrid/scripts/pi05_service.py`

首次部署命令（在 192 上执行；已有同名容器时勿重复创建）：

```sh
docker load -i /data/pi05_hybrid/pi05-hybrid-v1.tar.gz
docker run -d --name pi05-service \
  --entrypoint /bin/bash \
  --runtime=ascend --privileged --network host \
  --restart unless-stopped \
  --log-opt max-size=20m --log-opt max-file=3 \
  -e ASCEND_VISIBLE_DEVICES=0 \
  -v /data:/data \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  pi05-hybrid:v1 -c \
  'source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec python3 -u /data/pi05_hybrid/scripts/pi05_service.py npu 8642'
```

实际镜像 `ENTRYPOINT=["bash"]`，因此必须显式覆盖入口，避免原指南中 `bash bash -c ...` 导致的 `cannot execute binary file`。保留文档要求的 privileged 模式；该模式下容器能看见全部 NPU，`ASCEND_VISIBLE_DEVICES=0` 不构成严格设备隔离，服务默认使用 NPU 0。

Docker 已开机自启，容器设置 `unless-stopped`，日志保留最多 3 个 20MB 文件。宿主 Python 无需安装模型依赖。

## 运维与连接

```sh
docker ps --filter name=pi05-service
docker logs --tail 100 pi05-service
curl -fsS http://127.0.0.1:8642/health
docker restart pi05-service
docker stop pi05-service
docker start pi05-service
```

服务在 192 上监听 `0.0.0.0:8642`，无应用层鉴权；保持在受控内网或经 SSH 隧道使用，无需开放公网端口。本次未变更防火墙或云安全组。

客户端通过跳板建立隧道（第二跳使用 192 的 root 认证）：

```sh
ssh -N -J root@159.138.11.11 -L 8642:127.0.0.1:8642 root@192.168.0.192
```

客户端设置 `ASTRA_PI05=http://127.0.0.1:8642`。输入使用 HWC uint8 原始 RGB 字节的 base64 编码，以及 `shapes`；不是 JPEG 文件字节。相机键为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`，状态为 14 维，输出为 `50×14` 绝对关节动作。

## 验收

依赖版本、openpi 补丁与 NPU 可用性检查已通过。模型首次加载约 5 分钟，`GET /health` 返回 `{"ok": true, "device": "npu"}`。

使用共享盘中已有的 RoboDojo 官方演示 episode 2700（瓶子入桶任务），读取三路 PNG 和对应 14 维状态，通过真实 HTTP `/infer` 检查：

| 演示帧偏移 | 输出形状 | 所有值有限 | 服务推理耗时 | HTTP 总耗时 |
|---|---|---|---|---|
| 0 | 50×14 | 是 | 8115.2 ms（首次预热） | 8143.3 ms |
| 50 | 50×14 | 是 | 572.5 ms | 611.7 ms |
| 100 | 50×14 | 是 | 562.7 ms | 588.8 ms |

192 本机回环地址、192 的内网地址以及跳板机到 `http://192.168.0.192:8642/health` 的访问均通过。

验收时容器运行正常，自动重启次数为 0。这证明当前部署的观测到动作推理链路可用；未执行闭环机器人任务，也未将上述检查作为策略成功率评测。

## 服务器端仿真、渲染和录像（同日追加）

新增容器 `robodojo-sim`，与 `pi05-service` 分离。源代码、资产、离线安装包和运行记录在本机 NVMe `/data_nv0/robodojo-sim-20260917`，不占用 PC 的仿真内存。原 PC 的 `runs/` 历史记录未删除。

运行环境：Python 3.11.15、MuJoCo 3.3.7、NumPy 2.0.2、Pillow 11.3.0、imageio-ffmpeg 0.6.0；系统安装 `libosmesa6` 和 FFmpeg 4.4.2。独立 `/opt/sim-venv`，清空继承的 `PYTHONPATH` 后 `pip check` 通过。GL 实测为 `llvmpipe (LLVM 15.0.7, 128 bits)` / Mesa 23.2.1，属于 **CPU 软件渲染，不是昇腾 NPU 图形渲染**。

依赖已固化为本地镜像 `robodojo-sim:osmesa-20260917`，镜像 ID：`sha256:c517102564f9533c5e78ca8074d8275621a7ee61829844b115abe2e934c516a2`。由 `pi05-hybrid:v1` 安装上述系统库及独立 Python 环境后生成；离线 ARM64 wheels 保留在 `sim-wheels/`，可用 `pip install --no-index --find-links=/app/sim-wheels ...` 重装。镜像不包含绑定挂载的项目文件和运行记录，备份时需同时保存项目目录。

新建运行容器的配置如下（已有同名容器时使用 `docker start`，不要重复创建）：

```sh
docker run -d --name robodojo-sim --runtime=runc \
  --entrypoint /opt/sim-venv/bin/python --network host \
  --restart unless-stopped --memory 64g --cpus 16 \
  --log-opt max-size=20m --log-opt max-file=3 \
  -e PYTHONPATH= -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
  -e LP_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  -e SIMHOST=192.168.0.192 -e SIMPORT=8763 \
  -e ASTRA_SIM=http://192.168.0.192:8763 -e ASTRA_HTTP_TIMEOUT=300 \
  -e IMAGEIO_FFMPEG_EXE=/usr/bin/ffmpeg \
  -v /data_nv0/robodojo-sim-20260917:/app -w /app \
  robodojo-sim:osmesa-20260917 -u harness/simsvc.py
```

容器不需要 privileged 或 NPU 设备。端口仅绑定 192 内网地址；未开放公网端口或修改安全组。HTTP 无鉴权，只适合可信内网/SSH 转发使用。

Windows 显示页面时建立转发（本机 8763 必须未被本地仿真占用）：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8763:192.168.0.192:8763 root@159.138.11.11
```

然后打开 `http://127.0.0.1:8763/live`。此时网页只接收服务器生成的图像，PC 不运行 MuJoCo。SSH 转发进程关闭后网页断开，但服务器容器继续运行。

服务管理：

```sh
docker logs --tail 80 robodojo-sim
docker stats robodojo-sim --no-stream
curl -fsS http://192.168.0.192:8763/health
docker restart robodojo-sim
```

重启会清除内存中的活动会话，不会删除 `runs/` 文件。服务启动后等待创建场景和发送动作，不会自行运行策略。远端控制命令示例：

```sh
docker exec robodojo-sim /opt/sim-venv/bin/python client.py start
docker exec robodojo-sim /opt/sim-venv/bin/python client.py infer --session SESSION_ID
docker exec robodojo-sim /opt/sim-venv/bin/python client.py follow --session SESSION_ID --steps 3
docker exec robodojo-sim /opt/sim-venv/bin/python client.py finish --session SESSION_ID
```

每批动作后重新观测、推理，再执行下一批，不能复用过期 proposal。客户端默认超时仍为 90 秒，服务器配置 `ASTRA_HTTP_TIMEOUT=300`；Windows 手动控制时也应设置此变量。超时不代表动作未执行，不要盲目重新提交，应先查看最新步数。软件渲染较慢，建议小批次（例如 3 步）保持页面可响应；单线程服务会在一个动作批次结束后响应页面请求。

### 实测验收

- 服务器 15 项回归全部通过，138.916 秒；含分类场景、固定几何、计数过滤、资源释放和夹爪限幅。进程峰值 RSS 7,900,104 KiB，约 7.53 GiB。
- 首次集成会话 `309491d5581c`：幂等重放和过期请求拒绝通过；20 步批次超过原 90 秒客户端等待，但服务实际推进到第 25 步。为此增加可配置超时，不自动重放动作。
- 同一会话真实相机输入调用 π0.5，输出有限 `50×14` 动作，推理 576.1 ms；执行前 3 步成功，耗时 14.032 秒，最终 `t=28`。这不是实时 25fps；25Hz 指仿真时间和录像播放时间。
- 三路 MP4 全部解码无错误：640×480、25fps、29 帧（初始帧加 28 步）；结束后最终观测仍可读取。
- 配置 300 秒等待后，`verify_http.py` 完整复测通过，会话 `9ee14b8bb2ca`：25 步、幂等、过期保护及三路录像均成功，不再出现客户端提前超时。
- 上述为环境/接口验收，不是成功抓取任务或官方 benchmark 成绩；分类策略成功率和全部布局尚未验证。
