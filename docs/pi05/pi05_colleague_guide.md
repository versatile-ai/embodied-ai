# π0.5 推理服务交付包（Docker）

## 包含内容

| 文件 | 大小 | 说明 |
|---|---|---|
| `/data/pi05_hybrid/pi05-hybrid-v1.tar.gz` | 8.9G | Docker 镜像（代码+依赖，**权重不在镜像内**） |
| `/data/pi05_hybrid/weights/robodojo_pi05_pt/` | 6.8G | **模型权重（共享盘，运行时挂载加载）** |

| `/data/pi05_hybrid/robodojo_ckpt/` | 42G | 原始 JAX checkpoint（可选，重新转换用） |

## 同事快速启动（3 条命令）

```bash
# 1. 加载镜像
docker load < /data/pi05_hybrid/pi05-hybrid-v1.tar.gz

# 2. 启动推理服务
docker run -d --name pi05-service \
  --privileged --network host \
  -e ASCEND_VISIBLE_DEVICES=0 \
  -v /data:/data \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  pi05-hybrid:v1 \
  bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
           python3 /data/pi05_hybrid/scripts/pi05_service.py npu 8642"

# 3. 验证
curl -s http://localhost:8642/health
# → {"ok": true, "device": "npu"}
```

## 调用接口

```python
import base64, json, urllib.request
import numpy as np

def pi05_infer(images, state, prompt):
    """
    Args:
        images: dict，3 个 H×W×3 uint8 数组
                键名: cam_high / cam_left_wrist / cam_right_wrist
        state:  list of 14 floats
                [左6关节角(rad), 左爪(0-1), 右6关节角(rad), 右爪(0-1)]
        prompt: str，任务指令（英文）

    Returns:
        np.ndarray (50, 14)
        50 步 × 14 维关节角序列（绝对位置，25Hz 执行）
        每步 = [左J1..J6(rad), 左爪(0-1), 右J1..J6(rad), 右爪(0-1)]
    """
    req = {
        "images": {k: base64.b64encode(v.tobytes()).decode()
                   for k, v in images.items()},
        "shapes": {k: list(v.shape) for k, v in images.items()},
        "state": list(state),
        "prompt": prompt
    }
    r = urllib.request.urlopen(urllib.request.Request(
        "http://<155_IP>:8642/infer",
        data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"}), timeout=120)
    return np.array(json.load(r)["actions"])
```

## 环境版本（镜像内已装好）

| 依赖 | 版本 |
|---|---|
| PyTorch | 2.7.1 (CPU) |
| torch_npu | 2.7.1.post2 |
| CANN | 8.5.2 |
| transformers | 4.53.2 + openpi 补丁 |
| openpi | Physical-Intelligence/openpi |
| MuJoCo | 3.11.0 |
| Python | 3.11 |

## 注意事项

0. **权重在共享盘不在镜像内**：Docker 镜像只有代码+依赖；模型权重从 `-v /data:/data` 挂载的共享盘加载。同事需要 /data 的读写权限。
1. **需要昇腾 NPU**：推理跑在 910B3 上，需要 `--privileged` + driver 挂载
2. **首次加载约 4 分钟**：权重 6.8G 从共享盘读入内存
3. **延迟 ~570ms/chunk**：含预处理+NPU前向+后处理
4. **camera 图像格式**：HWC uint8，服务端自动转 CHW + resize 224
5. **镜像基于**：`k3-train:cann852-v14` + 我们的修改
