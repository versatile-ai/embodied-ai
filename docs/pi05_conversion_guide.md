# π0.5 权重 JAX→PyTorch 转换指南

> 适用对象：需要在昇腾 NPU 或 CPU 上部署 RoboDojo 官方 π0.5 微调权重的工程师。
> 转换产物：PyTorch safetensors 格式，可直接用 openpi 推理管线加载。

---

## 1 前置条件

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 | 必须（3.12 不兼容 openpi 依赖） |
| JAX | 0.5.3 | 仅转换时需要，推理不需要 |
| jaxlib | 0.5.3 | CPU 版即可（aarch64 无 CUDA wheel） |
| orbax-checkpoint | 0.11.13 | 必须钉死此版本 |
| flax | 0.10.2 | 必须钉死此版本 |
| jaxtyping | 0.2.36 | 必须钉死此版本 |
| numpy | 2.3.x | 2.4+ 会炸 jax（StringDType 改名） |
| transformers | 4.53.2 | + openpi 补丁（见 §2） |
| torch | 2.7.1 | 与 torch_npu 兼容 |
| safetensors | any | |

### 不可行的版本组合（踩过的坑）

| 尝试 | 结果 |
|---|---|
| transformers 4.57.x / 5.5.x | 缺 siglip.check 模块 → ImportError |
| jax 0.10.2 + numpy 2.4 | numpy StringDType 改名 → AttributeError |
| orbax 0.10.x / 0.13.x | API 不兼容 |
| lerobot main (0.6.2) | 需要 Python ≥3.12 |
| jaxtyping ≠ 0.2.36 | openpi monkeypatch 找不到目标 |

---

## 2 环境搭建

### 2.1 安装 pinned 依赖

pip install "jax==0.5.3" "jaxlib==0.5.3" "flax==0.10.2" \
  "orbax-checkpoint==0.11.13" "jaxtyping==0.2.36" "numpy==2.3.5" \
  "transformers==4.53.2" "torch==2.7.1" \
  jaxlib flax optax tyro einops ml_collections sentencepiece \
  imageio imageio-ffmpeg decorator beartype

### 2.2 应用 openpi transformers 补丁（必须）

git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r openpi/src/openpi/models_pytorch/transformers_replace/* $SITE_PACKAGES/transformers/

补丁内容：AdaRMS 支持、激活精度控制、KV cache 不更新。

### 2.3 验证补丁

from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()

---

## 3 源权重获取

HF dataset: RoboDojo-Benchmark/RoboDojo
路径: ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-{0|1|2}/59999/
大小: ~44.7GB per seed（转换只需 params/ 目录）

export HF_HUB_DISABLE_XET=1  # 必须，否则大文件 0 字节卡死

---

## 4 执行转换

cd /path/to/openpi
export PYTHONPATH=/path/to/openpi/src

python3 examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /data/pi05_hybrid/robodojo_ckpt/.../59999 \
  --config-name pi05_aloha \
  --output-path /data/pi05_hybrid/weights/robodojo_pi05_pt \
  --precision bfloat16

产物: model.safetensors (~6.8G) + config.json

---

## 5 转换后必须做的清理

### 5.1 清零未使用的 lm_head（必须）

JAX π0.5 模型没有 gemma_expert.lm_head 参数（不参与推理），
但 PyTorch 架构定义自带 → 每次转换留下不同的随机初始化值。

from safetensors.torch import load_file, save_file
import torch
w = load_file("model.safetensors")
key = "paligemma_with_expert.gemma_expert.lm_head.weight"
w[key] = torch.zeros_like(w[key])
save_file(w, "model.safetensors")

### 5.2 验证确定性

跑两次转换，对比 812 个 tensor，清零 lm_head 后应全 < 1e-6。

---

## 6 部署到昇腾 NPU

### 6.1 环境要求
- torch_npu 2.7.1.post2 + CANN 8.5.2 + source set_env.sh

### 6.2 关键代码

cfg = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50,
                            max_token_len=200, pytorch_compile_mode=None)
model = PI0Pytorch(cfg)
safetensors.torch.load_model(model, "model.safetensors")
model = model.to("npu").eval()

### 6.3 NPU 已知坑

| 坑 | 修复 |
|---|---|
| GEInitialize 失败 | source set_env.sh |
| 缺 decorator 包 | pip install decorator |
| torch.compile 崩 | pytorch_compile_mode=None |
| bool mask SDPA | 已修复（2.7.1） |

---

## 7 推理服务化

端点: http://<155_IP>:8642/infer
输入: 3×base64 JPEG (480×640) + state[14] + prompt
输出: (50, 14) 关节角 + 延迟 ms
详细接口见: docs/eval_test_manual.md

---

## 8 坑汇总（14 项）

| # | 问题 | 修复 |
|---|---|---|
| 1 | siglip.check ImportError | openpi 补丁 |
| 2 | transformers 版本 | 钉 4.53.2 |
| 3 | jax/numpy 不兼容 | numpy 2.3.5 |
| 4 | lerobot.common 缺失 | shim 模块 |
| 5 | tokenizer gated | HF token |
| 6 | HF 下载卡死 | HF_HUB_DISABLE_XET=1 |
| 7 | NPU 初始化失败 | source set_env.sh |
| 8 | 缺 decorator | pip install |
| 9 | torch.compile 崩 | mode=None |
| 10 | macOS GL 主线程 | 单线程 HTTPServer |
| 11 | lm_head 随机 | 转换后清零 |
| 12 | 物体被弹飞 | kp=1000+重力前馈 |
| 13 | 篮子滑动 | 修碰撞壳 |
| 14 | 躺瓶滚动 | 盒体近似 |
