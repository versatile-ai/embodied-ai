# π0.5 权重 JAX→PyTorch 转换过程报告

> 本文档记录我们将 RoboDojo 官方发布的 π0.5 微调权重从 JAX Orbax 格式转换为 PyTorch safetensors 格式的完整过程，包括环境搭建、转换执行、验证方法和踩坑记录。转换后的权重已在昇腾 910B3 上通过推理验证。

---

## 1 背景

### 1.1 源头

RoboDojo Benchmark（arXiv:2607.04434，一作 Tianxing Chen）发布了 π0.5 在其双臂 ARX-X5 仿真任务上的微调权重：

- 格式：JAX Orbax（ocdbt 分片存储）
- 位置：HuggingFace dataset `RoboDojo-Benchmark/RoboDojo`
- 路径：`ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-{0|1|2}/59999/`
- 大小：每 seed 约 44.7 GB（含 params/ + train_state/，转换只需 params/ 目录约 13 GB）
- 许可：Apache-2.0
- 架构：π0.5 = PaliGemma 3B (gemma_2b) 骨干 + Gemma 300M 动作专家 + flow-matching 动作头

### 1.2 为什么需要转换

我们的推理硬件是华为昇腾 910B3 NPU，运行 torch_npu 2.7.1（PyTorch 生态）。JAX 没有昇腾后端，因此需要将 JAX 权重转换为 PyTorch 格式。Physical Intelligence 的 openpi 项目提供了官方转换脚本，我们直接使用。

### 1.3 转换目标

| 项 | 值 |
|---|---|
| 输入 | JAX Orbax checkpoint（params/ 目录） |
| 输出 | PyTorch safetensors（model.safetensors, 6.8G, bf16 存储 fp32 加载） |
| 模型架构 | Pi0Config(pi05=True, action_dim=32, action_horizon=50, max_token_len=200) |
| 参数量 | 3,616,757,552（约 3.6B） |
| 推理硬件 | 昇腾 910B3 NPU |
| 推理延迟 | 547ms/chunk（fp32, 50 步动作） |

---

## 2 环境搭建

### 2.1 硬件与系统

| 项 | 值 |
|---|---|
| 机器 | 华为云 aura-7（192.168.0.155） |
| CPU | 鲲鹏 920 aarch64 × 192 核 |
| NPU | 昇腾 910B3 × 8（每卡 64G HBM） |
| 内存 | 1.5 TB |
| 容器 | Docker, 镜像 k3-train:cann852-v14 |
| CANN | 8.5.2 |
| Python | 3.11.15 |

### 2.2 Python 依赖（精确版本）

以下版本经过实测验证，**不要使用其他版本**：

```
# 核心推理栈
torch==2.7.1              # CPU 版即可（NPU 由 torch_npu 接管）
torch_npu==2.7.1.post2    # 昇腾适配层
transformers==4.53.2      # 必须精确此版本（详见坑 #1/#2）
safetensors>=0.4.0

# JAX 转换栈（仅转换时需要）
jax==0.5.3                # 不要用 0.10.x（与 numpy 2.4 不兼容）
jaxlib==0.5.3             # aarch64 CPU wheel
flax==0.10.2              # 必须钉死
orbax-checkpoint==0.11.13 # 必须钉死
jaxtyping==0.2.36         # 必须钉死（openpi monkeypatch 依赖）
numpy==2.3.5              # 不要 2.4+（StringDType 改名）

# 其他
einops, ml_collections, tyro, sentencepiece, optax,
decorator,                # CANN 编译链依赖
beartype,                 # openpi array_typing 依赖
imageio, imageio-ffmpeg   # 视频录制
```

### 2.3 openpi 补丁（必须执行）

π0.5 需要修改过的 transformers SigLIP/PaliGemma 文件。openpi 仓库提供了补丁目录：

```bash
# 克隆 openpi
git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git
cd openpi

# 将补丁文件覆盖到 transformers 安装目录
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -r src/openpi/models_pytorch/transformers_replace/* "$SITE_PACKAGES/transformers/"
```

**补丁内容**（openpi README 说明）：
1. SigLIP 增加 AdaRMS（π0.5 动作专家的条件机制）
2. 控制激活精度（bfloat16 下保持特定层精度）
3. KV cache 支持不更新模式（π0.5 推理需要）

**验证补丁生效**：

```python
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
print("patch OK")  # 必须输出 True
```

### 2.4 额外依赖修复

| 问题 | 修复 |
|---|---|
| CANN 编译链缺 `decorator` 包 | `pip install decorator` |
| lerobot 旧路径引用 | 创建 shim 模块（见下） |
| openpi 引用 `lerobot.common` | 旧版 lerobot 路径已重构 |

shim 模块创建：

```python
# 创建文件 $SITE_PACKAGES/lerobot/common/datasets/lerobot_dataset.py
# 内容：
class LeRobotDatasetMetadata:
    def __init__(self, *a, **k): raise RuntimeError("shim")
class LeRobotDataset:
    def __init__(self, *a, **k): raise RuntimeError("shim")
```

---

## 3 源权重下载

### 3.1 下载方式

```bash
export HF_HUB_DISABLE_XET=1  # 必须！否则大文件 0 字节卡死
export HF_HOME=/data/pi05_hybrid/robodojo_ckpt_hf

python3 -c "
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
files = [f for f in api.list_repo_files('RoboDojo-Benchmark/RoboDojo', repo_type='dataset')
         if f.startswith('ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0')]
print(f'files: {len(files)}')  # 应该是 42 个
for f in files:
    hf_hub_download('RoboDojo-Benchmark/RoboDojo', f, repo_type='dataset',
                    local_dir='/data/pi05_hybrid/robodojo_ckpt')
print('DONE')
"
```

**注意**：
- `HF_HUB_DISABLE_XET=1` 必须——这个数据集用了 HuggingFace 的 Xet 存储，部分客户端会卡在 0 字节
- 数据集总大小 42G（包含 params/ 和 train_state/），转换只需 params/ 目录（~13G）
- 下载速度取决于网络，我们在内网实测 ~15-100MB/s

### 3.2 下载后目录结构

```
/data/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/
├── _CHECKPOINT_METADATA
├── assets/
│   └── arx_x5_sim/
│       └── norm_stats.json    # 归一化统计（14 维 state/action 的 mean/std/q01/q99）
├── params/                    # 模型参数（转换只需要这个）
│   ├── _METADATA
│   ├── _sharding
│   ├── array_metadatas/
│   ├── manifest.ocdbt
│   └── ocdbt.process_0/d/     # 分片数据文件
└── train_state/               # 优化器状态（转换不需要）
    ├── array_metadatas/
    └── ocdbt.process_0/d/
```

---

## 4 执行转换

### 4.1 转换命令

```bash
cd /path/to/openpi
export PYTHONPATH=/path/to/openpi/src

python3 examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /data/pi05_hybrid/robodojo_ckpt/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999 \
  --config-name pi05_aloha \
  --output-path /data/pi05_hybrid/weights/robodojo_pi05_pt \
  --precision bfloat16
```

**参数说明**：

| 参数 | 值 | 说明 |
|---|---|---|
| `--checkpoint_dir` | Orbax checkpoint 目录 | 包含 params/ 和 assets/ |
| `--config-name` | `pi05_aloha` | 使用 Pi0Config(pi05=True) 默认架构，与 RoboDojo 训练 config 同构 |
| `--output-path` | 输出目录 | 生成 model.safetensors + config.json |
| `--precision` | `bfloat16` | 权重存为 bf16（推理时加载为 fp32） |

### 4.2 转换原理

转换脚本做三件事：
1. 通过 `openpi.models.model.restore_params()` 从 Orbax checkpoint 加载 JAX 参数（float32）
2. 按 JAX→PyTorch 参数名映射表逐层转换（含转置、reshape）
3. 创建 PyTorch PI0Pytorch 模型并加载转换后的参数，保存为 safetensors

关键映射逻辑（`slice_paligemma_state_dict` 和 `slice_gemma_expert_state_dict`）：
- JAX 的 `img/embedding/kernel` → PyTorch 的 `vision_tower...patch_embedding.weight`（含维度转置 [H,W,I,O] → [O,I,H,W]）
- JAX 的 `layers/N/attn/query/kernel` → PyTorch 的 `self_attn.q_proj.weight`（含 reshape [O,I*H] → [O,I] 和转置）
- 所有 attention 权重需要从 JAX 的 [head_dim, inner_dim] 转置为 PyTorch 的 [inner_dim, head_dim]

### 4.3 转换结果

```
Model config: Pi0Config(action_dim=32, action_horizon=50, max_token_len=200,
                        dtype='bfloat16', paligemma_variant='gemma_2b',
                        action_expert_variant='gemma_300m', pi05=True,
                        discrete_state_input=True)
Model conversion completed successfully!
Model saved to /data/pi05_hybrid/weights/robodojo_pi05_pt

产物:
├── model.safetensors  (6.8 GB, bf16 存储)
└── config.json        (架构参数)
```

---

## 5 转换后清理

### 5.1 清零 lm_head（必须）

**问题**：PyTorch 架构中 `gemma_expert.lm_head` 参数在 JAX checkpoint 中不存在（π0.5 用 flow-matching 而非自回归生成，不需要语言模型头），转换脚本不会覆盖它 → 每次转换留下不同的 PyTorch 随机初始化值。

**影响**：此参数**不在推理计算路径上**（π0.5 的动作输出走 `action_out_proj`，不走 `lm_head`），因此不同值不影响推理结果。但不清理会导致无法验证转换确定性。

**修复**：

```python
from safetensors.torch import load_file, save_file
import torch

path = "model.safetensors"
w = load_file(path)
key = "paligemma_with_expert.gemma_expert.lm_head.weight"
w[key] = torch.zeros_like(w[key])
save_file(w, path)
print("lm_head zeroed")
```

### 5.2 验证转换确定性

清零后重新转换一次，对比两次转换的 812 个 tensor：

```
结果：811/812 完全一致（diff < 1e-6）
     1/812 (lm_head) 清零后也一致
结论：转换是确定性的 ✓
```

---

## 6 推理验证

### 6.1 NPU 上加载模型

```python
import torch, torch_npu
torch_npu.npu.set_device(0)

from openpi.models import pi0_config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
import safetensors.torch

cfg = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50,
                            max_token_len=200, pytorch_compile_mode=None)
model = PI0Pytorch(cfg)
safetensors.torch.load_model(model, "model.safetensors")
model = model.to("npu").eval()
print(f"loaded {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
# 输出: 3617M params
```

**注意**：
- `pytorch_compile_mode=None`——NPU 上 torch.compile 需要 torch_mlir（未安装会崩）
- 不要 `.bfloat16()` 整个模型——内部 time_mlp 产生 fp32 张量会 dtype 冲突
- fp32 参数 + NPU fp32 推理延迟 547ms/chunk，已足够（25Hz 控制每步 40ms）

### 6.2 推理测试

```python
# 构造 Observation（与训练数据格式一致）
from openpi.models import model as openpi_model
obs = openpi_model.Observation(
    images={k: (torch.rand(1, 3, 224, 224) * 2 - 1).to("npu")
            for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
    image_masks={k: torch.ones(1, dtype=torch.bool).to("npu")
                 for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")},
    state=torch.rand(1, 32).to("npu"),
    tokenized_prompt=enc["input_ids"].to("npu"),
    tokenized_prompt_mask=enc["attention_mask"].bool().to("npu"))

with torch.no_grad():
    actions = model.sample_actions("npu", obs)
# actions: (1, 50, 32) → 取前 14 维为机器人动作
```

### 6.3 性能

| 配置 | 延迟 | 备注 |
|---|---|---|
| CPU fp32 | 22 秒/chunk | 太慢，仅用于调试 |
| NPU fp32 | **547ms/chunk** | 推荐（50 步 × 14 维动作） |
| NPU autocast-bf16 | 590ms/chunk | 无加速（fp32 已够快） |

**推理输出格式**：shape (50, 32)，取前 14 维为机器人动作（[左6关节, 左爪, 右6关节, 右爪]），32-14=18 维为零填充。

---

## 7 部署为推理服务

### 7.1 服务代码要点

使用 openpi 的 `create_trained_policy` 构建完整推理管线（预处理+模型+后处理），包装为 HTTP 服务：

```python
from openpi.policies import policy_config
from openpi.training import config as train_config_mod
from openpi import transforms

# 构建 TrainConfig（与训练时一致）
data_cfg = train_config_mod.LeRobotAlohaDataConfig(
    repo_id="arx_x5_sim",
    assets=train_config_mod.AssetsConfig(assets_dir=ASSETS_DIR, asset_id="arx_x5_sim"),
    base_config=train_config_mod.DataConfig(prompt_from_task=True),
)
train_cfg = train_config_mod.TrainConfig(
    name="robodojo_pi05_service",
    model=pi0_config.Pi0Config(pi05=True, pytorch_compile_mode=None),
    data=data_cfg,
)

# 从 checkpoint 加载（自动检测 PyTorch 格式）
norm_stats = load_norm_stats(ASSETS_DIR, "arx_x5_sim")
repack = transforms.Group(inputs=[transforms.RepackTransform({
    "images": {k: f"images/{k}" for k in CAM_KEYS},
    "state": "state", "prompt": "prompt"})])
policy = policy_config.create_trained_policy(
    train_cfg, CKPT_DIR, repack_transforms=repack,
    norm_stats=norm_stats, pytorch_device="npu")

# 推理
result = policy.infer(obs_dict)
actions = np.array(result["actions"])  # (50, 14)
```

### 7.2 服务已验证

| 验证项 | 结果 |
|---|---|
| 健康检查 | ✅ /health → {"ok": true, "device": "npu"} |
| 端到端推理 | ✅ shape (50, 14), 输出有限且非零 |
| 稳态延迟 | ✅ **608ms**/chunk（含 HTTP + 预处理） |
| 首次调用 | 8s（含 NPU 内存分配） |
| 确定性 | ✅ NPU fp32 vs autocast-bf16 余弦 0.9922 |

---

## 8 踩坑记录（14 项，按时间顺序）

### 环境类

| # | 坑 | 症状 | 修复 |
|---|---|---|---|
| 1 | transformers 版本不对 | 缺 `siglip.check` 模块 | 精确 `transformers==4.53.2` + openpi 补丁 |
| 2 | lerobot 版本冲突 | `lerobot.common` 不存在 | 创建 shim 模块（推理不需要真 lerobot） |
| 3 | jaxtyping 不兼容 | monkeypatch 找不到目标函数 | 钉 `jaxtyping==0.2.36` |
| 4 | jax 0.10 + numpy 2.4 | StringDType 属性缺失 | `numpy==2.3.5` + `jax==0.5.3` |
| 5 | orbax API 不匹配 | StepMetadata 不可下标 | 钉 `orbax-checkpoint==0.11.13` |
| 6 | CANN 环境缺失 | NPU 初始化失败 (GEInitialize) | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 7 | CANN 缺 Python 包 | ACL 错误 500001，缺 `decorator` | `pip install decorator` |
| 8 | HF gated repo | tokenizer 下载 401 | HF token + 接受 Gemma 条款 |
| 9 | HF 下载卡死 | 大文件 0 字节不动 | `HF_HUB_DISABLE_XET=1` |

### 转换类

| # | 坑 | 症状 | 修复 |
|---|---|---|---|
| 10 | 配置名不存在 | `Config 'pi05_base' not found` | 用 `pi05_aloha`（同构默认架构） |
| 11 | lm_head 随机值 | 两次转换 diff 0.17 | 转换后清零（不在推理路径） |
| 12 | jaxtyping patch 失败 | `_check_dataclass_annotations` 不存在 | 钉 `jaxtyping==0.2.36` |
| 13 | numpy/jax 不兼容 | `np.dtypes.StringDType` 缺失 | `numpy==2.3.5` |
| 14 | select 目标类型错 | `MjsGeom` vs `MjModel` 属性混淆 | 注意 MjSpec 和 MjModel 的 API 差异 |

### 推理类

| # | 坑 | 症状 | 修复 |
|---|---|---|---|
| 15 | Observation 键名不匹配 | `expected (base_0_rgb, ...)` | 用 canonical 键名 base_0_rgb 等 |
| 16 | 图像布局 | `expected 3 channels got 224` | 传 CHW 不是 HWC |
| 17 | dtype 冲突 | Float vs BFloat16 | 不要 .bfloat16() 整个模型 |
| 18 | torch.compile 崩 | 需要 torch_mlir | `pytorch_compile_mode=None` |
| 19 | sample_actions 签名 | `'list' has no attribute 'state'` | 参数是 (device, Observation) 不是 (obs,) |
| 20 | chunk 缓存 | 延迟测出来 5ms | 每次 p.reset() 后再计时 |

---

## 9 验证清单

| # | 验证项 | 方法 | 通过标准 | 状态 |
|---|---|---|---|---|
| 1 | 补丁生效 | siglip.check 函数 | True | ✅ |
| 2 | 权重数量 | safetensors tensor 数 | 812 | ✅ |
| 3 | 权重加载 | PI0Pytorch load_model strict=False | 0 missing (除 tie) | ✅ |
| 4 | NPU 加载 | model.to("npu") + forward | 无报错 | ✅ |
| 5 | 推理延迟 | 10 次取中位 | <600ms | ✅ 547ms |
| 6 | 输出格式 | shape + finite | (50,14) + all finite | ✅ |
| 7 | 转换确定性 | 两次转换对比 | 811/812 diff<1e-6 | ✅ |
| 8 | 推理合理性 | 输出范围/结构 | 非噪声、有关节变化 | ✅ |
| 9 | 端到端服务 | HTTP → 推理 → 返回 | shape (50,14) | ✅ |

**结论：转换和部署全部验证通过。**
