# π0.5 训练/推理适配器固定源码核查

日期：2026-09-18。只读调查；未启动实验、未修改生产代码。范围是适配器契约，不是权重数值等价证明。

## 核心结论

作者指定的 `pi05_base_aloha_full_sim_arx-x5_seed_0` 配置实际采用：

```text
adapt_to_pi = False
use_delta_joint_actions = True
asset_id = arx_x5_sim
model = Pi0Config(pi05=True)
```

**应该关闭 ALOHA 专用关节符号/夹爪几何转换，但保留关节 delta→absolute 还原。不能把两个开关一起关闭。**

本地转换报告第7.1节只传 repo_id/assets/base_config，没有显式指定两个开关，因此是否正确取决于安装的 `LeRobotAlohaDataConfig` 默认值。主代理在192在线服务独立读取到 generic OpenPI 默认 `adapt_to_pi=True`，这与下面固定训练配置不一致。该在线观测由主代理提供，本调查独立确认的是上游固定源码的 `False/True`。

## 版本链：SOURCE 的 openpi_commit 不是 PI 主仓库提交

1. [GPT-as-Policy SOURCE.json](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/SOURCE.json#L13) 固定 RoboDojo `ee67a1468510da7624a089164402359f2afc72c8`、openpi `432f82b1758c5b1202e42a3dfe014546dbc50871`。
2. [RoboDojo .gitmodules](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/.gitmodules) 指向 `https://github.com/XPolicyLab/XPolicyLab.git`。GitHub tree API 核实该版本 `XPolicyLab` gitlink SHA正是 `432f82...`。
3. 真正 π0.5 配置在 **XPolicyLab** 的 `policy/Pi_05/openpi/src/openpi/training/config.py`。不是 `policy/Pi_0`，也不是 Physical-Intelligence/openpi 的该SHA（后者查询404）。
4. [作者 checkpoint.py](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/pi05_server/checkpoint.py#L28) 调用 `config.get_config(ROBODOJO_CONFIG)`，只替换 assets 路径，没有覆写上述两开关；[server.py](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/pi05_server/server.py#L26) 将该配置交给 `create_trained_policy`。

## 精确训练配置

[固定 Pi_05 config.py:238](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/training/config.py#L238) 定义：

- 241行 `use_delta_joint_actions: bool = True`。
- 247行 `adapt_to_pi: bool = False`。
- 269–270行将实例 `adapt_to_pi` 显式传入 `AlohaInputs` 和 `AlohaOutputs`，因此不会采用这两个 transform 类自身的True默认值。
- 272–276行使用 `make_bool_mask(6, -1, 6, -1)`，训练输入加 `DeltaActions`，输出加 `AbsoluteActions`；只有12个arm joint做delta，两个gripper保持绝对值。

[seed0配置636–669行](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/training/config.py#L636) 没有覆盖这两个开关，因此继承上述False/True。其余参数：repo_id=`RoboDojo_sim_arx-x5_v30`、asset_id=`arx_x5_sim`、三个camera原生名称cam_high/cam_left_wrist/cam_right_wrist、state=`observation.state`、actions=`action`、prompt=`prompt`、prompt_from_task=True、seed=0、batch_size=256、fsdp_devices=2、num_train_steps=60000。模型初始化权重为 `gs://openpi-assets/checkpoints/pi05_base/params`。

## 错开 adapt_to_pi=True 会做什么

以下均来自[固定 aloha_policy.py](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/policies/aloha_policy.py#L110)：

1. 输入 state 将索引1、2、8、9取负，即左右两臂 joint2、joint3。mask为 `[1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1]`。
2. 输入夹爪走 ALOHA/Interbotix 几何公式：先映射到0.01844–0.05800，再以arm_length=0.036、horn_radius=0.022做arcsin，最后按0.5476–1.6296归一化。不是 X5 原生0–1 opening的恒等变换。
3. 输出动作再次使用上述符号mask；两个夹爪输出按 `(value + 0.5476 + 0.6213) / (1.4910 + 0.6213)` 转换，即 `(value+1.1689)/2.1123`。
4. 因此若网络未经过该几何适配训练，而它输出的原生夹爪数值为0，错误后处理会把它变为约 **0.5534**；输出1则变为约 **1.0268**（后续若clip到1只会掩盖越界，不修复错误）。这能解释“关爪请求始终约0.55”的一种机制，但本调查没有验证当前每个在线输出的网络中间值。

不能只把最终动作joint2/3再翻一次当完整修复：错误适配已经改变了模型输入state及其归一化，必须在policy构建时对齐整个输入/输出管线。

## 归一化与delta顺序

[固定 policy_config.py:75](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/policies/policy_config.py#L75) 明确：输入先data_transforms再Normalize再model_transforms；输出先model输出变换，再Unnormalize，再data输出变换。

因此 `adapt_to_pi` 错误不只改变最终符号：原始X5 state在进入与训练对应的 `arx_x5_sim` norm_stats前就已错误变换。内部delta关节动作则是训练约定的一部分，不应移除；最终服务提供的动作应是绝对关节目标。

## 对本地转换报告的影响

模型结构方面，[固定 pi0_config.py:19](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/models/pi0_config.py#L19) 的默认值为dtype=bfloat16、paligemma=gemma_2b、action_expert=gemma_300m、action_dim=32、action_horizon=50。`pi05=True`自动令max_token_len=200、discrete_state_input=True。这些与本地报告列出的架构一致；`pytorch_compile_mode=None`是本地NPU执行选项差异。未发现seed0配置还覆写其他模型参数；这不构成对所有代码/权重数值等价的证明。

[本地报告7.1](D:/code/versatile-ai/embodied-ai/docs/pi05_conversion_report.md:328) 的“与训练时一致”注释不能由同名类和相同模型架构保证。报告使用 Physical-Intelligence/openpi 未钉提交，示例构造器省略adapt开关；而真实RoboDojo fork修改了这个默认值。

报告中的 shape、finite、非零输出、转换重复性和NPU延迟测试，都不能证明输入/输出坐标语义与训练一致。**这不等价于证明JAX→PyTorch权重转换错误**：转换是否正确与推理适配器是否正确是两个独立问题。

建议后续最小验证：显式设置 `adapt_to_pi=False,use_delta_joint_actions=True`，记录实际transform配置；对同一保存观察使用可控同噪声测试，保存归一化前state、逆归一化后动作和最终absolute action，再进行短程控制验证。本调查没有实施这些改动或测试。
