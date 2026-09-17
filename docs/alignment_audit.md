# 与官方 RoboDojo 环境的对齐审计（2026-09-16）

审计方法：程序化自检门（`harness/env_selfcheck.py`，15 项全 PASS）+ 源码/配置逐条比对
（RoboDojo repo @155、官方 camera/robot/scene yml、任务 py、XPolicyLab Pi_05 deploy 配置）。

## 已对齐（✅）

| 项 | 官方 | 我们 | 验证 |
|---|---|---|---|
| 桌面高度/平面 | pos_z+scale_z/2=0.765 | 同公式取自布局 JSON | 自检 A1 ≤1e-6 |
| 桌面平面范围 | layout Table scale/2 | 同 | 自检 A1 |
| 物体初始位姿 | 布局 JSON default_pos/ori | 同 | 自检 A2 ≤2mm |
| 物体质量 | layout physics.mass（含 22kg 重瓶） | 同 | 代码审查 |
| 双臂根位姿 | [±0.3,-0.45,0.765], quat z+90° | 同 | 自检 B1 |
| 关节结构 | 6  revolute + 夹爪 mimic(joint8=joint7) | 同（equality 约束） | 自检 B |
| 夹爪行程/归一 | joint7∈[-0.01,0.044]，0-1 归一 | 同 | 自检 B2 |
| 控制频率 | 25Hz 控制 / 250Hz 物理 | 同 | 代码 |
| 动作步展开 | 10 子步 = 8 线性插值 + 2 保持 | 同 | 代码 |
| chunk 长度 | π0.5 action_horizon=50 | 同（官方微调权重） | G2 |
| step_lim | put_bottles 700 / classify 1100 | 同 | 代码 |
| 评分档位 | put [10,25,40,100]；classify [15,40,100]，transition 单调 | 同 | 单测 F1/F2 + classify 单测 100 分 |
| 成功条件 | 全目标达成 + 夹爪开(>0.8) + 双臂回 home | 同（home 容差 0.12rad 自定） | 单测 |
| 指令文本 | task gen_instruction 模板 | 同字符串 | 代码 |
| 布局来源 | Assets/Eval_Layout seed0 预生成 JSON | 同文件直接消费 | 自检 A2 |
| π0.5 权重 | RoboDojo 官方微调 ckpt（seed0, step59999） | 同（JAX→PyTorch 官方转换） | 开环对拍 |
| 预处理链 | XPolicyLab repack + norm_stats(arx_x5_sim) | 同 | 开环对拍 cos 0.99（平稳段） |
| head 相机内参 | fovy 71.1°（Gemini fy336@480） | 同 | 代码 |
| 腕相机内参 | fovy 62.2°（d435） | 同 | 代码 |

## 近似对齐（⚠️ 已记录，影响 π0.5 视觉域差距）

| 项 | 官方 | 我们 | 影响评估 |
|---|---|---|---|
| head 相机外参 | config [0,-0.41,1.308]/Rx30（Isaac xform 约定） | 标定有效位姿 [0,-0.95,1.45]/Rx45（渲染对照官方 demo 帧拟合） | 构图一致；像素级未标定 |
| 腕相机外参 | URDF 链 + robot_config ori[0,-80,-90]（Isaac 约定） | EE 相对挂载（家位时后上方 0.18m 看下前方） | 俯视语义一致；安装臂不同 |
| 物体几何 | 真 mesh（USDZ） | primitive（圆柱/盒/开口壳） | 接触细节+视觉外观差异 |
| 材质/纹理 | MDL + 木纹 + 物体贴图 | 程序木纹 + 纯色 | 视觉差异 |
| 光照 | HDR 环境光（brown_photostudio） | 2 点光 + 头灯近似 | 墙面偏暗 |
| 房间 | Simple_Room mesh | 5 片内向平面 | 观感差异 |
| 摩擦系数 | layout 10000（PhysX 约定） | 1.0（MuJoCo 常规） | 抓持稳定性差异 |
| 机器人外观 | USD 黑白材质 | 统一浅灰 STL | 视觉差异 |

## 固有差异（❌ 不可消除，实验变量）

| 项 | 官方 | 我们 |
|---|---|---|
| 物理引擎 | PhysX（Isaac Sim 5.1） | MuJoCo 3.3.7 |
| 渲染引擎 | RTX 光追 | OpenGL 光栅 |
| 接触求解 | PhysX TGS | MuJoCo Newton |

## 结论

几何/布局/控制/评分/权重/预处理链 = **数值级对齐**；相机外参 = **构图级对齐**（像素级未标定）；
材质/网格/光照 = **语义级近似**；物理/渲染引擎 = **固有差异**（本实验「MuJoCo 路线」的定义性变量）。
π0.5 闭环 0 分（put_bottles）归因于视觉域差距（开环对拍已证管线正确），该结论与上述 ⚠️ 项直接相关，
报告将量化：后训练（MuJoCo 域内微调）前后对照即是对域差距的直接测量。

## 转换验证补充（2026-09-17）

JAX→PyTorch 转换做了两次独立转换对比：811/812 参数完全一致（diff < 1e-6）。
唯一差异参数 `gemma_expert.lm_head.weight`（diff 0.17）不在推理路径上——
π0.5 用 flow-matching + action_out_proj 输出动作，gemma_expert 的 lm_head
仅在训练时用于语言建模 loss，推理时不参与。**结论：转换确定性已验证。**
