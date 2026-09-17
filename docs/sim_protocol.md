# 仿真服务 ↔ Mac harness 段边界协议（v0.1）

角色：155 跑两个无状态服务（仿真服务 `simsvc`、π0.5 推理服务 `pi05svc`，本机回环互联）；
Mac 跑评测 harness（大脑决策、IK 钳制、统计），持有全部 API key。

## 会话生命周期

```
Mac                          155 simsvc                    155 pi05svc
 | POST /session {task_id, seed, layout_yaml, arms}        |
 |--------------------------------------------------------->|
 |<-- {session_id, n_steps_budget}                          |
 | 循环（段边界）：                                           |
 | GET /session/{id}/observe  → {images[3] (JPEG b64), state(16+2 gripper), t, done, score}
 |      （simsvc 内部调 pi05svc: POST /chunk {images, state, task} → chunk 50x16）
 |<-- observe + candidate_chunk (50x16 joints) + ee_traj (50x2x3) |
 | Mac: 组装 prompt → qwen 决策 → IK+钳制 → action 段          |
 | POST /session/{id}/act {mode: follow|correct, joints: Kx14 或 ee_targets} |
 |--------------------------------------------------------->|
 |      simsvc 按 25Hz 步进 K 步（物理 250Hz 子步），段末暂停    |
 | ...直到 done → GET /session/{id}/result {score, success, intervention_stats}
```

## 关键约定

- **控制频率**：物理 250Hz（timestep 0.004），控制 25Hz（每控制步 10 物理子步）。
  π0.5 chunk = 50 控制步。follow 段 K∈[1,15]，correct 段 K∈[1,5]（与原文一致）。
- **动作接口**：simsvc 接受 14 维关节目标（位置伺服，actuator ctrlrange 内），
  correct 模式下 Mac 已用 ik_clamp 把 EE 目标转成关节轨迹，simsvc 不做 IK。
- **观测 schema**（对齐 pi05_base）：`base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`
  各 224x224x3 uint8 + `observation.state` 32 维（16 关节位 + 16 零速占位，
  与 lerobot RoboDojo 适配一致——待规格书确认 XPolicyLab 的 state 拼法）。
- **评分**：simsvc 内移植 RoboDojo `task/*/get_score`（transition 模式部分分）
  + `run_reward` 二值成功；布局 YAML 用 RoboDojo config，seed 由 Mac 下发。
- **pending 语义**：段边界 simsvc 暂停物理（原文口径：物理时长不含模型等待）；
  墙钟另计，用于报告 token/秒成本分析。
- **干预统计**：simsvc 记录每控制步属于 follow 还是 correct → 干预率对标 14.4%。
- **传输**：HTTP JSON + base64 JPEG（每回合 <1MB）；Mac↔155 走 ssh 隧道或直连内网。

## 待定（等任务移植规格书）

- state 32 维的确切拼法（XPolicyLab Pi_05 适配器）；
- 每任务的相机精确位姿（RoboDojo env_cfg/Sensor 配置）；
- 布局随机化协议（3 seed 的 official 抽样方式）。

## v0.2 增补（据 robodojo_task_port_spec.md）

- **动作步展开**：1 个 25Hz 动作步 = 10 个 250Hz 物理步；前 8 步从当前关节位置线性插值到目标，后 2 步保持目标（position actuator）。simsvc 必须照此实现，不得直接阶跃给目标。
- **夹爪映射**：策略夹爪值 g∈[0,1] → joint7 = g×0.054 − 0.01；joint8 = joint7（mimic）。
- **动作语义**：14 维 = [左6关节, 左爪, 右6关节, 右爪] 绝对关节位置（RoboDojo XPolicyLab Pi_05 接口）。
- **cam_head**：pos [0,-0.41,1.308]，欧拉 [30,0,0]°（xyz），fovy 71.1°（Gemini 345Lg，640×480 渲染后 resize 224）。
- **腕相机**：挂 URDF 自带 camera link，本地位姿 pos [0,0,0.001]、欧拉 [0,-80,-90]°，fovy 62.2°（d435）。
  ⚠ Isaac 与 MuJoCo 相机朝向约定不同，需渲染对照校验一次（TODO）。
- **评测布局**：`Assets/Eval_Layout/RoboDojo/arx_x5/{0,1,2}/{task}_{i}.json` 预生成固化布局，
  运行期零随机化 → 下载 JSON 即可完全复刻初始条件（物体 pos/quat/质量/摩擦全在内）。
- **state 维度待决**：RoboDojo 接口 state=14 维；pi05_base 期望 32 维（preprocessor stats 32 槽）。
  zero-shot 用 base 时的 14→32 槽位映射需查 XPolicyLab/lerobot aloha 约定后定（open question）。
- **回合终止**：run_reward 全过 或 步数达 step_lim；score 按 get_score 状态机每步更新。
- **支援臂**：make_kong / imitate_sorting_sequence 需第三台 Franka 回放 Assets/Traj pkl（这两个任务移植排最后）。
