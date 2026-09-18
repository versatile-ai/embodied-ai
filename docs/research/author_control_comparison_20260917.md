# 作者控制链路与本地 MuJoCo 移植对比

核查日期：2026-09-17。范围：只读源码比较；没有启动模型、控制机器人、停止实验或修改运行代码。

作者仓库 main 经 GitHub API 核实为 `8f3d362b077d8efb77e2a7274d5b2c20e2243846`。下面链接均固定该提交。本地指当前工作树，包含未提交修改，不等同于 Git HEAD。

## 结论

目前是作者方法的 MuJoCo 适配，不是相同控制实现。分段审核与接管条件基本一致，但**连续夹爪动作被执行端二值化、夹爪观测语义改变、额外关节限速/滤波、edit 插值不同、IK 算法不同**。这些差异足以使同一 π0.5 proposal 在两端产生不同实行动作，不能把当前接近瓶子无进展直接归因于 GPT 审核策略。具体因果仍需记录跟踪误差或隔离对照验证。

## 已证实差异

| 项目 | 作者固定提交 | 本地当前工作树 | 判断 |
|---|---|---|---|
| 审核时序 | 每段先 fresh H50 π0.5 inference，审核上一段 outcome 与下一段 intent，再执行；不是每个物理 tick 调 GPT，也不是等整个子目标完成 | `run_author_hybrid.py` 每个外层循环重新 infer，Teacher 返回后执行选定前缀 | 方法框架一致 |
| 分段长度 | student 1–15；edit/eef 1–5，弃用剩余 proposal | schema + validate 同样限制 | 基本一致 |
| 接管门控 | failed 或 misaligned 方可接管；还硬检查 tick=0 iff not_started，各 evidence 字段非空 | failed/misaligned 硬检查存在；缺少上述 tick 状态等价检查，TEXT 无 minLength，仅 reason 非空 | 本地校验更松 |
| student 夹爪 | 连续 0=关、1=开，仅范围限制；native command 直接交给 env.take_action | `_step` 将 <=0.2 置0，>=0.8 置1，中间保持旧目标，之后执行滤波 | **真实语义不一致，不仅是提示词差异** |
| 夹爪 proprio | 最近控制器 opening command，不是实际指缝 | `state14()` 返回测量 joint7 再归一化 | π0.5 输入分布发生变化 |
| 低层跟踪 | hybrid wrapper 调用原生 RoboDojo `env.take_action`，并记录真实 ACK | 默认 arm target 每 tick ±0.028 rad 限幅，再乘0.85；夹爪物理目标±0.008m再乘0.85；40物理子步中32步插值 | 本地额外控制动态，不能声称 native 等价 |
| edit 偏移 | 前缀内 alpha=(i+1)/steps，将 position/rotation offset 逐步增至全量 | 每步直接加完整 offset，再由本地 IK 限界 | 同样 edit JSON 生成不同轨迹 |
| 纠偏 IK | 一次数值 Jacobian DLS，阻尼矩阵 0.05² I；task error 限2cm/0.1rad，最终 joints±0.05 | 最多80轮解析 Jacobian DLS；姿态权重0.7、阻尼0.01 I；迭代幅±0.025，最终±0.05，再FK回溯保证最终空间增量2cm/0.1rad | 界限名称相近，数值行为不同；非简单“本地阻尼更大所以更慢”可直接定论 |
| FK/EEF | URDF robot-only FK，environment-origin link6，两臂同一坐标轴，wxyz | MuJoCo robot-only model，*_ee site移至link6，包含link6到jaw的偏移校验 | 坐标契约基本对齐，但几何实现不同 |

### 上游固定源码依据

- 审核时序、动态前缀、工具/历史及夹爪契约：[skill/SKILL.md](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/skill/SKILL.md#L33)。
- native25Hz、link6、2cm/0.1rad/0.05rad、夹爪命令态：[eef_control.md](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/skill/context/eef_control.md#L14)。
- 审核结构硬校验与 tick 状态约束：[gate_assessment.py](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/gate_assessment.py#L33)。
- edit 插值 alpha：[action_edit_kinematics.py](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/action_edit_kinematics.py#L38)。
- keep 连续 opening 的恢复、纠偏每实际 ACK 重算：[client.py](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/client.py#L59)、[执行循环](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/client.py#L273)。
- 单次 DLS：[kinematics.py](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/kinematics.py#L42)。注意2cm/0.1rad限制施加于error向量，代码没有像本地一样验证最终FK增量；不能把文档上限当成上游代码已完全证明的后置条件。
- 原生动作透传与ACK：[session.py](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/session.py#L140)。

### 本地文件定位

- [夹爪观测测量态](D:/code/versatile-ai/embodied-ai/simulation/mujoco/harness/simsvc.py:649)。
- [限速、平滑与阈值默认值](D:/code/versatile-ai/embodied-ai/simulation/mujoco/harness/simsvc.py:47)。
- [真实执行路径的滞回、限速与插值](D:/code/versatile-ai/embodied-ai/simulation/mujoco/harness/simsvc.py:735)。默认最大 arm command 改变量 = 0.028×0.85 = **0.0238rad/tick**，按25Hz为0.595rad/s。这是执行器目标变化上限，不是实测关节速度；且环境变量可覆盖默认值。
- [edit直接全偏移](D:/code/versatile-ai/embodied-ai/simulation/mujoco/author_protocol.py:265)、[IK最终关节/笛卡尔回溯](D:/code/versatile-ai/embodied-ai/simulation/mujoco/author_protocol.py:302)。
- [本地IK迭代](D:/code/versatile-ai/embodied-ai/simulation/mujoco/harness/control.py:52)。
- [本地gate校验](D:/code/versatile-ai/embodied-ai/simulation/mujoco/author_protocol.py:166)。
- [执行器参数](D:/code/versatile-ai/embodied-ai/simulation/mujoco/assets/x5/dual_x5_scene.xml:151)：双臂 kp=100/kd=10；四夹爪 kp=1000/kd=50，另有夹爪关节 damping=5。仅报告本地值；未确认作者原生servo数值。

## 与“靠近瓶子却不前进”的关系：假设，不是已定根因

1. **轨迹跟踪滞后优先核查。** 本地限速/滤波作用于 student 与 correction 全部动作；FK proposal 显示的是原始关节目标，不是经过滤波后的执行器目标，也不是实际到达姿态。GPT 可能据此判断 π0.5 意图合理，但机器人实际到不了。应比较每tick raw proposal、command_ctrl、测量q和link6的误差；不要只看发送动作数组。
2. **夹爪状态分布偏移。** 作者 fine-tuned π0.5 接收命令态，本地接收测量态；接触、限速或受阻时两者可显著不同。可能影响接近→闭爪→抬升时序，但未对当前局做因果实验。
3. **纠偏的数值不同。** 本地迭代解后逐关节裁切/空间回溯，可能改变最终位移方向或缩小进度；原作者一次DLS本身也可能跟踪不足。不能只凭两者参数推断哪边必然更优。
4. **动作模态混淆。** 已明确 link6/jaw 0.12657m 偏移，不应直接断言当前还在用错坐标；仍需核查在线决策指定的目标是否按这个几何计算。

## 不确定性与边界

- 上游 GPT-as-Policy 仓库的混合控制层没有内嵌 RoboDojo 原生低层控制器。SOURCE.json 固定外部 RoboDojo commit `ee67a1468510da7624a089164402359f2afc72c8`。本次未取得那个依赖的 servo/physics 完整源码，故**不能断言原作者没有任何平滑/限速，也不能给出作者 physics Hz 或精确 kp/kd**；能确认的是 native25Hz 及 hybrid层直接调用 env.take_action。
- 作者文档说 zero/keep 保留 student trajectory；实际上游 `client.py` 对 edit 两臂仍都走 IK，未发现本地这种直接还原零编辑臂原始7维 action 的分支。因此“zero/keep逐关节完全原样”在上游应区别文档意图与执行实现；本地此处反而更强地保留原始动作。
- 本地 `student` 前缀原样传给 /act，不代表物理端原样执行；必须追踪到 `_step`。
- 本次仅静态调研，没有中途更改正在运行的实验；该轮结果应继续标记 Windows/MuJoCo 适配实验，而非官方同条件复现。
