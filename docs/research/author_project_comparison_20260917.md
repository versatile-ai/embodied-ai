# 当前实现与作者公开代码对照（2026-09-17）

## 范围与结论

本次只读比较，不修改、停止或重启正在运行的实验。

- 作者仓库：https://github.com/anonymous-report-421/GPT-as-Policy 。GitHub API核实main仍为 `8f3d362b077d8efb77e2a7274d5b2c20e2243846`，提交时间2026-09-16 01:42:39 UTC，消息Initial public release；不是又出现了新的更新。
- 本地HEAD：`5ee1f789d2d7f1f9cd7fdf1b4cfaefb3b254c630`，同时包含尚未提交的author_protocol/run_author_hybrid/author_windows等。本次比较的是当前工作树，而不是只比较已提交版本。
- 当前运行记录：`simulation/mujoco/runs/author_hybrid_20260917_225148/report.json`，Windows、HTTP、GPT-6 medium、固定layout0、单局700步上限；运行服务grasp_assist=false。
- **结论：核心分段审核思路已移植，但不是作者实现的等价复现。** 当前是自写MuJoCo适配器，底层控制、观测、运行器和评测条件仍有实质差异。

## 主要差异

| 维度 | 作者源码 | 当前运行实现 | 影响 |
|---|---|---|---|
| 物理/视觉 | Isaac Sim 5.1、原生RoboDojo资产及环境 | MuJoCo、1000Hz物理/25Hz控制，碰撞壳和材质/相机存在近似 | 接触、抓取和视觉分布不等价；不是换操作系统这么简单 |
| π0.5 | OpenPI/JAX，SOURCE钉住RoboDojo/OpenPI版本、seed0 step59999、配置/归一化器 | 文档描述JAX转PyTorch权重、昇腾NPU服务，通过HTTP调用 | 同一权重来源不等于同输出；本次没有重新校验远端权重hash、归一化和同噪声数值对拍 |
| GPT强度 | gpt-6-astra/xhigh | 用户批准的gpt-6-astra/medium | 是功能联调设置，不应称为作者相同推理条件 |
| 主运行器 | 持久Codex app-server，追加start/infer/execute动态工具；模型在完整agent内推进流程 | Python外层固定observe→infer→GPT最终JSON→act，每段codex exec/resume | 会话持续不等于整个agent编排一致；交互结构、消息角色和工具反馈不同 |
| 审核提示 | 原SKILL+gate_prompt+teacher_context，作为developerInstructions | author_protocol.py中的手工压缩/重写PROMPT，每段作为输入重新送入 | 保留结果/意图审核，但不能称为原提示逐字复用 |
| 动作段 | student 1–15，edit/eef 1–5，新H50候选 | 相同名义范围和H50，每段丢弃未执行后缀 | 核心审核频率已基本对齐，不是每个物理子步都审核 |
| student动作落地 | 原生控制、连续归一化opening | 服务额外限速/平滑，并以0.2/0.8滞回将夹爪目标二值化 | π规划轨迹不等于实际送给执行器的轨迹；详见控制专项记录 |
| proprio | 夹爪部分为控制命令状态 | state14以实测关节开度归一化 | π模型输入分布改变，接触/受力时尤其明显 |
| 纠偏算法 | edit渐进施加偏移，原生数值Jacobian DLS | 每步全偏移目标+本地迭代IK+额外最终FK回退 | 即使上限相同，运动行为仍不同；专项文档列源码行号 |
| GPT图像附件 | 默认最长边480预览；原分辨率保留用于工具，π图像不缩放 | 直接附本地640×480 PNG，未实现同样的teacher预览层 | token、可见细节及延迟有差异；不能把GPT预览缩放误当π预处理 |
| 工具与认证边界 | 运行专用CODEX_HOME/工作区权限，保留正常工具配置并追加仿真工具 | 本机登录+Windows非隔离子进程，web关闭，额外限制MCP/浏览器/外部读取的策略及事后审计 | 已真实验证shell/NumPy/Pillow/看图/笔记/resume，但不等于完整工具环境一致 |
| 网络/输入错误恢复 | 识别已结束的网络失败后同线程续跑；输入错误反馈给agent重填，不重置/不重放动作 | 当前CLI内部重试外加600秒超时；任何验证/请求异常导致当前局结束 | 作者具有我们尚未移植的应用层恢复状态机，不能用无限重试替代 |
| 评测范围/种子 | 十任务五案例，公开每个案例的eval/layout/reset/simulator/policy种子 | 当前单任务layout0；报告明确policy sampling not seeded | 不构成原50案例基准；未形成受控同期π-only基线 |
| 成绩 | 原生RoboDojo判分，public_results保留native_score等 | 自写MuJoCo投瓶计数和阶段评分 | 两套分数不能直接横比；应分开报告“本地环境成功”与“原生基准复现” |

## 对当前抓取困难的意义

### 已证实而非猜测

1. `harness/simsvc.py:735` 先将连续opening变成0/1滞回目标，再限速/低通后执行。默认手臂每tick执行器目标变化上限为0.028×0.85=0.0238rad；这不是实测关节速度上限，更不是关节一定能跟随该目标。
2. `state14()` 返回测量开度，而不是作者的命令状态。
3. `author_protocol.py` 名义保留student候选，但后面的仿真服务仍会改写执行目标。因此“Python审核器没有改动作”不能证明“student完全原样执行”。
4. 当前运行明确关闭grasp_assist。旧README中“默认开启辅助”的描述已过时，不能作为当前实验状态。

### 尚需测试的假设

动作平滑/执行器跟踪、夹爪语义、模型预处理与转换数值差异、视觉/接触域差异，都可能导致当前反复接近无进展；本次静态比较无法确定其中哪个是主要原因，不能直接归因于medium或GPT决策错误。

优先建议（本次未执行）：同一测量状态下比对原始π action、过滤后的data.ctrl目标和下一时刻qpos；随后比对命令/测量夹爪输入与连续输出；再做官方JAX与NPU的同图像、同状态、同采样噪声对拍。不要未经验证直接取消稳定性处理。

## 作者投瓶结果提供的参照

公开 `evaluation_cases.json` 中混合方法投瓶layout0–4均成功，步数485、397、436、410、534；纠偏步数分别0、5、0、0、0，步数上限均700。指令与当前任务一致。

这意味着作者环境下该任务主要依靠π0.5执行完成；它支持优先调查我们的基础执行/观测迁移，而不是不断加强GPT干预。**这不是当前MuJoCo环境也应必然成功的证明**，不能直接把5/5结果当成我们环境的预期成功率。

## 证据索引

上游链接固定到本次核实的提交：

- [README（依赖、评测及发布边界）](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/README.md)
- [SOURCE（代码版本、checkpoint身份）](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/SOURCE.json)
- [settings（模型、强度、480附件）](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/settings.py)
- [run（agent编排、权限、输入恢复、网络续跑）](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/skill/run.py)
- [network_recovery（识别与续跑条件）](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/skill/network_recovery.py)
- [image_preview（教师附件与π输入分离）](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/skill/image_preview.py)
- [codex_backend/config（运行专用认证和默认工具）](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/codex_backend/config.toml)
- [公开案例与种子](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/public_results/evaluation_cases.json)

本地对应：

- [审核运行器](D:/code/versatile-ai/embodied-ai/simulation/mujoco/run_author_hybrid.py:80)
- [提示与校验](D:/code/versatile-ai/embodied-ai/simulation/mujoco/author_protocol.py:78)
- [本地IK纠偏](D:/code/versatile-ai/embodied-ai/simulation/mujoco/author_protocol.py:258)
- [执行目标处理](D:/code/versatile-ai/embodied-ai/simulation/mujoco/harness/simsvc.py:735)
- [夹爪状态输入](D:/code/versatile-ai/embodied-ai/simulation/mujoco/harness/simsvc.py:649)
- [Windows工具/传输](D:/code/versatile-ai/embodied-ai/simulation/mujoco/author_windows.py:30)
- [转换说明](D:/code/versatile-ai/embodied-ai/docs/pi05_conversion_guide.md)
- [控制专项逐项比较](D:/code/versatile-ai/embodied-ai/docs/research/author_control_comparison_20260917.md)

注意：未读取远端服务当前进程或权重文件，模型部署细节来自仓库文档和本轮健康记录；未做新的模型推理、物理测试或参数调整。开源快照未包含全部仿真资产、模型权重和原始运营日志，所以仅靠该仓库不能完成数值等价认证。
