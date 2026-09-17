# RoboDojo 混合架构复现评测

本分支包含 RoboDojo 官方资料、π0.5 NPU 推理交付说明和 Mac 端 MuJoCo 双 X5 仿真环境。

## 目录

```text
embodied-ai/
├── simulation/mujoco/   # Mac 仿真服务、场景资产、布局、客户端和验收脚本
├── docs/                # RoboDojo 协议、任务规格、π0.5 部署/转换和对齐审计
└── README.md            # 项目入口
```

## 快速入口

- 仿真与端到端部署：[`simulation/mujoco/README.md`](simulation/mujoco/README.md)
- 官方任务与评分规格：[`docs/robodojo_task_port_spec.md`](docs/robodojo_task_port_spec.md)
- 环境对齐审计：[`docs/alignment_audit.md`](docs/alignment_audit.md)
- π0.5 同事部署指南：[`docs/pi05_colleague_guide.md`](docs/pi05_colleague_guide.md)
- 仿真服务协议：[`docs/sim_protocol.md`](docs/sim_protocol.md)

当前 MuJoCo 环境用于 Astra/π0.5 的控制协议、视觉输入、动作执行和本地回归；它不是 Isaac Sim/PhysX/RTX 的官方等价运行时。
