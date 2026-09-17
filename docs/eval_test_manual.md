# 测试入口迁移说明

旧版 8663/8665、Qwen runner 及旧控制协议已经停用。当前唯一正式策略入口为 `simulation/mujoco/eval_control.py`。

完整部署和测试流程统一维护在 [仿真 README](../simulation/mujoco/README.md) 的“当前测试入口”章节，包含复位、独立 GPT 上下文、净化观测、π0.5 协议、动作校验和性能计时。
