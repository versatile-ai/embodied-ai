# RoboDojo 四瓶入桶实验归档

本分支保存 `put_bottles_into_dustbin`、Layout 0 / seed 0 的 GPT-6 Astra、π0.5 和 GPT+π0.5 实验。请先读 [结果索引](docs/RESULTS.md)，再按 [复现指南](docs/REPRODUCE.md) 搭建环境。三份 HTML 报告中的录像、图表和报告引用的数据均使用仓库内相对路径；为使浏览器读取动态原始数据，请从仓库根目录启动 `python3 -m http.server 8778` 后经 `http://127.0.0.1:8778/` 打开。

- [GPT-6 直接控制报告](hybrid_rollout/robodojo/reports/gpt6_layout0_20260919/gpt6_test_report.html)
- [π0.5 单独控制报告](hybrid_rollout/robodojo/reports/pi05_layout0_20260919/pi05_test_report.html)
- [GPT-6 + π0.5 与三方案比较报告](hybrid_rollout/robodojo/reports/hybrid_gpt_pi_layout0_20260919/hybrid_gpt_pi_test_report.html)
- [九回合逐轮交互报告](docs/robodojo-decision-report-20260929.html)；[离线单文件版](docs/robodojo-decision-report-20260929-offline.html)内嵌九段录像
- [下一阶段一周验证计划](docs/embodied_low_cost_1week_workplan.html)，属于计划而非已完成实验

仓库结构：`hybrid_rollout/robodojo/` 是 RoboDojo 适配器、控制器、测试与实验记录；`npu/pi05_service.py` 是昇腾 π0.5 HTTP 推理入口；`docs/` 是阅读和复现指南；`scripts/verify_archive.py` 校验归档文件和结果口径。原分支中的 Mac MuJoCo 仿真与旧文档已经从当前分支移除，历史仍保留在 Git 旧提交中。

首次阅读可运行：

```bash
python3 scripts/verify_archive.py
```

此命令仅核对归档，不会启动模型或修改实验。原始 RoboDojo、OpenPI、模型权重与 Docker 镜像不在 Git 中，位置与依赖见复现指南。报告是历史观测，视觉复核与官方 native evaluator 的成功口径分别列出，不能混用。

逐轮报告可由 `python3 -m scripts.build_decision_report` 从归档数据重新生成；加 `--inline-videos` 可生成无需视频旁文件的离线单文件版。建议日常阅读 8 MB 的普通版，离线分享时用约 72 MB 的单文件版。

实验适配代码源自本地 `GPT-as-Policy` 工作树，包含当时未提交的运行修正；归档提交记录了本次快照，但不等于原实验时的精确源码提交。配置版本和已知复现边界见 [复现指南](docs/REPRODUCE.md)。
