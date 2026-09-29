# 实验结果与证据索引

可从[九回合逐轮交互报告](robodojo-decision-report-20260929.html)切换 GPT、π0.5 与混合路线，查看 357 轮决策的输入、输出、动作、耗时和录像片段；[离线单文件版](robodojo-decision-report-20260929-offline.html)内嵌九段 MP4。

任务是官方 RoboDojo 的 `put_bottles_into_dustbin`，Layout 0 / seed 0，双 ARX X5 机械臂，250 Hz 物理与 25 Hz 控制，控制上限 700 步。墙钟耗时包括模型等待、仿真执行、数据交换及未单独埋点的开销；录像按 25 fps 编码，因此 700 控制步约 28 秒视频，不能用视频长度推断墙钟耗时。

| 策略与记录 | 控制步 / 决策 | 墙钟 | 完成情况 | 入口 |
| --- | ---: | ---: | --- | --- |
| GPT-6 Astra，固定 5 步 | 700 / 140 | 5471.34 秒（91.19 分钟） | 视频人工复核四瓶入桶；native success=false、分数 0.40 | [报告](../hybrid_rollout/robodojo/reports/gpt6_layout0_20260919/gpt6_test_report.html)，[原始结果](../hybrid_rollout/robodojo/reports/gpt6_layout0_20260919/run_20260919_120207/result.json) |
| GPT-6 Astra，固定 50 步 | 700 / 14 | 1142.14 秒（19.04 分钟） | 视觉完成 1/4；native success=false | [原始结果](../hybrid_rollout/robodojo/reports/gpt6_layout0_h50_20260919/run_20260919_235529/result.json) |
| GPT-6 Astra，自适应 5–50 步 | 700 / 64 | 2583.58 秒（43.06 分钟） | 视觉完成 2/4；native success=false | [原始结果](../hybrid_rollout/robodojo/reports/gpt6_layout0_adaptive50_v3_20260920/run_20260920_083346/result.json) |
| π0.5 单独，四轮 | 各 700 / 14 | 约 2.88–4.18 分钟 | 人工复核分别为 2、3、3、4 瓶；这批运行未得到有效官方奖励 | [报告和四轮录像](../hybrid_rollout/robodojo/reports/pi05_layout0_20260919/pi05_test_report.html) |
| GPT-6 + π0.5，首轮 | 502 / 34 | 1899.11 秒（31.65 分钟） | 视觉成功，native success=true | [结果](../hybrid_rollout/robodojo/reports/hybrid_gpt_pi_layout0_20260919/run_20260919_152000/result.json) |
| GPT-6 + π0.5，复测 | 700 / 49 | 2616.85 秒（43.61 分钟） | 视觉成功；native success=false、分数 0.40 | [综合报告与录像](../hybrid_rollout/robodojo/reports/hybrid_gpt_pi_layout0_20260919/hybrid_gpt_pi_test_report.html) |

“视觉成功”是回看整段录像得到的人工判断，不是官方通过。GPT 单独一轮和混合复测在视觉上完成、官方却未触发成功，仍需逐瓶几何条件和奖励项日志才能解释分歧。π 单独四轮的奖励注册无效，不能报为官方成功率 1/4。

## 性能口径

GPT 固定 5 步一轮总墙钟 5471.34 秒，其中记录为模型等待 4748.12 秒（86.8%）、仿真动作执行约 582.18 秒（10.6%），余下约 141 秒含启动与收尾。每轮模型等待平均 33.92 秒、p95 约 65.46 秒。首次可见事件延迟代理均值 26.10 秒，后续可见事件的输出间隔代理中位数 33.48 ms/token。两者都不是服务端真实 TTFT 或 TPOT：首次可见事件前还混有排队、网络、视觉编码和不可见内部处理，不能单独归因为 prefill。总输入约 20.74M token（其中缓存约 20.49M）、输出 29,038 token；输入上下文首轮约 20,823，末轮约 221,298。详情见 GPT 报告的“推理”和“性能”部分。

π 单独的最佳一轮墙钟 172.979 秒；14 次 HTTP 推理往返合计 42.197 秒，服务端 NPU 计算合计 7.696 秒（包含在 HTTP 时间内），14 段动作执行合计 129.508 秒。单次 NPU 计算均值 0.550 秒，但客户端往返均值 3.014 秒，瓶颈不能仅看芯片计算。仿真执行约占墙钟 74.9%。

混合首轮墙钟 1899.11 秒，GPT 审核累计约 753.74 秒、π HTTP 往返约 105.39 秒、动作执行约 316.22 秒。复测墙钟 2616.85 秒，对应约 1027.47、135.56、351.70 秒；尚有未逐项埋点的残差，不能把它全部算给网络或模型。π 的服务端推理时间包含于 HTTP 往返，严禁与 HTTP 时间重复相加。三方案并非相同动作预算和控制策略下的纯模型吞吐比较。

## 数据范围和缺口

HTML 中嵌入了关键 JSON，仓库同时保存 GPT / 混合运行的 `result.json`、`run.json`、`history.json`、请求与响应、token 记录、每轮执行 `NPZ`，以及四轮 π 摘要和完整多视角 MP4。为控制 Git 大小，未归档每步原始相机 `NPZ`（单轮可达数百 MB）、模型权重、Docker 镜像、官方 RoboDojo 资源及部分 GPU 主机侧完整控制步 JSON。混合复测附有不含逐帧 `NPZ` 的仿真轨迹包。数据清单和校验通过 `scripts/verify_archive.py` 查看。
