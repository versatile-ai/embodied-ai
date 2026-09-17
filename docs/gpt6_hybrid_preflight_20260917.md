# GPT6 + π0.5 三局实验：预检记录

日期：2026-09-17（北京时间）。最新状态：**用户升级后 GPT6 文本、三图输入与结构化响应预检通过；正式实验仍为 0/3 局启动**。以下保留首次失败记录。

## 已批准的实验协议

- 三局独立初始化的 `put_bottles_into_dustbin_0`，每局最多 700 个控制步。
- π0.5 提案，GPT6 在每个动作段执行前审查，默认每段 3 步。
- GPT6 可批准、请求重规划或给出受限末端纠偏；纠偏计入控制步数。
- 保留当前接触触发 weld 抓取辅助，明确不是官方纯物理评测。
- 先短程验证再正式实验；异常停止，无模型替换或纯 π0.5 回退。
- 记录三路画面、状态、完整提案、GPT6 输入/输出、执行请求/响应、耗时及异常。
- 最终生成 HTML 报告；三局用于初步验证，不能据此宣称优于纯 π0.5。

## 本次检查结果

1. 本地仿真 `127.0.0.1:8763/health` 正常，版本 `astra-isolated-v4-grip-sync`，物理 1000 Hz，控制 25 Hz，grasp_assist=true。
2. 原 π0.5 隧道不在运行。恢复经跳板机到 192 服务器 8642 端口的本地转发后，健康检查返回 `{"ok":true,"device":"npu"}`。此项只证明健康端点可用，本次未执行策略推理。
3. 本地 Codex 已通过 ChatGPT 登录，CLI 版本 `0.152.0`。
4. 使用 `codex exec --ignore-user-config --ephemeral --skip-git-repo-check -s read-only -m gpt-6-astra -c model_reasoning_effort=low --json` 做无工具、无机器人动作的文本接入预检。
5. 请求未成功；服务返回 HTTP 400 / invalid_request_error：

   > The 'gpt-6-astra' model requires a newer version of Codex. Please upgrade to the latest app or CLI and try again.

6. 进程退出码 1。没有成功的 GPT6 决策，不把请求中的模型名称当作成功调用证据。
7. 仿真仍停留在先前已结束的接口测试会话 `9bc54991471b`（t=25，done=true），没有新建正式实验会话。

## 首次预检后的待解决事项（历史）

请求用户批准升级本机 Codex CLI，或提供已配置且可验证的 GPT6 调用通道。升级后仍须重新验证文本、图像及结构化决策，不能假定升级必然解决问题。当前未实现或启动混合实验运行器，未生成成绩报告。

后续记录应区分请求模型、服务可验证的模型元数据和不可取得的快照信息；保存短程验证与正式实验的独立目录。记录阶段时间应至少区分观测、π0.5 推理、GPT6 决策、动作执行、视频编码。中断或环境错误不计为策略失败。

参考：Codex 非交互执行官方文档 https://learn.chatgpt.com/docs/non-interactive-mode 。实际账户接入结果以上述本地预检为准。

## 用户升级后的复测

- 实际 CLI 版本：`0.154.0`；请求模型：`gpt-6-astra`，reasoning effort=`low`，ChatGPT 登录通道，无模型回退。
- 文本预检成功返回 `PREFLIGHT_OK`，进程退出码 0；输入 token 23762，输出 token 8。
- 视觉预检附带已结束会话 `9bc54991471b` 第 25 步的主相机、左腕、右腕三张原始 PNG。使用 JSON Schema 要求 `images_visible`、`scene_description`、`decision`、`reason`。
- 图像与结构化响应成功，退出码 0；返回 `images_visible=true`、具体场景描述和 `decision=hold`。输入 token 25015，输出 token 141。没有提交任何仿真动作。
- 原始结构化响应：`simulation/mujoco/runs/gpt6_preflight_20260917/vision_decision.json`；Schema 在同目录 `decision.schema.json`。运行事件未另存完整流，以上为本次工具输出汇总。
- 图像命令首次因可变数量 `-i` 参数吞掉末尾提示而在本地报错，未发模型请求；添加 `--` 分隔后成功。
- 仿真健康检查正常，π0.5 健康检查返回 NPU 正常。本次未做 π0.5 推理或机器人短程控制。
- 此结果确认指定模型调用与多模态结构化输入输出可用，不代表混合控制器已实现或正式实验已执行；服务未提供不可变模型快照标识。后续仍需接入日志化闭环、测试，然后进行独立短程验证和三局正式实验。
