# 原作者协议对齐进度（2026-09-17）

## 状态

最新进展：按用户要求切换 Windows 原生功能联调，暂不恢复三局。WSL运行器 `author_hybrid_20260917_215627` 的18步预检通过，但正式第1局会话 `9aac3c58f825` 在第0步等待GPT超过600秒，于22:10超时退出，已结束仿真会话并保留记录。下文保留先前验证记录。

## Windows 原生功能联调

- 入口：在 `simulation/mujoco` 下运行 `.venv/Scripts/python.exe run_author_hybrid.py --backend windows --trials 1 --reasoning-effort medium`，先18步预检再单局700步上限。用户先批准high，后明确改medium并要求先诊断连接；原作者xhigh默认保留，报告和调用记录标注实际强度。
- 完整工具验证入口：`.venv/Scripts/python.exe verify_author_tools.py --backend windows`，无仿真动作。
- Windows使用本机Codex、PowerShell、独立NumPy/Pillow环境；仿真/渲染本来就在Windows，π0.5仍通过现有隧道调用远程NPU。
- 持续GPT会话、动态动作段、审核/纠偏条件及数据记录不变。不是预写脚本代替GPT。
- 本次是功能测试，**不宣称读隔离或官方基准等价**。供应证据不可改、不得访问凭证/隐藏真值等为策略约束，图像路径检查为事后审核。
- 原生CLI旧 `-s workspace-write` 参数实测会话仍只读；改为官方 `default_permissions=":workspace"` 后，会话日志确认工作区可写，但执行命令仍被沙箱策略拒绝。按用户“不隔离、先Windows跑通”的要求，本次Windows分析子进程使用 `default_permissions=":danger-full-access"`，不是全局配置修改，不绕过既有exec规则。参考：https://learn.chatgpt.com/docs/permissions 。这是非隔离的功能联调，不能作为隔离环境验收通过。
- 新增Windows路径、后端选择、原生日志定位测试，总计24项相关测试通过；Python定向审查通过。真实完整工具与仿真结果另记运行目录。
- Windows真实完整工具验证 `author_tools_20260917_221553/report.json` 为passed：两轮同线程 `01a0afb9-650a-7683-86b8-7b09722e3b1a`，实际计算45、写/读NOTES.md、追加resumed_ok、裁图像素核对及两次图片返回验证均通过。此前只读/工作区沙箱失败记录保留，不计作通过。
- Windows xhigh预检 `author_hybrid_20260917_221803` 在0步等待审核数分钟；用户批准high后，写STOP并结束该GPT子进程，仿真会话 `8964ce4b835d` 正常清理，原记录保留。
- Windows high预检 `author_hybrid_20260917_222353`：首轮GPT请求出现TLS EOF、连接重置10054、WebSocket重连失败和HTTP回退重试，最终529.704秒返回student12；π+FK为2.046秒，执行12步11.39秒。用户要求medium并诊断网络后，在12步停止，完整记录保留。不是18步预检通过，更不是投瓶成功。

### 连接诊断（22:35）

- 子进程继承HTTP_PROXY/HTTPS_PROXY指向本机127.0.0.1:7897，监听进程为verge-mihomo。
- Clash应用日志22:27:15.515、22:27:25.751应用配置；服务日志同时间段大量chatgpt.com:443、ws.chatgpt.com:443出口连接超时，时间对应Codex TLS EOF/10054/WebSocket错误。可定位代理出口链路故障，但不据此断言具体DNS/节点根因，也不能解释此前所有推理等待。
- 诊断时两次通过现有代理的HTTP连通性探测均返回405（只验证端点可达，不代表鉴权推理通过），耗时6.40秒/3.31秒；绕过显式HTTP代理为1.06秒，但可能仍走系统TUN，不作为真正直连结论。
- 实际本机Codex GPT-6 medium无工具诊断请求成功返回CONNECTION_OK，turn.completed；线程01a0afca-37d1-70b1-b845-47308e2777c8。说明诊断时鉴权和短推理已恢复，不保证长连接持续稳定。
- 未修改代理、DNS或全局Codex配置。实验保持停止，下一次联调指定medium。

### 第0步停滞修复与恢复（22:53）

- `author_hybrid_20260917_223716` 的medium首轮600秒超时。进程存活，仿真可响应；原生会话只有初期推理事件、没有工具调用，未发出任何动作。不能把该等待算作机械臂执行慢。
- 同一完整输入（原图片、H50动作、FK轨迹、提示和schema）对照：禁用WebSocket改用HTTP的诊断耗时33.406秒，退出码0并出现turn.completed；原输入未缩减。支持原流式传输路径停滞的判断，不据此断言服务端或代理内部的具体故障。
- Windows后端使用自定义author_http，端点仍为https://chatgpt.com/backend-api/codex，使用现有OpenAI认证，supports_websockets=false；首次和resume一致。未改全局配置，未绕过审核及动作校验。
- `probe_teacher_transport.py` 是诊断用途：不直接发机器人动作，但保留原分析工作区和工具权限，非隔离重放；拒绝resume。诊断结果不作为真实动作回执。
- 25项相关测试及定向Python审查通过。
- 新运行 `author_hybrid_20260917_225148`：Windows/HTTP/GPT6-medium，18步预检会话60333183bb6b已完成，同线程01a0afda-4ab5-76b1-b352-a7e15c9fcf7a，student15+student3，两轮GPT32.125秒/38.360秒，动作16.922秒/4.109秒。随后自动进入第一局，完整报告和数据位于该运行目录。预检完成不等于投瓶成功。

旧固定3步实验 `gpt6_hybrid_20260917_211332` 停在第一局24步，后两局取消。数据和录像保留，不计入新协议成绩。

## 对齐对象

- 报告：https://anonymous-report-421.github.io/public-website/?lang=zh&view=1
- 代码：https://github.com/anonymous-report-421/GPT-as-Policy
- 核对的提交：`8f3d362b077d8efb77e2a7274d5b2c20e2243846`
- 对象是 RoboDojo 的 GPT-6 + π0.5 混合策略，不是 GPT Direct 或 RoboLab。

## 已实现

- 每次新观测产生50步π候选，GPT自选student 1–15步或edit/eef 1–5步；剩余候选丢弃。
- 结构化阶段进度、上一段执行结果、下一段意图；只有失败/意图不对齐才能纠偏。
- 每局持续GPT会话，gpt-6-astra/xhigh，无模型回退。
- 双臂link6机器人运动学轨迹，与实测夹爪中心交叉校验。
- 每个真实ACK后重算IK；纠偏最终关节增量≤0.05rad、末端增量≤2cm/0.1rad，含最终FK回退校验。
- 纯零偏移keep保留该臂原π动作，不冻结另一臂。
- 原始数据留在审计目录；模型观测使用白名单，不传评分、抓取标签或物体位置。
- STOP、异常结束、本局会话回收和HTML报告。

## 工具隔离

Windows原生权限测试未达标：测试文件外读和仿真端口连接未被阻止。因此不使用该方案。

改用现有 Ubuntu-22.04 WSL，独立安装 Codex 0.154.0 与 NumPy/Pillow分析环境。未覆盖Windows或WSL登录凭证。

实际Linux权限测试通过：

- 允许读取证据，禁止修改和删除证据；
- 禁止读取目录外已存在文件；
- 禁止网络连接与监听；
- 允许分析目录内计算、裁图、笔记写入。

`author_sandbox.py` 为命令配置限制。图像路径额外做WSL规范路径审计，拒绝目录外路径和符号链接逃逸；该审计是事后决策有效性检查，不宣称是所有原生工具的操作系统级读隔离。外部MCP/浏览器/联网工具不属于策略权限。

## 实际验证与当前阻塞

- 18项轻量协议/运行器/旧接口测试通过（尚需后续变更后再跑）。
- 初版18步模拟短测通过，三次决策共享同一会话：student12、eef3、eef3。它发生在最终WSL工具环境之前，不作为最终验收。
- `author_hybrid_20260917_212838` 是人工中断的中间预检，不计成绩。
- 最终完整工具的离线真实GPT测试：`simulation/mujoco/runs/author_tools_20260917_213927/report.json`，失败。原因是WSL旧ChatGPT凭证无法刷新，返回401。尚未验证WSL中真实GPT的裁图、笔记及跨轮工具能力。
- 需要用户重新登录WSL中的专用Codex，然后依次运行 `verify_author_tools.py`、`run_author_hybrid.py --pilot-only`；全部通过后才可启动正式三局。

## 尚存环境差异

MuJoCo物理/渲染与原作者Isaac Sim不同；昇腾转换权重推理与OpenPI/JAX不同；本地夹爪状态是实测归一化位置，原作者为控制命令。三局同布局不是原作者十任务五案例完整基准，不能直接比较62.60分或48%成功率。

## 登录恢复后的完整工具验证

- `author_tools_20260917_215001`：两轮相同GPT会话，真实Python/NumPy计算45，写入并跨轮读取NOTES.md，裁剪输入图像中央320×240区域，两轮均实际调用view_image并收到图片。
- 最初检查器因Codex简化JSONL未输出code-mode内层view_image事件而失败。没有将模型自述当成证据。
- 改为从这一个策略会话的原始日志提取工具调用ID、路径、轮次和图片返回标志，不导出私有推理或完整原始会话。
- `verified_tool_evidence.json`补充验证通过，原失败report.json保留。裁图逐像素核对输入图像区域，笔记内容及相同会话ID均验证。
- 拒绝多次看图混在同一exec、无法解析的看图路径、没有对应调用的图片返回；每次实际控制前检查图片路径。该检查仍是事后有效性审计，不是原生看图工具的预读取安全边界。
- 新增日志回归测试后21项相关测试通过，Python审查通过。
- 最终WSL仿真预检会话 `3d75edd66516`：student12、eef3、eef3，共18步；同一GPT线程持续决策，三次审核约61.75/64.67/62.42秒，正常结束并编码录像。正式第1局使用独立的新会话和GPT上下文。
