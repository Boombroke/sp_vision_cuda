# HeroAimer 调参文档

针对英雄云台**惯量大、响应慢**的物理特性，HeroAimer 的高速分支不再"追"装甲板，而是**让云台 yaw 锁在车中心方向不动，等装甲板自己转过来撞进瞄准窗口再开火**。本文说明该分支的工作原理、参数含义、实测方法和故障排查。

## 一、设计思路

### 慢云台为什么不能"追"

英雄云台 yaw/pitch 的一阶响应时间常数（τ）通常在 100-300ms 量级。从下达 cmd 到云台真到位，目标已经位移了 `w·τ` 弧度——对 |w|=8rad/s 的小陀螺，这就是 80°-130°，差不多换了一块板。

如果像步兵那样追板，会出现：
- cmd 一直在变，云台永远跟不上
- 开火判定 `|ypr - cmd| < 阈值` 几乎不满足
- 偶尔满足是云台穿越的一瞬，不是稳态

### 慢云台的正确策略：**卡位 + 等待**

```
cmd 锁在车中心连线方位 (yaw 不动)
    └ pitch 跟"将正对那块板"的高度（缓慢变化）

板自己周期性转到瞄准方向
    └ 进入"位置窗口" + 法线朝你 + 云台稳态 → 开火
```

云台需要做的事变少（只调 pitch + 跟随车整体移动），就跟得动了。

## 二、双分支结构

```
HeroAimer::aim()
├─ |w| > center_mode_w_threshold  → 高速分支（瞄中心等板）
│   ├─ predict 时间链 = now + shoot_delay + fly_time + gimbal_tau
│   ├─ xyz0.xy = 车中心 - r 方向（不追板）
│   ├─ xyz0.z  = 命中时刻最先正对的那块板的 z
│   ├─ cmd 一阶低通（cmd_smooth_tau）
│   └─ 三道开火闸门：板位置 + 板法线 + 云台稳态
│
└─ |w| ≤ center_mode_w_threshold  → 普通分支（原 2 次迭代）
    ├─ predict → choose_aim_point → 解弹道 → 再 predict 一次校验
    └─ 单条云台对齐判定开火
```

## 三、参数清单（hero.yaml）

| 参数 | 默认 | 单位 | 含义 |
|---|---|---|---|
| `shoot_delay` | 0.05 | s | 发弹延时（cmd → 子弹离开枪口） |
| `max_fire_error_yaw` | 0.08 | rad | 单帧云台 yaw 误差上限（高速分支用作稳态判定） |
| `max_fire_error_pitch` | 0.08 | rad | 单帧云台 pitch 误差上限 |
| `max_shoot_angle` | 30 | degree | 普通分支板法线偏角上限 |
| `center_mode_w_threshold` | 8.0 | rad/s | **\|w\| 超过此值切换到高速分支** |
| `gimbal_yaw_tau` | 0.15 | s | yaw 云台一阶响应时间常数（**必须实测**） |
| `gimbal_pitch_tau` | 0.20 | s | pitch 云台一阶响应时间常数（**必须实测**） |
| `cmd_smooth_tau` | 0.30 | s | cmd 一阶低通时间常数（让 cmd 平缓，给云台跟随时间） |
| `fire_pos_window` | 0.05 | rad | 板位置进瞄准方向的角度窗口（≈3°） |
| `fire_facing_window` | 0.40 | rad | 板法线偏离视线最大角（≈23°，防侧倾跳弹） |
| `fire_steady_frames` | 5 | 帧 | 连续 N 帧云台稳态才允许开火 |

## 四、必做：实测 `gimbal_yaw_tau / gimbal_pitch_tau`

文档默认值是猜的。**不实测就用，云台预测会全错**，开火基本打不准。

### 步骤

1. 关掉自瞄，让英雄停下。
2. 打开 PlotJuggler，订阅 cmd 和 ypr 的 UDP plotter 信号。
3. 让云台 yaw 从静止跳到 10°：
   ```cpp
   // 在某个测试入口里发一次 cmd，或者用遥控器突变拨杆
   command.yaw = 0.0; command.pitch = 0.0;  // 起始稳定 1s
   command.yaw = 10 * CV_PI / 180;          // 跳变
   ```
4. 在 PlotJuggler 里测从 cmd 跳变那一刻到 ypr[0] 到达 9°（cmd 的 90%）所用的时间 `t_90`。
5. `tau = t_90 / 2.3`（一阶系统的标准换算）。
6. **pitch 单独测一遍**（重心偏置使 pitch 通常更慢）。

### 例：

| cmd 跳变 | t_90 实测 | tau 计算 |
|---|---|---|
| yaw 0° → 10° | 350 ms | 0.152 s |
| pitch 0° → 5° | 480 ms | 0.209 s |

填进 yaml：

```yaml
gimbal_yaw_tau: 0.152
gimbal_pitch_tau: 0.209
cmd_smooth_tau: 0.40   # ≈ 2 × max(0.152, 0.209)
```

### `cmd_smooth_tau` 的取值原则

- **过小**（< gimbal_tau）：cmd 抖得比云台快，云台跟不上 → 云台稳态判定永远不通过，开火不出去。
- **过大**（> 3×gimbal_tau）：cmd 自身滞后过头，板已经转走才发 cmd，pitch 高度不对 → 弹打偏。
- **推荐 1.5×~2×max(yaw_tau, pitch_tau)**。

## 五、调参流程（按顺序）

### Step 1：低速场景验证（不动高速分支参数）

让英雄打静止 / 缓慢平移目标，确认普通分支正常：
- 弹道解算正确（pitch 大致对得上）
- 开火节奏正常
- `command.shoot` 在云台对齐后能触发

如果这一步就有问题，先排查 `bullet_speed`、相机标定、`yaw_offset` / `pitch_offset`。

### Step 2：进入高速分支但**关闭开火**

```yaml
auto_fire: false   # 关掉开火，只看 cmd 行为
```

让英雄面对中等速度小陀螺（|w|=10-15 rad/s）：

观察 PlotJuggler：
- **cmd_yaw 应该锁在某个稳定值附近**，不像步兵那样跟着板抖
- **cmd_pitch 应该缓慢上下**（跟随将正对那块板的 z）
- **ypr[0] 和 cmd_yaw 之间有 gimbal_tau 量级的滞后**，应当能跟上

如果 cmd_yaw 抖动很大：
- 增大 `cmd_smooth_tau`（→ 0.5）
- 检查 EKF：高速时 `x[7]` 抖得厉害？把 Tracker NIS 阈值检查一下

### Step 3：放开开火，看触发频率

```yaml
auto_fire: true
```

正常应该看到（|w|=10-15 rad/s 普通车 4 板）：

- 每个板转到瞄准方向需要 `T = 2π/(armor_num·|w|) ≈ 0.10-0.16s`
- 实际开火频率受装填限制，1-2 发/秒是合理上限
- 日志应该有规律的 `########## fire ##########`

### Step 4：精细调整

打不出去（fire 触发率低）：
1. 放宽 `fire_pos_window`（0.05 → 0.08）—— 让更多板进窗口
2. 放宽 `max_fire_error_yaw/pitch`（0.08 → 0.12）—— 容忍云台抖动
3. 减少 `fire_steady_frames`（5 → 3）—— 缩短稳态判定窗口
4. 降低 `center_mode_w_threshold`（8 → 6）—— 让高速分支更早接管

打偏（fire 触发但弹丸没中）：
1. 重新实测 `gimbal_tau`（最常见原因）
2. 减小 `cmd_smooth_tau`（cmd 滞后过大导致 pitch 错位）
3. 缩小 `fire_pos_window`（板还没真到位置就开火）

侧倾跳弹（命中但弹偏弹）：
1. 缩小 `fire_facing_window`（0.40 → 0.30）

云台抖动严重：
1. 增大 `cmd_smooth_tau`
2. 检查 cboard PID 参数（`yaw_kp/kd, pitch_kp/kd`）

## 六、故障排查表

| 症状 | 可能原因 | 检查项 |
|---|---|---|
| `auto_fire: true` 但从不开火 | 云台稳态判定不过 | PlotJuggler 看 ypr-cmd 误差是否 > `max_fire_error_*` |
| | 板位置/法线判定不过 | 临时加日志打印 `pos_diff / facing_diff` |
| | 高速分支根本没进 | 确认 `\|w\|` > `center_mode_w_threshold` |
| 开火但持续打偏 | gimbal_tau 测错 | 重新做阶跃响应实验 |
| | cmd_smooth_tau 过大 | cmd 自己滞后了，把它减一半 |
| | fly_time 估算错 | 检查 `bullet_speed` 是否 hardcode 11.8（确实是） |
| cmd 抖动大 | EKF 在高速时 x[7] 不稳 | Tracker 调 nis_thresh、Target 调 v2 |
| | cmd_smooth_tau 过小 | 增大到 ≥ 2× gimbal_tau |
| 云台跟不上 cmd | gimbal_tau 偏小（实际更慢） | 重新实测 |
| | cboard PID kp 偏小 | 调底层云台 PID |

## 七、PlotJuggler 推荐订阅信号

```
cmd_yaw, cmd_pitch                  # HeroAimer 输出
ypr[0], ypr[1]                      # 云台实测
cmd_shoot                           # 开火方波（0/1）
debug_aim_point.xyza                # 命中时刻瞄准点
ekf_x[6], ekf_x[7]                  # 整车 yaw, w
```

绘图布局：
- 上：cmd_yaw vs ypr[0]，看滞后量
- 中：cmd_pitch vs ypr[1]
- 下：ekf_x[7] (车 w) vs cmd_shoot (开火方波)，看高速时是否有 fire

## 八、参数变更记录建议

每次实战调参后，记录：
- 哪一组参数（commit hash 或 yaml 备份）
- 测试场景（静止 / 平移 / 小陀螺 |w| 范围）
- 命中率 / 开火频率 / 失误模式

便于回滚和迭代。
