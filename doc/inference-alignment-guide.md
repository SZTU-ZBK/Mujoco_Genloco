# GenLoco → MuJoCo 推理迁移完全指南

> 目标：把在 PyBullet/Isaac Gym 里训练好的 GenLoco 策略，**原封不动**地放到 MuJoCo 上跑起来。  
> 核心原则：**观测对齐、动作对齐、物理对齐**。

---

## 1. 整体架构一览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           单条控制循环 (33 ms)                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ Reference│ →  │  Policy  │ →  │  Action  │ →  │   PD     │ →  │ MuJoCo │ │
│  │  Motion  │    │  (MLP)   │    │ Pipeline │    │  Torque  │    │  Step  │ │
│  │ (phase)  │    │          │    │          │    │          │    │ (33×)  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│       ↑                                                            ↓        │
│       └────────────── 观测 Buffer (15帧历史) ←──────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 文件职责速查

| 文件 | 职责 |
|------|------|
| `infer/run.py` | 主循环：初始化 + 控制周期 + Viewer |
| `infer/urdf_to_mjcf.py` | URDF → MJCF 转换器（无外部依赖） |
| `infer/mujoco_a1.py` | MuJoCo 模型封装：关节索引、读状态、发 PD |
| `infer/policy_mlp.py` | 从 `.pt` 加载 Actor MLP，前向推理 |
| `infer/genloco_obs.py` | 观测 Buffer：15 帧历史，严格对齐训练侧顺序 |
| `infer/genloco_motion.py` | JSON 动作片段加载、采样、SLERP 插值、Cycle Offset |
| `infer/genloco_a1_cfg.py` | A1 机器人常量：关节名、初始位姿、PD 参数、限幅 |
| `infer/action_filter.py` | Butterworth 低通滤波器（复现训练侧 `enable_action_filter`） |
| `infer/quat_np.py` | 四元数工具：xyzw ↔ wxyz、转 Euler RPY |

---

## 2. URDF → MJCF：把机器人模型喂给 MuJoCo

### 2.1 为什么需要转换？

GenLoco 训练侧用 **PyBullet**，直接加载 URDF。  
MuJoCo **原生只认 MJCF**（XML 格式），因此需要把 A1 的 URDF 转成 MuJoCo 能理解的 XML。

### 2.2 转换流程

```
A1 URDF ──► parse links / joints / inertial / visual / collision ──► MJCF XML string
   │                                                                      │
   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐         │
   └──►│  link tree  │───►│ joint chain │───►│ geometry / mesh │─────────┘
       └─────────────┘    └─────────────┘    └─────────────────┘
```

**关键处理点：**

| URDF 元素 | MJCF 对应 | 说明 |
|-----------|-----------|------|
| `<link>` | `<body>` | 惯性、几何体挂在 body 下 |
| `<joint type="fixed">` | 无 joint 标签，直接嵌套 body | 固定关节只变父子关系 |
| `<joint type="revolute">` | `<joint type="hinge">` | 旋转关节 → hinge |
| `<origin rpy="...">` | `quat="w x y z"` | **RPY 转四元数**（wxyz 顺序） |
| `<inertial>` | `<inertial>` | mass + pos + `fullinertia`（6 个数） |
| `<visual>` / `<collision>` | `<geom class="vis">` / `<geom class="coll">` | 分离显示几何与碰撞几何 |
| — | `<freejoint/>` | 给 trunk 加 6-DOF 浮动关节 |
| — | `<motor>` | 每个 revolute 关节生成一个 torque motor |

### 2.3 坐标系对齐：RPY → Quat

URDF 的 `<origin rpy="rx ry rz"/>` 是 **欧拉角 xyz 顺序**（_extrinsic_，即先绕固定轴 x，再 y，再 z）。  
代码里先拼成旋转矩阵 `R = Rz(ry) * Ry(rp) * Rx(rr)`，再转成 **wxyz 四元数**写入 MJCF：

```python
# urdf_to_mjcf.py
R = _rpy_to_R(rx, ry, rz)      # Rz * Ry * Rx
wxyz = _R_to_quat_wxyz(R)      # wxyz 顺序
```

> MuJoCo 的 `quat` 属性就是 **wxyz**，这里不用翻转。

### 2.4 世界环境

转换器在 MJCF 里自动加了：
- **Checkerboard 地面**：`type="plane"`，摩擦力 `0.9 0.05 0.002`
- **天空盒 + 平行光**：让 Viewer 有基本视觉效果
- `integrator="implicitfast"`：MuJoCo 默认快速隐式积分器
- `timestep="0.001"`：1 ms 物理步长（与训练侧 `sim_step=0.001` 一致）

---

## 3. MuJoCo A1 封装：`mujoco_a1.py`

### 3.1 关节索引映射

MuJoCo 里找关节/执行器不靠名字字符串匹配，而是靠 **整数 ID**。  
初始化时遍历 `MOTOR_NAMES`（12 个），把每个关节的地址一次性查好：

```
┌────────────────────────────────────────────┐
│              MotorLayout                    │
│  ┌──────────┬──────────┬─────────────────┐ │
│  │ qpos_adr │  dof_adr │    ctrl_adr     │ │
│  │  (12,)   │   (12,)  │     (12,)       │ │
│  ├──────────┼──────────┼─────────────────┤ │
│  │ 关节在    │ 关节在   │ 执行器在 d.ctrl │ │
│  │ d.qpos 中│ d.qvel 中│   中的索引      │ │
│  │ 的位置索引│ 的速度索引│                │ │
│  └──────────┴──────────┴─────────────────┘ │
└────────────────────────────────────────────┘
```

```python
# 伪代码
for name in motor_names:
    jid  = mj_name2id(model, OBJ_JOINT, name)      # 查关节 ID
    qpos.append(model.jnt_qposadr[jid])            # 记录 qpos 地址
    dof.append(model.jnt_dofadr[jid])              # 记录 qvel 地址
    aid  = mj_name2id(model, OBJ_ACTUATOR, f"torque_{name}")
    ctrl.append(aid)                               # 记录执行器地址
```

> **为什么重要**：MuJoCo 的 `d.qpos` 里前 7 个数是 trunk 的 freejoint（3 位置 + 4 四元数），后面才是 12 个关节角度。地址不固定，必须运行时查表。

### 3.2 状态读取

| 量 | 来源 | 说明 |
|----|------|------|
| 关节角度 `q` | `d.qpos[qpos_adr]` | 12 个电机角度（rad） |
| 关节速度 `dq` | `d.qvel[dof_adr]` | 12 个电机角速度（rad/s） |
| 躯干四元数 | `d.qpos[free_adr+3 : free_adr+7]` | **wxyz** 顺序 |
| 躯干角速度（世界系）| `d.qvel[free_dof+3 : free_dof+6]` | 世界坐标系 |
| 躯干角速度（体坐标系）| `R_body^T @ omega_world` | **转到体坐标系**，与 IMU 一致 |

> GenLoco 训练侧的 IMU 读的是 **体坐标系角速度**。MuJoCo 的 `d.qvel` 给的是世界系，因此要做一次旋转：`omega_body = R.T @ omega_world`。

### 3.3 PD 控制

```python
tau = kp * (q_des - q) - kd * dq
```

- `kp` = `(100, 100, 100) × 4`  — 与训练侧 `ABDUCTION_P_GAIN=100, HIP_P_GAIN=100, KNEE_P_GAIN=100` 完全一致
- `kd` = `(1, 2, 2) × 4`  — 与训练侧 `ABDUCTION_D_GAIN=1, HIP_D_GAIN=2, KNEE_D_GAIN=2` 完全一致
- 输出钳位到执行器力矩上限（URDF 里 `limit effort`，A1 为 33.5 N·m）

### 3.4 子步进与动作插值

训练侧有两个关键开关：
- `num_action_repeat = 33`：一个策略步 = 33 个物理步
- `enable_action_interpolation = True`：在这 33 步内线性地从 **上一步目标** 插到 **当前步目标**

```python
# mujoco_a1.py  step_substeps()
for i in range(n):               # n = decimation = 33
    lerp = (i+1) / n
    q_interp = last_q_des + lerp * (q_des - last_q_des)
    apply_pd(q_interp)
    mj_step(model, data)
```

同时还有一个 **单步最大角度变化限制**：
```python
max_delta_per_step = 0.2       # 与训练侧 MAX_MOTOR_ANGLE_CHANGE_PER_STEP 一致
q_interp = clip(q_interp, current_q - 0.2, current_q + 0.2)
```

---

## 4. 策略网络加载：`policy_mlp.py`

### 4.1 从训练 checkpoint 提取 Actor

GenLoco 训练保存的 `.pt` 通常是一个字典，可能包含：
- `model_state_dict`（PPO / RL 库常用）
- `state_dict`
- `actor` / `critic` 分开的键

提取逻辑：
```python
checkpoint = torch.load("policy.pt")
state_dict = checkpoint["model_state_dict"]   # 或 fallback 到 checkpoint 本身
```

过滤规则：
- 排除带 `"critic"` 的键
- 只保留带 `"actor"` 或 `"policy"` 的 **2D weight** 张量
- 按层序号排序（如 `actor.0.weight`, `actor.2.weight`…）
- 每层配一个 bias，拼成 `nn.Linear + nn.ReLU` 的 `nn.Sequential`

```
checkpoint keys
    │
    ├─ actor.0.weight  ──┐
    ├─ actor.0.bias    ──┼──► Linear( in=451, out=512 ) + ReLU
    ├─ actor.2.weight  ──┤
    ├─ actor.2.bias    ──┼──► Linear( in=512, out=256 ) + ReLU
    ├─ actor.4.weight  ──┤
    ├─ actor.4.bias    ──┼──► Linear( in=256, out=12  )
    └─ critic.xxx      ──┘    (丢弃)
```

> 输入维度 **451** = 15×6 (IMU历史) + 15×12 (LastAction历史) + 15×12 (MotorAngle历史) + 1 (phase)。

### 4.2 推理

```python
obs = np.array([...], dtype=np.float32)   # (451,)
action = net(torch.tensor(obs)).cpu().numpy()  # (12,)
```

无 batch 维度时自动 `unsqueeze(0)` 再 `squeeze(0)`。

---

## 5. 观测构造：最刁钻的对齐环节

这是**最容易出错**的地方。训练侧用 `HistoricSensorWrapper` 把传感器历史堆成向量，顺序由 `sensor.get_name()` **字典序**决定。

### 5.1 训练侧传感器注册顺序

```python
# GenLoco/env_builder.py
sensors = [
    HistoricSensorWrapper(MotorAngleSensor, num_history=15),   # 名字排序："HistoricSensorWrapper(LastAction)" < "MotorAngle" < ...
    HistoricSensorWrapper(IMUSensor, num_history=15),
    HistoricSensorWrapper(LastActionSensor, num_history=15),
]
```

注意：`flatten_observations()` 按字典 **键的字母顺序**遍历。三个传感器的名字分别是：
1. `HistoricSensorWrapper(IMU)` → 以 `"H"` 开头
2. `HistoricSensorWrapper(LastAction)` → 以 `"H"` 开头，但括号内 `"I"` < `"L"` < `"M"`？  
   实际字符串比较：`"HistoricSensorWrapper(IMU)"` < `"HistoricSensorWrapper(LastAction)"` < `"HistoricSensorWrapper(MotorAngle)"`

**等等，不对！** 仔细看：
- `"HistoricSensorWrapper(IMU)"` — 括号后是 `I`
- `"HistoricSensorWrapper(LastAction)"` — 括号后是 `L`
- `"HistoricSensorWrapper(MotorAngle)"` — 括号后是 `M`

字母顺序：`I < L < M`，因此训练侧展平顺序是：
1. **IMU** history (15 × 6 = 90)
2. **LastAction** history (15 × 12 = 180)
3. **MotorAngle** history (15 × 12 = 180)

**但我们的代码里是：**

```python
# genloco_obs.py
flat = np.concatenate([
    np.concatenate(self._imu_hist),           # 90
    np.concatenate(self._last_action_hist),   # 180
    np.concatenate(self._motor_angle_hist),   # 180
])
```

这与训练侧 `flatten_observations` 的字母序 **完全一致**。

### 5.2 历史 Buffer 结构

```
┌─────────────────────────────────────────────────────────────────┐
│                      GenLocoObsBuffer                            │
│                                                                  │
│   每个传感器维护一个 deque(maxlen=15)，新数据 appendleft         │
│                                                                  │
│   IMU history (15 frames × 6 dims)                               │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │ [R, P, Y, dR, dP, dY]  ← 最新 (index 0)                │  │
│   │ [R, P, Y, dR, dP, dY]                                   │  │
│   │ ...                                                     │  │
│   │ [R, P, Y, dR, dP, dY]  ← 最旧 (index 14)               │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   LastAction history (15 × 12)                                   │
│   MotorAngle history (15 × 12)                                   │
│                                                                  │
│   展平后拼接：90 + 180 + 180 + 1(phase) = 451                    │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Reset 时的填充策略

训练侧 `on_reset` 会把当前观测 **复制 15 次** 填满历史：

```python
def reset(self, joint_pos, imu_rpy, imu_ang_vel_b, last_actions):
    for _ in range(15):
        self._imu_hist.appendleft(imu.copy())
        self._last_action_hist.appendleft(action.copy())
        self._motor_angle_hist.appendleft(motor.copy())
```

这样策略在第一步看到的不是空历史，而是 15 个相同的初始帧——**与训练时完全一致**。

---

## 6. 动作管道：从网络输出到电机力矩

### 6.1 全流程

```
Network Output:  a_raw  ∈ ℝ¹²  (范围大致 [-1, 1])
       │
       ▼  clip(action_clip=1.0)
   a_clipped ∈ [-1, 1]
       │
       ▼  set_last_actions(a_clipped)  ← 写入 obs buffer 的 "LastAction"
       │
       ▼  q_des = offset + scale × a_clipped
   q_des = init_angles + action_limit × a
       │
       ▼  ActionFilterButter (可选，默认启用)
   q_filtered = lowpass(q_des)
       │
       ▼  step_substeps(decimation=33, last_q_des, max_delta=0.2)
   33 个子步，每步线性插值 + 单步限幅 + PD 力矩
       │
       ▼  mj_step(model, data)
   物理推进
```

### 6.2 动作缩放公式

```python
offset = INIT_MOTOR_ANGLES = (0.0, 0.9, -1.8) × 4    # 12 个初始角度
scale  = ACTION_LIMIT     = (2.0,) × 4                # 每条腿 3 个关节都一样

q_des = offset + scale * action
```

| 关节 | offset | scale | 实际范围（action∈[-1,1]）|
|------|--------|-------|--------------------------|
| hip  | 0.0    | 2.0   | [-2.0, 2.0]              |
| thigh| 0.9    | 2.0   | [-1.1, 2.9]              |
| calf | -1.8   | 2.0   | [-3.8, 0.2]              |

> 这与训练侧的 `motor_commands = init_angles + action_limit * policy_output` 完全对应。

### 6.3 Butterworth 动作滤波

训练侧 `enable_action_filter=True` 时，用一个 **2 阶 Butterworth 低通** 平滑目标角度：

```python
# action_filter.py
# 设计参数：fc=4.0 Hz, fs≈30.3 Hz (1/0.033)
y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
```

系数硬编码，与 scipy.signal.butter(2, 4/15.1515) 输出一致。  
初始化时用 `init_history(offset)` 把滤波器状态填满初始角度，防止启动瞬态。

---

## 7. 参考运动（Reference Motion）

### 7.1 JSON 动作文件格式

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.01667,
  "EnableCycleOffsetPosition": true,
  "EnableCycleOffsetRotation": false,
  "Frames": [
    [x, y, z, qx, qy, qz, qw, j0, j1, ..., j11],
    ...
  ]
}
```

| 字段 | 含义 |
|------|------|
| `LoopMode=Wrap` | 循环播放；`Clamp` 则播完停住 |
| `FrameDuration` | 每帧时长（秒）|
| `EnableCycleOffsetPosition` | 每循环躯干位置累加 Δx, Δy（用于走/跑）|
| `EnableCycleOffsetRotation` | 每循环朝向累加 Δyaw |
| 每帧 19 个数 | 3 (pos) + 4 (quat xyzw) + 12 (joints) |

### 7.2 采样与插值

```
time ──► phase = time / duration  ──►  frame_index = phase * (N-1)
                                           │
                                    ┌──────┴──────┐
                                    ▼             ▼
                                  idx0          idx1
                                 frame0        frame1
                                    │             │
                                    └──────┬──────┘
                                           ▼
                                    blend = frac(index)
                                           │
                              ┌────────────┼────────────┐
                              ▼            ▼            ▼
                           root_pos    root_quat     joints
                           线性插值    SLERP 球面    线性插值
```

- **位置/关节/速度**：线性插值 `v0 + blend*(v1-v0)`
- **姿态四元数**：**SLERP**（球面线性插值），避免万向节锁和归一化漂移
- **速度**：预先通过相邻帧差分计算，再同样插值

### 7.3 Cycle Offset（循环漂移补偿）

对于周期性运动（如 pace），每播完一轮，躯干会相对起点有偏移：

```
Cycle 0: 起点 (0,0) ──► 终点 (0.3, 0)   Δpos = (0.3, 0, 0)
Cycle 1: 起点 (0.3,0) ──► 终点 (0.6, 0)  自动加上 1×Δpos
Cycle 2: 起点 (0.6,0) ──► 终点 (0.9, 0)  自动加上 2×Δpos
```

这样参考运动可以无限延伸，而不是原地踏步。

---

## 8. Viewer 与 Ghost 可视化

启动 `--viewer` 时，除了主机器人，还维护一个 **透明 Ghost 模型**：

```
主模型 (robot)          Ghost 模型 (ghost)
   │                         │
   │  物理仿真               │  纯运动学（mj_forward）
   ▼                         ▼
  真实姿态              参考运动姿态
   │                         │
   └────────┬────────────────┘
            ▼
       在同一个 Viewer 中绘制
       Ghost = 半透明绿球（躯干）+ 蓝球（足端）
```

Ghost 不跑物理，只把参考运动的 `root_pos + root_quat + joints` 写入 `ghost_data.qpos`，调 `mj_forward` 更新正运动学，然后在 `user_scn` 里画几个半透明球标记关键 body 位置。

**Reset 检测**：如果 `data.time` 倒退（说明调了 `mj_resetData`），主循环自动重置 obs buffer、action filter 和步数计数器。

---

## 9. 运行命令

```bash
# 1. 激活环境
source venv/bin/activate

# 2. 无头模式（只打印频率）
python -m infer.run \
  --checkpoint checkpoints/policy.pt \
  --motion motion/a1_pace.txt \
  --decimation 33 \
  --timestep 0.001 \
  --action_clip 1.0

# 3. 带可视化
python -m infer.run \
  --checkpoint checkpoints/policy.pt \
  --motion motion/a1_pace.txt \
  --viewer

# 4. 换 URDF（如改腿长的版本）
python -m infer.run \
  --checkpoint checkpoints/policy.pt \
  --motion motion/a1_pace.txt \
  --urdf robots/a1/my_custom_a1.urdf
```

### 关键超参数

| 参数 | 默认值 | 训练侧对应 | 说明 |
|------|--------|-----------|------|
| `--decimation` | 33 | `num_action_repeat=33` | 策略频率 = 1/(33×0.001) ≈ 30.3 Hz |
| `--timestep` | 0.001 | `sim_step=0.001` | MuJoCo 物理步长 |
| `--action_clip` | 1.0 | 默认 ±1.0 | 策略输出裁剪范围 |
| `--device` | `cpu` | — | `cuda` 也可，网络很小通常 CPU 更快 |

---

## 10. 对齐检查清单

如果你要迁移自己的策略，按这个清单逐项核对：

- [ ] **URDF 关节名**与 `MOTOR_NAMES` 完全一致（大小写敏感）
- [ ] **初始角度** `INIT_MOTOR_ANGLES` 与训练一致
- [ ] **PD 增益** `kp/kd` 与训练一致（100/1 for abduction, 100/2 for hip/knee）
- [ ] **动作缩放** `offset + scale * action` 公式与训练一致
- [ ] **动作裁剪** `clip` 值与训练一致
- [ ] **观测顺序** IMU → LastAction → MotorAngle → phase（按名字字母序）
- [ ] **历史长度** 15 帧
- [ ] **历史方向** 最新帧在前（`appendleft`）
- [ ] **IMU 内容** [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]（体坐标系）
- [ ] **角速度坐标系** 从世界系转到体坐标系（`R.T @ omega`）
- [ ] **四元数顺序** MuJoCo 用 wxyz；JSON motion 用 xyzw；转换时注意
- [ ] **子步插值** `enable_action_interpolation` 对应线性插值 33 步
- [ ] **单步限幅** `max_delta_per_step=0.2` 与 `MAX_MOTOR_ANGLE_CHANGE_PER_STEP` 一致
- [ ] **动作滤波** `ActionFilterButter` 与 `enable_action_filter` 一致
- [ ] **Phase 计算** `time / duration`，Wrap 模式取小数部分

---

## 11. 常见坑

### 11.1 观测顺序错一位

如果策略输出动作但机器人抽搐/倒地，**90% 是观测顺序不对**。  
用 `print(obs.shape)` 确认是 `(451,)`，再切片检查：
- `obs[0:6]` 应该接近当前 RPY + gyro
- `obs[90:102]` 应该接近上一步的 action
- `obs[270:282]` 应该接近当前关节角度

### 11.2 四元数顺序混了 xyzw / wxyz

- MuJoCo `d.qpos[3:7]`：**wxyz**
- 我们的 `quat_np.py` 和 motion JSON：**xyzw**
- 代码里凡是和 MuJoCo 交互都用 wxyz；策略输入、motion 文件用 xyzw

### 11.3 角速度坐标系错了

如果训练侧 IMU 是体坐标系，但推理侧给了世界系角速度，策略会把躯干旋转误判，导致原地打转。

### 11.4 PD 增益单位不一致

MuJoCo 的 `motor` 执行器是 **力矩模式**（`gear=1`，输出直接是 N·m）。  
确保 `kp/kd` 数值与训练侧 PyBullet  PD 一致即可，单位天然对齐。

### 11.5 时间不对齐

训练侧 `env_time_step = 0.001 * 33 = 0.033 s`。  
推理侧 `step_dt = decimation * timestep = 33 * 0.001 = 0.033 s`。  
如果改了 timestep 但没改 decimation，策略频率会变，动作会失效。

---

## 12. 扩展：换机器人/换策略

| 需求 | 修改点 |
|------|--------|
| 换 URDF（改腿长）| `--urdf` + 确保 joint name 不变 |
| 换 motion | `--motion` 指向新的 JSON |
| 换 PD | 改 `genloco_a1_cfg.py` 的 `PD_STIFFNESS / PD_DAMPING` |
| 换观测历史长度 | 改 `genloco_obs.py` 的 `history_length`（需重训策略）|
| 换网络结构 | `policy_mlp.py` 自动解析层数，无需改代码 |
| 上真机 | 把 `MuJoCoA1` 替换成真实电机接口，保留 obs + action pipeline |

---

*文档版本：2025-05-14*  
*对应分支：`feat/inference-alignment`*
