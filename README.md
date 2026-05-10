# mujoco_for_genloco

## 环境

```bash
conda env create -f environment.yml
conda activate genloco_mujoco
```

有 GPU 时在 [PyTorch 官网](https://pytorch.org) 选对应 CUDA 版本，在**已激活的环境中**重装 `torch` / `torchvision` / `torchaudio`（`environment.yml` 里默认 CPU 轮子是占位）。

## 运行推理

在项目根目录：

```bash
python -m infer.run --checkpoint /path/to/policy.pt --motion motion/a1_pace.txt
```

可视化：

```bash
python -m infer.run --checkpoint /path/to/policy.pt --motion motion/a1_pace.txt --viewer
```

**说明**：`checkpoint` 内需含名字里带 `actor` 或 `policy` 的线性层权重（与 RSL / Isaac 导出习惯一致）；否则需改 `[infer/policy_mlp.py](infer/policy_mlp.py)` 里的键过滤逻辑。

## 修改URDF 腿长（可选）

```bash
# 只看参数说明
python scripts/adjust_a1_leg_length.py --help

# 四条腿同时改大腿 + 小腿（默认 0.2 m；可分别指定）
python scripts/adjust_a1_leg_length.py --thigh 0.21 --calf 0.19 -o ./my_a1.urdf

# 只改某一侧小腿（shin），例如右前 FR
python scripts/adjust_a1_leg_length.py --legs FR --part calf --calf 0.22

# 多选腿：`--legs FR RL`（省略 `--legs` 表示四条腿全开）
python scripts/adjust_a1_leg_length.py --legs RR RL --part both --thigh 0.2 --calf 0.21
```

`--part`：`calf` 只动小腿链路，`thigh` 只动大腿，`both` 两者都改。视觉 mesh / 惯性未随长度缩放；大改腿长后若要物理一致需在 xacro / CAD 侧重做或重训策略。

## 目录


| 路径                     | 说明                                                                   |
| ---------------------- | -------------------------------------------------------------------- |
| `[infer/](infer/)`     | MuJoCo：`urdf_to_mjcf`、机器人与 PD、观测缓冲、`.pt` Actor、`run.py`；与训练对齐的常量见 `infer/genloco_a1_cfg.py`，参考运动加载见 `infer/genloco_motion.py` |
| `[scripts/](scripts/)` | `adjust_a1_leg_length.py`：按腿、按段改 URDF 腿长                             |
| `[robots/](robots/)`   | A1 URDF 与 mesh                                                       |


