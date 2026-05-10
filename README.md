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

## URDF 腿长（可选）

脚本 `[scripts/adjust_a1_leg_length.py](scripts/adjust_a1_leg_length.py)` 按计划调整 **关节轴向长度**（`*_lower_joint` / `*_toe_fixed` 的 `-Z` 偏移）与 **小腿/大腿碰撞盒**的第一维长度，**不改变**盒子的横向截面尺寸（后两维不变）。默认读仓库里的 `[robots/a1/a1_description/urdf/a1.urdf](robots/a1/a1_description/urdf/a1.urdf)`，**默认写出到当前目录** `./a1.urdf`（可用 `-o` 改路径）。

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
| `[train/](train/)`     | 训练侧配置与 motion 加载（`a1_cfg.py`、`motion_loader.py`、`gen_loco_env.py` 等） |
| `[infer/](infer/)`     | MuJoCo：`urdf_to_mjcf`、机器人与 PD、观测缓冲、`.pt` Actor、入口 `run.py`           |
| `[scripts/](scripts/)` | `adjust_a1_leg_length.py`：按腿、按段改 URDF 腿长                             |
| `[robots/](robots/)`   | A1 URDF 与 mesh                                                       |


