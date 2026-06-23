# Spider Bot Training


Multi-robot locomotion training using PPO on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab).

12-DOF spider robots trained with reinforcement learning for velocity-tracking locomotion and trot gait coordination.

| Spider (SG90) | Big Bertha (MG995) |
|:---:|:---:|
| <img src="assets/gifs/spider.gif" width="360"> | <img src="assets/gifs/big_bertha.gif" width="360"> |

## Features

- **12-DOF spider robot** — 4 legs × 3 joints (hip, thigh, calf), modeled in URDF/USD
- **200 parallel environments** for efficient training
- **PPO via RSL-RL and skrl** with tuned hyperparameters
- **AMP support** via skrl
- **12 reward terms** — velocity tracking, trot gait, smoothness, joint regularization
- **Trot gait** — diagonal foot pairs (FL+RR, FR+RL)
- **Keyboard teleoperation** — WASD + QE velocity control
- **Checkpoint export** — JIT and ONNX

## Repository Structure

```
spider_bot_training/
├── assets/
│   ├── gifs/             # demo recordings
│   ├── images/           # static media
│   ├── URDF/
│   │   ├── spider_rl/       # Spider SG90 URDF
│   │   └── big_bertha/      # Big Bertha MG995 URDF
│   └── usd/              # USD robot description files
├── scripts/
│   ├── rsl_rl/           # train.py, play.py, play_fixed_vel.py, play_teleop.py, cli_args.py
│   ├── skrl/             # train.py, play.py
│   ├── random_agent.py
│   ├── zero_agent.py
│   ├── list_envs.py
│   ├── train.sh          # interactive train launcher
│   └── play.sh           # interactive play launcher
└── source/
    ├── spider_rl/        # SG90 spider IsaacLab extension
    │   ├── config/extension.toml
    │   ├── setup.py
    │   └── spider_rl/
    │       ├── assets/spider.py        # robot articulation config
    │       ├── tasks/direct/spider/    # RL env + agent configs
    │       └── utils/keyboard_input.py
    └── big_bertha/       # MG995 big_bertha IsaacLab extension
        ├── config/extension.toml
        ├── setup.py
        └── big_bertha/
            ├── assets/big_bertha.py    # robot articulation config
            └── tasks/direct/big_bertha/ # RL env + agent configs
```

## Setup

```bash
# Install the extension
python -m pip install -e source/spider_rl
python -m pip install -e source/big_bertha
```

## URDF → USD Conversion

Convert URDF robot descriptions to USD using Isaac Sim's URDF Importer:

```bash
# Spider (SG90)
ISAACLAB=$(find ~ -type d -name "IsaacLab" 2>/dev/null | head -n 1)
SPIDER=$(find ~ -type d -name "spider_bot_training" 2>/dev/null | head -n 1)
python3 "$ISAACLAB/scripts/tools/convert_urdf.py" \
  --merge-joints \
  "$SPIDER/assets/URDF/spider_rl/spider_description.urdf" \
  "$SPIDER/assets/usd/spider_rl/spider.usd"

# Big Bertha (MG995)
ISAACLAB=$(find ~ -type d -name "IsaacLab" 2>/dev/null | head -n 1)
SPIDER=$(find ~ -type d -name "spider_bot_training" 2>/dev/null | head -n 1)
python3 "$ISAACLAB/scripts/tools/convert_urdf.py" \
  --merge-joints \
  "$SPIDER/assets/URDF/big_bertha/Spyder_mg995_description.urdf" \
  "$SPIDER/assets/usd/big_bertha/big_bertha.usd"
```

> USD files are tracked via Git LFS (see `.gitattributes`). After conversion, add the `.usd` file using `git lfs track`.

## Package Commands

| Package | Robot | Task ID | RL Engine | Train | Play |
|---------|-------|---------|-----------|-------|------|
| `spider_rl` | Spider (SG90) | `spider_3` | RSL-RL | `python scripts/rsl_rl/train.py --task spider_3 --num_envs 200 --headless` | `python scripts/rsl_rl/play.py --task spider_3 --checkpoint <path.pt>` |
| `spider_rl` | Spider (SG90) | `spider_3` | skrl | `python scripts/skrl/train.py --task spider_3 --num_envs 200 --headless` | `python scripts/skrl/play.py --task spider_3 --checkpoint <path.pt>` |
| `big_bertha` | Big Bertha (MG995) | `big_bertha` | RSL-RL | `python scripts/rsl_rl/train.py --task big_bertha --num_envs 10000 --max_iterations 1000 --headless` | `python scripts/rsl_rl/play.py --task big_bertha --checkpoint <path.pt>` |
| `big_bertha` | Big Bertha (MG995) | `big_bertha` | skrl | `python scripts/skrl/train.py --task big_bertha --num_envs 10000 --max_iterations 1000 --headless` | `python scripts/skrl/play.py --task big_bertha --checkpoint <path.pt>` |
| `big_bertha` | Big Bertha (MG995) | `big_bertha` | RSL-RL (fixed vel) | — | `python scripts/rsl_rl/play_fixed_vel.py --task big_bertha --checkpoint <path.pt> --vx 0.3 --omega 0.0` |

### Interactive Launchers

```bash
bash scripts/train.sh   # interactive train prompt
bash scripts/play.sh    # interactive play prompt
```

### Resume Training

```bash
python scripts/rsl_rl/train.py --task spider_3 --resume \
    --load_run <run_dir> --checkpoint model_xxxx.pt
```

### Fixed Velocity Inference

```bash
# Spider (SG90)
python scripts/rsl_rl/play.py --task spider_3 --checkpoint <path.pt> --fixed_velocity 0.5 0.0 0.0

# Big Bertha (MG995) — run at fixed forward/turning velocity
python scripts/rsl_rl/play_fixed_vel.py --task big_bertha --checkpoint <path.pt> --vx 0.3 --vy 0.0 --omega 0.0 --real-time
```

### Teleoperation

```bash
python scripts/rsl_rl/play_teleop.py --task spider_3
```

| Key | Action |
|-----|--------|
| W/S | Forward/Backward |
| A/D | Strafe Left/Right |
| Q/E | Turn Left/Right |
| SPACE | Stop |

### Notes

- **Spider (SG90):** 200 envs default, lightweight (4GB+ VRAM)
- **Big Bertha (MG995):** heavier model
  - 8GB+ VRAM: `--num_envs 10000 --max_iterations 1000`
  - 4–6GB VRAM: `--num_envs 1000 --max_iterations 3000`

## Reward Terms

### Spider (SG90) — `spider_rl`

| Term | Scale | Purpose |
|------|-------|---------|
| `lin_vel` | 1.5 | Track target velocity |
| `yaw_rate` | 0.5 | Track target turning |
| `z_vel` | -0.5 | Prevent bouncing |
| `ang_vel` | -0.02 | Penalize tilt |
| `joint_torque` | -1e-5 | Energy efficiency |
| `joint_accel` | -1e-7 | Smooth acceleration |
| `action_rate` | -0.01 | Smooth actions |
| `flat_orientation` | -1.5 | Keep upright |
| `joint_activity` | 0.1 | Encourage joint use |
| `feet_air_time` | 0.5 | Foot lift |
| `alternating_gait` | 2.0 | Trot coordination |
| `foot_dragging` | -1.0 | Penalize sliding feet |

### Big Bertha (MG995) — `big_bertha`

| Term | Scale | Purpose |
|------|-------|---------|
| `lin_vel` | 3.0 | Track target velocity |
| `z_vel` | -0.25 | Prevent bouncing |
| `ang_vel` | -0.02 | Penalize tilt |
| `joint_torque` | -1e-5 | Energy efficiency |
| `joint_accel` | -1e-7 | Smooth acceleration |
| `action_rate` | -0.005 | Smooth actions |
| `flat_orientation` | -1.5 | Keep upright |
| `joint_activity` | 0.3 | Encourage joint use |
| `feet_air_time` | 2.5 | Foot lift |
| `yaw_rate` | 3.0 | Track target turning |
| `alternating_gait` | 4.0 | Trot coordination |

### Physical Params
| Joint | lower limit | upper_limit | off | channel |
|-----|-----|-----|-----|-----|
| FL_hip_joint | 180 | 50 | 0 | 2 |
| FL_thigh_joint | 50 | 180 | +10 | 1 |
| FL_calf_joint | 150 | 0 | +2 | 0 |
|
| FR_hip_joint | 30 | 150 | 0 | 10 |
| FR_thigh_joint | 140 | 180 | 0 | 9 |
| FR_calf_joint | 40 | 180 | +8 | 8 |
| (Yet to be calculated) |
| RL_hip_joint | 0 | 0 | 0 | 0 |
| RL_thigh_joint | 0 | 0 | 0 | 0 |
| RL_calf_joint | 0 | 0 | 0 | 0 |
| RR_hip_joint | 0 | 0 | 0 | 0 |
| RR_thigh_joint | 0 | 0 | 0 | 0 |
| RR_calf_joint | 0 | 0 | 0 | 0 |


## Development

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
