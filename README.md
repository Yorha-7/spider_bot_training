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
- **11 reward terms** — velocity tracking, trot gait, smoothness, joint regularization
- **Trot gait** — diagonal foot pairs (FL+RR, FR+RL)
- **Keyboard teleoperation** — WASD + QE velocity control
- **Checkpoint export** — JIT and ONNX

## Repository Structure

```
spider_bot_training/
├── assets/
│   ├── gifs/             # demo recordings
│   ├── images/           # static media
│   ├── URDF/             # robot URDF + meshes
│   └── usd/              # USD robot description files
├── scripts/
│   ├── rsl_rl/           # train.py, play.py, play_teleop.py, cli_args.py
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
```

## Training

```bash
# RSL-RL
python scripts/rsl_rl/train.py --task spider_3 --num_envs 200 --headless

# skrl
python scripts/skrl/train.py --task spider_3 --num_envs 200 --headless

# Or use the interactive launcher
bash scripts/train.sh
```

### Resume Training

```bash
python scripts/rsl_rl/train.py --task spider_3 --resume \
    --load_run <run_dir> --checkpoint model_xxxx.pt
```

## Inference

```bash
python scripts/rsl_rl/play.py --task spider_3 --checkpoint <path/to/model.pt>

# Fixed velocity (X Y YAW)
python scripts/rsl_rl/play.py --task spider_3 --checkpoint <path> --fixed_velocity 0.5 0.0 0.0

# Or use the interactive launcher
bash scripts/play.sh
```

## Teleoperation

```bash
python scripts/rsl_rl/play_teleop.py --task spider_3
```

| Key | Action |
|-----|--------|
| W/S | Forward/Backward |
| A/D | Strafe Left/Right |
| Q/E | Turn Left/Right |
| SPACE | Stop |

## Reward Terms

| Term | Scale | Purpose |
|------|-------|---------|
| lin_vel | 5.0 | Track target velocity |
| yaw_rate | 1.0 | Track target turning |
| z_vel | -2.0 | Prevent bouncing |
| ang_vel | -0.02 | Penalize tilt |
| joint_torque | -1e-5 | Energy efficiency |
| joint_accel | -1e-7 | Smooth acceleration |
| action_rate | -0.01 | Smooth actions |
| flat_orientation | -1.5 | Keep upright |
| joint_activity | 0.1 | Encourage joint use |
| feet_air_time | 0.01 | Foot lift |
| alternating_gait | 0.2 | Trot coordination |

## Development

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
