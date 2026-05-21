# Spyder_2

Quadruped spider robot locomotion using PPO on NVIDIA Isaac Lab.

A 12-DOF spider robot trained with reinforcement learning for velocity-tracking locomotion, gait learning, and smooth walking policies.

## Features

- **12-DOF spider robot** — 4 legs x 3 joints (hip, thigh, calf), modeled in URDF/USD
- **200 parallel environments** for efficient training
- **PPO (RSL-RL & skrl)** with tuned hyperparameters
- **AMP support** via skrl
- **11 reward terms** — velocity tracking, trot gait, smoothness, joint regularization
- **Trot gait** — diagonal foot pairs (FL+RR, FR+RL)
- **Keyboard teleoperation** — WASD + QE velocity control
- **Checkpoint export** — PyTorch, TorchScript (JIT), ONNX
- **TensorBoard logging** for reward analysis

## Quick Start

```bash
# Install the package
python -m pip install -e source/Spyder_2

# List available tasks
python scripts/list_envs.py

# Train
python scripts/rsl_rl/train.py --task spider_3

# Play a trained policy
python scripts/rsl_rl/play.py --task spider_3
```

## Project Structure

```
source/Spyder_2/
  Spyder_2/
    assets/spider.py          # Robot articulation config
    tasks/direct/spyder_2/    # RL env, config, agent configs
      spyder_2_env.py         # Main environment (DirectRLEnv)
      spyder_2_env_cfg.py     # Env config (200 envs, 11 rewards)
      agents/                 # PPO hyperparams (RSL-RL + skrl)
    utils/keyboard_input.py   # Teleop input handler
URDF/                         # Robot description (URDF + USD)
scripts/
  rsl_rl/train.py             # RSL-RL training
  rsl_rl/play.py              # Play policy
  rsl_rl/play_teleop.py       # Teleoperation mode
  skrl/train.py               # skrl training
  skrl/play.py                # skrl play
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Task ID | `spider_3` |
| Experiment | `spider_velocity_control` |
| Action dim | 12 (joint targets) |
| Obs dim | 48 |
| Episode length | 20 s |
| Physics dt | 200 Hz |
| Action scale | 0.25 |
| Parallel envs | 200 |

## Rewards

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

## Teleoperation

```bash
python scripts/rsl_rl/play_teleop.py --task spider_3
```

| Key | Action |
|-----|--------|
| W/S | Forward/Back |
| A/D | Strafe left/right |
| Q/E | Turn left/right |
| SPACE | Stop |

```bash
# Fixed velocity mode
python scripts/rsl_rl/play.py --task spider_3 --fixed_velocity 0.5 0.0 0.0
```

## Development

```bash
# Pre-commit (ruff formatting)
pip install pre-commit
pre-commit run --all-files
```

## Acknowledgments

Built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab).
