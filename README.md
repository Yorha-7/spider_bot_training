# Spider Bot Training


Multi-robot locomotion training using PPO on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab).

12-DOF spider robots trained with reinforcement learning for velocity-tracking locomotion and trot gait coordination.

| Spider (SG90) | Big Bertha (MG995) v2.0.0 |
|:---:|:---:|
| <img src="assets/gifs/spider.gif" width="360"> | <img src="verification_artifacts/big_bertha_v2.0.0_seq.gif" width="360"> |

Big Bertha v2.0.0: forward, turn right 90°, forward, **reverse**, turn left 180°, stop.

### v2.0.0 measured performance

Clean-eval (domain randomisation pinned at distribution centres), 64 envs,
1000 steps after a 250-step settle.

| command | achieved | tracking |
|---|---|---|
| forward 0.30 m/s | +0.287 m/s | 96% |
| forward 0.12 m/s | +0.174 m/s | overshoots |
| **reverse -0.15 m/s** | **-0.188 m/s** | new in v2.0.0 |
| yaw 0.5 rad/s | +0.536 rad/s | 107% |

Two changes define this release. **base_link moved to the body centre**, so yaw
commands pivot the robot about itself rather than about one of its own legs, and
**reverse commands work** — v1.x had never been trained on a negative `vx` and
five separate terms treated the command as forward-only.

<img src="docs/figures/base_link_move.png" width="760">

### Gait diagnostics

[`scripts/plot_gait.py`](scripts/plot_gait.py) produces a footfall diagram, foot-tip
paths and support-polygon area from a `BB_GAIT_DUMP` rollout. It reads the same
contact sensor and FK the reward terms use, so the plots and the objective agree
by construction, and the same figure can be produced from a Gazebo or hardware
rollout for a like-for-like comparison.

| forward, cmd +0.30 | reverse, cmd -0.15 |
|:---:|:---:|
| <img src="docs/figures/gait_forward_v2.0.0.png" width="420"> | <img src="docs/figures/gait_reverse_v2.0.0.png" width="420"> |

These exposed something the reward curves did not: despite `crawl_gait` being a
reward term, the learned behaviour keeps only **~2.2 feet loaded** and is
statically unstable ~75% of the time, so it is closer to a trot than a crawl.
Peak contact force reaches **144 N against a 12.6 N body weight** under forward
commands. Reverse is markedly gentler at 52 N peak.

### Pipeline

[`docs/pipeline.md`](docs/pipeline.md) documents the 52-D observation contract,
the policy, and where Isaac, Gazebo and hardware diverge downstream of the joint
target. That divergence is the sim-to-real story: both simulators put a
force-producing element between the target and the joint, and hardware does not.

### Known limitations

- **The exported policy is still substantially clamp-saturated.** Closed-loop at
  cmd 0.30 it sits on the clamp 89% of the time with `abs(a)` up to 26.5. That is
  a 1000x improvement on v1.1.0 (`abs(a) ~ 2.8e4`, 100% clamped), but it is not a
  smooth graded-target policy. Simulators hide this because their actuators
  low-pass the target; hardware does not, so the bringup applies an EWMA before
  the servos.
- **Low command magnitudes overshoot in both directions.** Commanded 0.12 gives
  0.174, commanded -0.05 gives -0.127. The policy has a preferred cruising speed
  near 0.13-0.19 m/s and the command steers direction more than magnitude at the
  low end. Nav2 asks for slow approach speeds and will get faster ones.
- **Yaw drifts when commanded to stop.** Measured +37 degrees over 2.7 s
  immediately after a 180 degree turn, so the robot coasts rather than holding
  heading.

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
| Joint | lower limit | upper_limit | off | channel | direcetion |
|-----|-----|-----|-----|-----|-----|
| arm_a_2_1 | 180 | 50 | 0 | 2 | -1 |
| arm_b_2_1 | 50 | 180 | +10 | 1 | -1 |
| arm_c_2_1 | 150 | 0 | +2 | 0 | -1 |
| arm_a_1_1 | 30 | 150 | 0 | 10 | +1 |
| arm_b_1_1 | 140 | 0 | 0 | 9 | -1 |
| arm_c_1_1 | 180 | 40 | +8 | 8 | -1 |
| arm_a_3_1 | 140 | 0 | 0 | 6 | -1 |
| arm_b_3_1 | 50 | 180 | +10 | 5 | -1 |
| arm_c_3_1 | 0 | 150 | +5 | 4 | -1 |
| arm_a_4_1 | 45 | 180 | 0 | 14 | +1 |
| arm_b_4_1 | 135 | 0 |  0 | 13 | -1 |
| arm_c_4_1 | 40 | 180 | 0 | 12 | -1 |

## Development

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
