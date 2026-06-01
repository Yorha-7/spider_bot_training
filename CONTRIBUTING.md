# Contributing

Thank you for your interest in contributing. This project trains legged robots inside
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) using RSL-RL / SKRL. Please read
this guide before opening a PR.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Adding a New Environment or Robot](#adding-a-new-environment-or-robot)
- [Commit Messages](#commit-messages)
- [Pull Requests](#pull-requests)
- [Testing](#testing)

---

## Getting Started

**Requirements:**

| Dependency | Version |
|---|---|
| Python | 3.10 – 3.11 |
| Isaac Sim | 4.x (bundled with Isaac Lab) |
| Isaac Lab | ≥ 2.0 |
| CUDA | 12.x (RTX GPU required for Isaac Sim) |

Install Isaac Lab first by following the
[official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Then install this repo in editable mode inside the Isaac Lab Python environment:

```bash
# From the repo root
./isaaclab.sh -p -m pip install -e source/spider_rl
./isaaclab.sh -p -m pip install -e source/big_bertha
```

---

## Development Setup

Install pre-commit hooks so formatting and lint checks run automatically before
every commit:

```bash
pip install pre-commit
pre-commit install
```

Run all hooks manually at any time:

```bash
pre-commit run --all-files
```

---

## Code Style

- **Formatter:** `ruff format` (configured via `.pre-commit-config.yaml`)
- **Linter:** `ruff check`
- **Type hints:** required on all public functions and class fields
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings), one-liners acceptable for simple methods
- **Line length:** 120 characters
- **No raw `print()` calls** in `source/` — use `logging` or `omni.log` instead

Bad:
```python
print(f"reward: {reward}")
```
Good:
```python
import logging
log = logging.getLogger(__name__)
log.debug("reward: %s", reward)
```

---

## Adding a New Environment or Robot

### USD Assets

- Place USD files under `assets/usd/<robot_name>/`
- Name the root USD file `<robot_name>.usd`
- Articulation root must be set on the top-level prim
- Collision meshes must belong to a named collision group if
  `replicate_physics=True` is used in your scene config

### Python Package Layout

Follow the existing layout (e.g. `source/big_bertha/`):

```
source/<robot_name>/
├── <robot_name>/
│   ├── __init__.py
│   ├── assets/
│   │   └── <robot_name>.py          # ArticulationCfg
│   └── tasks/direct/<robot_name>/
│       ├── __init__.py              # gym.register(id="<robot_name>", ...)
│       ├── <robot_name>_env.py      # DirectRLEnv subclass
│       ├── <robot_name>_env_cfg.py  # EnvCfg dataclass
│       └── agents/
│           └── rsl_rl_ppo_cfg.py    # PPORunnerCfg
└── setup.py
```

### Registration convention

The gymnasium `id` **must** be the snake_case robot name:

```python
gym.register(
    id="<robot_name>",
    entry_point=f"{__name__}.<robot_name>_env:<ClassName>Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ...,
        "rsl_rl_cfg_entry_point": ...,
    },
)
```

### Physics defaults

| Parameter | Guidance |
|---|---|
| `dt` | 1/200 (200 Hz physics) |
| `render_interval` | equal to `decimation` |
| `replicate_physics` | `True` unless USD has non-replicable collision groups |
| `num_envs` | start with 32, scale up after smoke test |

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>
```

**Types:** `feat` · `fix` · `refactor` · `chore` · `docs` · `test` · `ci` · `merge`

Rules (enforced by CI):
- Type and subject must be **lowercase**
- No period at the end of the subject
- Subject ≤ 72 characters

Examples:

```
feat: add velocity tracking reward term
fix: correct joint name in actuator config
chore: update rsl-rl dependency to 2.3
docs: document physics parameters for big_bertha
```

---

## Pull Requests

1. **Open an issue first** for non-trivial changes so the approach can be discussed
2. Branch off the correct base:
   - `main` — project-wide changes (CI, docs, tooling)
   - `spider_imprv` — spider-specific work
   - `big_bertha` — big_bertha-specific work
3. Branch names follow `<type>/<short-name>` (e.g. `feat/contact-reward`)
4. Keep PRs focused — one feature or fix per PR
5. PR title must follow the same commit convention
6. Link the issue: `Fixes #<number>`

**PR checklist:**

- [ ] `pre-commit run --all-files` passes locally
- [ ] New environment runs for at least 100 training iterations without crashing
- [ ] No raw `print()` calls introduced in `source/`
- [ ] USD assets added to `assets/usd/` and committed via Git LFS
- [ ] `CONTRIBUTING.md` and `README.md` updated if adding a new robot

---

## Testing

There is no automated unit test suite yet. Until one is added, manually verify:

```bash
# Smoke-test a new environment (headless, 32 envs, 500 iterations)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task <robot_name> --num_envs 32 --max_iterations 500 --headless

# Confirm the play script loads a checkpoint without error
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task <robot_name> --num_envs 4 \
  --load_log <log_dir> --checkpoint model_000500.pt
```

If you have added reward terms, include a brief note in the PR description showing
the reward curve is non-trivial (tensorboard screenshot or text summary).
