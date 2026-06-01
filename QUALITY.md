# Quality Declaration — Level 2: Research-grade

This repository targets **Quality Level 2** of an Isaac Lab training quality scale
adapted from [ROS 2 REP-2004](https://ros.org/reps/rep-2004.html) for RL training repos.

## Quality Levels

| Level | Name | Summary |
|-------|------|---------|
| **3** | Experimental | Syntax valid, linting passes, conventional commits enforced |
| **2** | Research-grade | + Config structure validated, docstring coverage enforced ← **current** |
| **1** | Production-ready | + Training convergence verified, checkpoint reproducibility, USD asset integrity |

## Level 2 Criteria

| Criterion | CI Job | Status |
|-----------|--------|--------|
| Python syntax compiles | `smoke-test / syntax` | ✓ |
| Packages install without Isaac Lab | `smoke-test / syntax` | ✓ |
| Packages discoverable via importlib | `smoke-test / syntax` | ✓ |
| Ruff lint + format | `compliance / pre-commit` | ✓ |
| No raw `print()` in `source/` | `compliance / file-checks` | ✓ |
| SPDX license header present | `compliance / file-checks` | ✓ |
| Conventional commits enforced | `compliance / commit-lint` | ✓ |
| EnvCfg required fields validated (AST) | `quality-gate / config-validate` | ✓ |
| PPORunnerCfg required fields validated (AST) | `quality-gate / config-validate` | ✓ |
| `gym.register` kwargs validated (AST) | `quality-gate / config-validate` | ✓ |
| Docstring coverage ≥ 80% | `quality-gate / docstring-coverage` | ✓ |
| Dependency automation (pip + Actions) | Dependabot weekly | ✓ |
| Auto-labeling on PR | `labeler` | ✓ |

### Required EnvCfg fields

Every class inheriting `DirectRLEnvCfg` must define:

```
episode_length_s  decimation  action_space  observation_space
```

### Required PPORunnerCfg fields

```
num_steps_per_env  max_iterations  experiment_name
```

### Required `gym.register` kwargs

```
env_cfg_entry_point  rsl_rl_cfg_entry_point
```

Task IDs must be snake_case (all lowercase).

## Level 1 Roadmap (Production-ready)

| Criterion | Notes |
|-----------|-------|
| Training convergence CI | Requires self-hosted runner with Isaac Sim; reward threshold checked after N iterations |
| Checkpoint reproducibility | Seed-locked training, deterministic replay |
| Physics param bounds | dt, decimation, num_envs validated against safe ranges |
| USD asset integrity | Asset hash validation + LFS size limits |
| Isaac Lab version pinning | Explicit version constraint in `pyproject.toml` |

## Running Checks Locally

```bash
# Config structure validation (no Isaac Lab needed)
python scripts/validate_configs.py

# Docstring coverage
pip install interrogate
interrogate source/ --fail-under 80 --ignore-init-method --ignore-magic --ignore-semiprivate --ignore-private --ignore-module --verbose

# Full pre-commit suite
pre-commit run --all-files
```
