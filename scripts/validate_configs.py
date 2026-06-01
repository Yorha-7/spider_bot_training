"""Validate Isaac Lab training config files using AST — no Isaac Lab required.

Checks:
- *EnvCfg classes (inheriting DirectRLEnvCfg) define required fields
- PPORunnerCfg defines required training fields
- gym.register calls use snake_case IDs and include required kwargs

Exit 0 on success, 1 on any failure.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

ENV_CFG_REQUIRED = {"episode_length_s", "decimation", "action_space", "observation_space"}
PPO_CFG_REQUIRED = {"num_steps_per_env", "max_iterations", "experiment_name"}
GYM_KWARGS_REQUIRED = {"env_cfg_entry_point", "rsl_rl_cfg_entry_point"}

errors: list[str] = []


def _class_assignments(cls: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for node in cls.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def validate_env_cfg(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "") for b in node.bases
        }
        if "DirectRLEnvCfg" not in base_names:
            continue
        missing = ENV_CFG_REQUIRED - _class_assignments(node)
        if missing:
            errors.append(f"  {rel}: {node.name} missing required field(s): {', '.join(sorted(missing))}")


def validate_ppo_cfg(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    rel = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "PPORunnerCfg":
            continue
        missing = PPO_CFG_REQUIRED - _class_assignments(node)
        if missing:
            errors.append(f"  {rel}: PPORunnerCfg missing required field(s): {', '.join(sorted(missing))}")


def validate_gym_register(path: Path) -> None:
    src = path.read_text()
    if "gym.register" not in src:
        return
    tree = ast.parse(src, filename=str(path))
    rel = path.relative_to(ROOT)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "register"
            and isinstance(func.value, ast.Name)
            and func.value.id == "gym"
        ):
            continue
        task_id: str | None = None
        kwarg_keys: set[str] = set()
        for kw in node.keywords:
            if kw.arg == "id" and isinstance(kw.value, ast.Constant):
                task_id = str(kw.value.value)
            if kw.arg == "kwargs" and isinstance(kw.value, ast.Dict):
                for k in kw.value.keys:
                    if isinstance(k, ast.Constant):
                        kwarg_keys.add(str(k.value))
        if task_id and task_id != task_id.lower():
            errors.append(f"  {rel}: gym.register id '{task_id}' must be snake_case (all lowercase)")
        missing_kw = GYM_KWARGS_REQUIRED - kwarg_keys
        if missing_kw:
            errors.append(f"  {rel}: gym.register missing required kwargs: {', '.join(sorted(missing_kw))}")


def main() -> int:
    for p in sorted(ROOT.glob("source/**/*_env_cfg.py")):
        validate_env_cfg(p)
    for p in sorted(ROOT.glob("source/**/rsl_rl_ppo_cfg.py")):
        validate_ppo_cfg(p)
    for p in sorted(ROOT.glob("source/**/__init__.py")):
        validate_gym_register(p)

    if errors:
        print("Config validation FAILED:")
        for e in errors:
            print(e)
        return 1

    checked = (
        len(list(ROOT.glob("source/**/*_env_cfg.py")))
        + len(list(ROOT.glob("source/**/rsl_rl_ppo_cfg.py")))
        + len(list(ROOT.glob("source/**/__init__.py")))
    )
    print(f"Config validation PASSED ({checked} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
