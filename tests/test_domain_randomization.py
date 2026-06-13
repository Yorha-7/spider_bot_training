"""CI-safe checks that big_bertha's Heavy Domain Randomization stays configured.

Issue #43 asks for a test guarding the Heavy DR setup that keeps the policy
robust to the sim-to-sim gap. CI has no GPU and cannot boot Isaac Sim /
SimulationApp, so this module never imports the env. It parses the config and
env source with ``ast`` and asserts the randomization is present and non-trivial:

- ``EventCfg`` declares the 3 expected ``EventTerm`` randomizers
  (``physics_material``, ``add_base_mass``, ``push_robot``) with the right modes.
- ``push_robot`` randomizes roll, pitch and yaw (body-tilt robustness).
- ``action_noise_std`` is positive (joint-target action noise).
- The env actually applies observation noise and the action-target noise.

Run with ``pytest tests/`` or directly with ``python tests/test_domain_randomization.py``.
"""

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TASK_DIR = _REPO_ROOT / "source" / "big_bertha" / "big_bertha" / "tasks" / "direct" / "big_bertha"
_ENV_CFG = _TASK_DIR / "big_bertha_env_cfg.py"
_ENV = _TASK_DIR / "big_bertha_env.py"

# Expected EventTerm name -> mode declared on EventCfg.
_EXPECTED_EVENT_TERMS = {
    "physics_material": "startup",
    "add_base_mass": "startup",
    "push_robot": "interval",
}


def _parse(path: Path) -> ast.Module:
    """Parse a source file into an AST module (no import, no GPU)."""
    return ast.parse(path.read_text(), filename=str(path))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    """Return the first top-level/nested class definition with ``name``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in AST")


def _event_terms(event_cfg: ast.ClassDef) -> dict[str, ast.Call]:
    """Map each ``EventTerm(...)`` attribute on EventCfg to its call node."""
    terms: dict[str, ast.Call] = {}
    for node in event_cfg.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "EventTerm"
            and node.targets
            and isinstance(node.targets[0], ast.Name)
        ):
            terms[node.targets[0].id] = node.value
    return terms


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    """Return the value node for keyword ``name`` on a call, or None."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _module_assignment(tree: ast.Module, class_name: str, attr: str) -> ast.expr | None:
    """Return the RHS node of ``attr = ...`` declared on the given class."""
    cls = _find_class(tree, class_name)
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == attr for t in node.targets):
            return node.value
    return None


def test_event_cfg_has_three_randomization_terms():
    """EventCfg declares the 3 expected EventTerms with the correct modes."""
    cfg = _parse(_ENV_CFG)
    event_cfg = _find_class(cfg, "EventCfg")
    terms = _event_terms(event_cfg)

    for name, expected_mode in _EXPECTED_EVENT_TERMS.items():
        assert name in terms, f"EventCfg is missing the {name!r} randomization term"
        mode_node = _kwarg(terms[name], "mode")
        assert mode_node is not None, f"{name!r} EventTerm has no mode= kwarg"
        mode = ast.literal_eval(mode_node)
        assert mode == expected_mode, f"{name!r} mode is {mode!r}, expected {expected_mode!r}"


def test_push_robot_randomizes_orientation():
    """push_robot perturbs roll, pitch and yaw so the policy learns body tilt."""
    cfg = _parse(_ENV_CFG)
    event_cfg = _find_class(cfg, "EventCfg")
    push = _event_terms(event_cfg)["push_robot"]

    params = _kwarg(push, "params")
    assert isinstance(params, ast.Dict), "push_robot params must be a dict literal"
    velocity_range = None
    for key, value in zip(params.keys, params.values):
        if isinstance(key, ast.Constant) and key.value == "velocity_range":
            velocity_range = value
    assert isinstance(velocity_range, ast.Dict), "push_robot needs a velocity_range dict"

    axes = {k.value for k in velocity_range.keys if isinstance(k, ast.Constant)}
    for axis in ("roll", "pitch", "yaw"):
        assert axis in axes, f"push_robot velocity_range is missing the {axis!r} axis"


def test_action_noise_std_is_positive():
    """The env cfg keeps a positive action_noise_std (joint-target DR)."""
    cfg = _parse(_ENV_CFG)
    node = _module_assignment(cfg, "BigberthaEnvCfg", "action_noise_std")
    assert node is not None, "BigberthaEnvCfg.action_noise_std is not defined"
    value = ast.literal_eval(node)
    assert value > 0.0, f"action_noise_std must be > 0 for DR, got {value}"


def test_env_applies_observation_and_action_noise():
    """The env source applies observation noise and the action-target noise."""
    env_src = _ENV.read_text()
    assert "randn_like(obs)" in env_src, "env does not apply observation noise"
    assert "self.cfg.action_noise_std > 0.0" in env_src, "env does not gate action-target noise on action_noise_std"
    assert "randn_like(self._processed_actions)" in env_src, "env does not apply action-target noise"


if __name__ == "__main__":
    test_event_cfg_has_three_randomization_terms()
    test_push_robot_randomizes_orientation()
    test_action_noise_std_is_positive()
    test_env_applies_observation_and_action_noise()
