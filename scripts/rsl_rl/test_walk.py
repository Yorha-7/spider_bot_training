# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-phase walk quality gate test for Big Bertha."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Run multi-phase walk quality gate test.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--steps", type=int, default=200, help="Steps per phase (total = 4 * steps)")
parser.add_argument("--csv", type=str, default=None, help="Optional path to save raw data CSV")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import math
import os

import big_bertha.tasks  # noqa: F401
import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _deg(gx, gy):
    """Convert projected gravity (x,y) to approximate tilt angle in degrees."""
    return math.atan2(math.sqrt(gx**2 + gy**2), abs(gy) + 1e-8) * 180.0 / math.pi


PHASES = [
    {"name": "FORWARD", "cmd": (0.10, 0.00, 0.00)},
    {"name": "TURN L", "cmd": (0.00, 0.00, 0.50)},
    {"name": "TURN R", "cmd": (0.00, 0.00, -0.50)},
    {"name": "STOP", "cmd": (0.00, 0.00, 0.00)},
]


class MetricBuffer:
    """Lightweight rolling buffer for a single metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []

    def record(self, value):
        self.data.append(value)

    @property
    def tensor(self):
        return torch.tensor(self.data, dtype=torch.float)

    @property
    def mean(self):
        return float(self.tensor.mean()) if self.data else 0.0

    @property
    def std(self):
        return float(self.tensor.std()) if len(self.data) > 1 else 0.0


class PhaseRecorder:
    """Records all metrics for one test phase."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.vx = MetricBuffer()
        self.vy = MetricBuffer()
        self.vz = MetricBuffer()
        self.wz = MetricBuffer()
        self.gx = MetricBuffer()
        self.gy = MetricBuffer()
        self.gaz = MetricBuffer()
        self.base_height = MetricBuffer()
        self.vz_std = MetricBuffer()
        self.action_rate = MetricBuffer()
        self.feet_air = [MetricBuffer() for _ in range(4)]
        self.multi_swing = MetricBuffer()
        self.foot_clearance = MetricBuffer()
        self.collapsed = False

    def step(self, env):
        """Record one step of data from the unwrapped env (env 0)."""
        robot = env._robot.data
        imu = env.scene["imu"].data
        contact = env._contact_sensor.data
        negate = env._imu_negate

        lin_vel = robot.root_lin_vel_b[0]
        ang_vel = imu.ang_vel_b[0] * negate
        grav = imu.projected_gravity_b[0] * negate
        height = robot.root_pos_w[0, 2]
        air_time = contact.current_air_time[0, env._feet_ids]
        feet_z = robot.body_pos_w[0, env._feet_body_ids, 2]
        act_rate = (env._actions[0] - env._previous_actions[0]).abs().mean()

        self.vx.record(float(lin_vel[0]))
        self.vy.record(float(lin_vel[1]))
        self.vz.record(float(lin_vel[2]))
        self.wz.record(float(ang_vel[2]))
        self.gx.record(float(grav[0]))
        self.gy.record(float(grav[1]))
        self.base_height.record(float(height))
        self.action_rate.record(float(act_rate))

        for i in range(4):
            self.feet_air[i].record(float(air_time[i]))

        n_swing = int((air_time > 0.06).float().sum().item())
        self.multi_swing.record(1.0 if n_swing >= 2 else 0.0)

        swinging = (air_time > 0.06).float()
        clearance = (torch.exp(-torch.square((feet_z - 0.045) / 0.03)) * swinging).sum()
        self.foot_clearance.record(float(clearance))

        if height < 0.04:
            self.collapsed = True


def _gate(label, value, threshold, better="higher", units=""):
    """Check a single quality gate and return (passed, line)."""
    if better == "higher":
        passed = value >= threshold
    elif better == "lower":
        passed = value <= threshold
    elif better == "abs_lower":
        passed = abs(value) <= threshold
    else:
        passed = False
    arrow = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    val_str = f"{value:.3f}" if isinstance(value, float) else str(value)
    thresh_str = f"{threshold:.3f}" if isinstance(threshold, float) else str(threshold)
    return passed, f"  {label:.<20s} {val_str}{units}  (need {better} {thresh_str})  {arrow} {status}"


def run_phase(env, unwrapped, policy, steps, cmd, phase_name):
    """Run one phase and return a PhaseRecorder."""
    fixed_cmd = torch.tensor(cmd, device=unwrapped.device)
    unwrapped._commands[:] = fixed_cmd
    unwrapped._command_override = fixed_cmd

    recorder = PhaseRecorder()
    obs = env.get_observations()
    unwrapped._commands[:] = fixed_cmd

    for _ in range(steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            unwrapped._commands[:] = fixed_cmd
            recorder.step(unwrapped)

    return recorder


def print_gates(gates):
    """Print a list of (passed, line) tuples and return whether all passed."""
    all_pass = True
    for passed, line in gates:
        print(line)
        if not passed:
            all_pass = False
    return all_pass


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run the walk quality gate test."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    unwrapped = env.unwrapped

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=unwrapped.device)

    steps = args_cli.steps

    # --- Header ---
    print()
    print("=" * 70)
    print("  WALK QUALITY GATE TEST")
    print(f"  Checkpoint: {resume_path}")
    print(f"  Steps per phase: {steps}  (total: {len(PHASES) * steps})")
    print("=" * 70)
    print()

    all_phases_pass = True
    phase_results = []

    for phase in PHASES:
        name = phase["name"]
        cmd = phase["cmd"]
        print(f">>> Phase: {name}  (cmd: vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f})")
        print("-" * 50)

        rec = run_phase(env, unwrapped, policy, steps, cmd, name)

        gates = []

        if name == "FORWARD":
            cmd_vx = cmd[0]
            gates.append(_gate("Forward speed", rec.vx.mean, 0.5 * cmd_vx, "higher", " m/s"))
            gates.append(_gate("Lateral drift", rec.vy.mean, 0.020, "abs_lower", " m/s"))
            gates.append(_gate("Heading hold", rec.wz.mean, 0.050, "abs_lower", " rad/s"))
            gates.append(_gate("Base tilt", _deg(rec.gx.mean, rec.gy.mean), 5.0, "lower", "°"))
            gates.append(_gate("Vertical bounce", rec.vz.std, 0.080, "lower", " m/s"))
            gates.append(_gate("Gait pattern", rec.multi_swing.mean * 100, 15.0, "lower", " % multi"))
            gates.append(_gate("Smoothness", rec.action_rate.mean, 0.050, "lower", ""))
            gates.append(_gate("Collapse", 0 if rec.collapsed else 1, 1, "higher", ""))
            phase_ok = print_gates(gates)

        elif name == "TURN L":
            cmd_wz = cmd[2]
            gates.append(_gate("Yaw tracking", rec.wz.mean, 0.5 * cmd_wz, "higher", " rad/s"))
            gates.append(_gate("Base tilt", _deg(rec.gx.mean, rec.gy.mean), 6.0, "lower", "°"))
            gates.append(_gate("Collapse", 0 if rec.collapsed else 1, 1, "higher", ""))
            phase_ok = print_gates(gates)

        elif name == "TURN R":
            cmd_wz = cmd[2]
            gates.append(_gate("Yaw tracking", rec.wz.mean, 0.5 * cmd_wz, "lower", " rad/s"))
            gates.append(_gate("Base tilt", _deg(rec.gx.mean, rec.gy.mean), 6.0, "lower", "°"))
            gates.append(_gate("Collapse", 0 if rec.collapsed else 1, 1, "higher", ""))
            phase_ok = print_gates(gates)

        elif name == "STOP":
            gates.append(_gate("Stand still |vx|", abs(rec.vx.mean), 0.020, "lower", " m/s"))
            gates.append(_gate("Lateral drift", rec.vy.mean, 0.020, "abs_lower", " m/s"))
            gates.append(_gate("Heading hold", rec.wz.mean, 0.030, "abs_lower", " rad/s"))
            gates.append(_gate("Collapse", 0 if rec.collapsed else 1, 1, "higher", ""))
            phase_ok = print_gates(gates)

        else:
            phase_ok = True

        if not phase_ok:
            all_phases_pass = False

        phase_results.append(rec)
        print()

    # --- Overall verdict ---
    print("=" * 70)
    if all_phases_pass:
        print("  >>> OVERALL: PASS (all gates passed) <<<")
    else:
        print("  >>> OVERALL: FAIL (one or more gates failed) <<<")
    print("=" * 70)

    # --- Optional CSV export ---
    if args_cli.csv is not None:
        _save_csv(args_cli.csv, PHASES, phase_results)

    env.close()


def _save_csv(path, phases, results):
    """Write raw metric data to CSV."""
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        headers = [
            "phase",
            "step",
            "vx",
            "vy",
            "vz",
            "wz",
            "gx",
            "gy",
            "height",
            "action_rate",
            "air0",
            "air1",
            "air2",
            "air3",
            "multi_swing",
            "clearance",
        ]
        writer.writerow(headers)
        for phase, rec in zip(phases, results):
            n = len(rec.vx.data)
            for i in range(n):
                row = [
                    phase["name"],
                    i,
                    rec.vx.data[i] if i < len(rec.vx.data) else "",
                    rec.vy.data[i] if i < len(rec.vy.data) else "",
                    rec.vz.data[i] if i < len(rec.vz.data) else "",
                    rec.wz.data[i] if i < len(rec.wz.data) else "",
                    rec.gx.data[i] if i < len(rec.gx.data) else "",
                    rec.gy.data[i] if i < len(rec.gy.data) else "",
                    rec.base_height.data[i] if i < len(rec.base_height.data) else "",
                    rec.action_rate.data[i] if i < len(rec.action_rate.data) else "",
                    rec.feet_air[0].data[i] if i < len(rec.feet_air[0].data) else "",
                    rec.feet_air[1].data[i] if i < len(rec.feet_air[1].data) else "",
                    rec.feet_air[2].data[i] if i < len(rec.feet_air[2].data) else "",
                    rec.feet_air[3].data[i] if i < len(rec.feet_air[3].data) else "",
                    rec.multi_swing.data[i] if i < len(rec.multi_swing.data) else "",
                    rec.foot_clearance.data[i] if i < len(rec.foot_clearance.data) else "",
                ]
                writer.writerow(row)
    print(f"[INFO] Saved raw data to {path}")


if __name__ == "__main__":
    if args_cli.use_pretrained_checkpoint:
        args_cli.num_envs = 1
    main()
    simulation_app.close()
