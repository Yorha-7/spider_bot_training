# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy in Isaac Sim with live keyboard teleop (drive in the viewport).

Keys are bound to the Isaac viewport via carb input, so they register in the
SIM WINDOW (focus the window, not the terminal). Hold-to-move: hold a key to
command, release to stop. Only the velocity command (env._commands) the policy
already consumes is written -- env, env_cfg, reward, and actuators are untouched.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Teleoperate a trained RL agent from the Isaac viewport.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os

import big_bertha.tasks  # noqa: F401
import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import carb
import omni.appwindow

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

# Command magnitudes, clamped to the band the policy was trained on
# (forward-only vx, small lateral vy, wide yaw).
_VX = 0.12
_VY = 0.05
_WZ = 0.8


class ViewportTeleop:
    """Hold-to-move SE2 teleop bound to the Isaac viewport keyboard (carb input)."""

    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = appwindow.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key)

    def _on_key(self, event, *args):
        k = carb.input.KeyboardInput
        et = carb.input.KeyboardEventType
        if event.type in (et.KEY_PRESS, et.KEY_REPEAT):
            if event.input == k.W:
                self.vx = _VX
            elif event.input == k.S:
                self.vx = 0.0  # the crawl has no reverse; S stops forward motion
            elif event.input == k.A:
                self.vy = _VY
            elif event.input == k.D:
                self.vy = -_VY
            elif event.input == k.Q:
                self.wz = _WZ
            elif event.input == k.E:
                self.wz = -_WZ
            elif event.input == k.SPACE:
                self.vx = self.vy = self.wz = 0.0
        elif event.type == et.KEY_RELEASE:
            if event.input in (k.W, k.S):
                self.vx = 0.0
            elif event.input in (k.A, k.D):
                self.vy = 0.0
            elif event.input in (k.Q, k.E):
                self.wz = 0.0
        return True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Drive a trained RSL-RL policy with the viewport keyboard."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    teleop = ViewportTeleop()
    print(
        "\n=========== TELEOP: click/focus the ISAAC VIEWPORT WINDOW, then HOLD keys ===========\n"
        "  W = forward   A/D = strafe   Q/E = turn   S or SPACE = stop\n"
        "  (hold to move, release to stop; commands clamped to the trained range)\n"
        "====================================================================================="
    )

    cmd = torch.zeros(1, 3, device=env.unwrapped.device, dtype=torch.float)
    obs = env.get_observations()
    step = 0
    while simulation_app.is_running():
        cmd[0, 0] = teleop.vx
        cmd[0, 1] = teleop.vy
        cmd[0, 2] = teleop.wz
        with torch.inference_mode():
            env.unwrapped._commands[:] = cmd
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            env.unwrapped._commands[:] = cmd
        step += 1
        if step % 25 == 0:
            print(f"\r[cmd] vx={cmd[0, 0]:+.3f}  vy={cmd[0, 1]:+.3f}  wz={cmd[0, 2]:+.3f}   ", end="", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
