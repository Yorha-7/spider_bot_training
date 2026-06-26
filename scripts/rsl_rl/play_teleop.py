# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy in Isaac Sim with live keyboard teleop (WASD + QE).

Unlike a fixed-velocity replay, this lets you drive the robot interactively and
watch how it actually walks, strafes, and turns. It only writes the velocity
command (env._commands) that the policy already consumes -- the env, env_cfg,
reward, and actuators are untouched.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Teleoperate a trained RL agent with the keyboard.")
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
import time

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

from spider_rl.utils import KeyboardInput

# Clamp keyboard commands to the range the policy was actually trained on
# (big_bertha_env._reset_idx): forward-only vx, small lateral vy, wide yaw.
# Driving outside this band is out-of-distribution and just looks broken.
_VX_RANGE = (0.0, 0.12)
_VY_RANGE = (-0.05, 0.05)
_WZ_RANGE = (-0.8, 0.8)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Drive a trained RSL-RL policy with the keyboard."""
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
    dt = env.unwrapped.step_dt

    keyboard = KeyboardInput()
    keyboard.start()
    keyboard.print_controls()
    print(
        "--- Teleop (focus THIS terminal to type) ---\n"
        "W/S forward/back   A/D strafe   Q/E turn   SPACE stop   Ctrl-C quit\n"
        f"commands clamped to training range: vx{_VX_RANGE} vy{_VY_RANGE} wz{_WZ_RANGE}\n"
        "--------------------------------------------"
    )

    cmd = torch.zeros(1, 3, device=env.unwrapped.device, dtype=torch.float)
    obs = env.get_observations()
    last_print = 0.0
    try:
        while simulation_app.is_running():
            start_time = time.time()
            cmd[0, 0] = _clamp(keyboard.vel_x, *_VX_RANGE)
            cmd[0, 1] = _clamp(keyboard.vel_y, *_VY_RANGE)
            cmd[0, 2] = _clamp(keyboard.ang_z, *_WZ_RANGE)
            with torch.inference_mode():
                env.unwrapped._commands[:] = cmd
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                env.unwrapped._commands[:] = cmd
            if start_time - last_print > 0.5:
                print(
                    f"\r[cmd] vx={cmd[0, 0]:+.3f}  vy={cmd[0, 1]:+.3f}  wz={cmd[0, 2]:+.3f}   ",
                    end="",
                    flush=True,
                )
                last_print = start_time
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        keyboard.stop()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
