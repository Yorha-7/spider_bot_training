# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained policy through a scripted command sequence -- no keyboard needed.

Cycles the velocity command through forward, backward, strafe left/right, turn
left/right (each held for a few seconds) so you can watch every motion. Only the
velocity command (env._commands) is written; env, env_cfg, reward, and actuators
are untouched. Run with a window to watch live, or pass --video to record a clip.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RL agent through a scripted command sequence.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the sequence.")
parser.add_argument("--video_length", type=int, default=None, help="Video length in steps (default: one full cycle).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--phase_secs", type=float, default=5.0, help="Seconds to hold each command in the sequence.")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
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
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# (label, [vx, vy, wz]).  Trained ranges: vx in [0, 0.12] (forward only,
# the crawl has no reverse), vy in [-0.05, 0.05], wz in [-0.8, 0.8].
SEQUENCE = [
    ("forward", [0.10, 0.0, 0.0]),
    ("strafe left", [0.0, 0.05, 0.0]),
    ("strafe right", [0.0, -0.05, 0.0]),
    ("turn left", [0.0, 0.0, 0.5]),
    ("turn right", [0.0, 0.0, -0.5]),
    ("stop", [0.0, 0.0, 0.0]),
]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Drive a trained RSL-RL policy through the scripted command sequence."""
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

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # one full cycle of the sequence, in steps
    dt = None  # resolved after wrapping (need step_dt)

    if args_cli.video:
        cycle_steps_guess = int((len(SEQUENCE) * args_cli.phase_secs) / 0.02)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "sequence"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length if args_cli.video_length else cycle_steps_guess,
            "disable_logger": True,
        }
        print("[INFO] Recording the command sequence.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

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

    phase_steps = max(1, int(args_cli.phase_secs / dt))
    cycle_steps = phase_steps * len(SEQUENCE)
    cmd = torch.zeros(1, 3, device=env.unwrapped.device, dtype=torch.float)

    print("\n=========== SCRIPTED COMMAND SEQUENCE (no keyboard) ===========")
    for label, c in SEQUENCE:
        print(f"  {args_cli.phase_secs:>4.1f}s  {label:<44} cmd={c}")
    print("===============================================================\n")

    obs = env.get_observations()
    step = 0
    last_idx = -1
    while simulation_app.is_running():
        idx = (step // phase_steps) % len(SEQUENCE)
        if idx != last_idx:
            label, c = SEQUENCE[idx]
            cmd[0, 0], cmd[0, 1], cmd[0, 2] = c[0], c[1], c[2]
            print(f">>> [{step * dt:6.1f}s] {label}  ->  vx={c[0]:+.2f} vy={c[1]:+.2f} wz={c[2]:+.2f}")
            last_idx = idx
        with torch.inference_mode():
            env.unwrapped._commands[:] = cmd
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            env.unwrapped._commands[:] = cmd
        step += 1
        if args_cli.video and step >= cycle_steps:
            print("[INFO] Recorded one full cycle; exiting.")
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
