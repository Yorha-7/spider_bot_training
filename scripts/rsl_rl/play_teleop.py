# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a trained policy with teleoperation controls."""

import argparse

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained policy with teleoperation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--fixed_velocity",
    type=float,
    nargs=3,
    default=None,
    metavar=("LIN_X", "LIN_Y", "ANG_Z"),
    help="Fixed velocity command: linear_x linear_y angular_z",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_known_args()[0]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import spider_rl.tasks  # noqa: F401


def main():
    env_cfg = None
    for task_spec in gym.registry.values():
        if task_spec.id == args_cli.task:
            env_cfg_entry_point = task_spec.kwargs["env_cfg_entry_point"]
            module_path, class_name = env_cfg_entry_point.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            env_cfg = getattr(module, class_name)()
            break

    if env_cfg is None:
        raise ValueError(f"Could not find environment config for task: {args_cli.task}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="human")

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

    if args_cli.checkpoint:
        checkpoint = torch.load(args_cli.checkpoint)
        env.policy.load_state_dict(checkpoint["model_state_dict"])
        env.policy.eval()
        print(f"[INFO] Loaded model from: {args_cli.checkpoint}")

    env.unwrapped.set_teleop_mode(True)

    if args_cli.fixed_velocity:
        env.unwrapped.set_fixed_velocity(list(args_cli.fixed_velocity))
        print(
            f"[INFO] Fixed velocity: lin_x={args_cli.fixed_velocity[0]},"
            f" lin_y={args_cli.fixed_velocity[1]}, ang_z={args_cli.fixed_velocity[2]}"
        )
    else:
        from spider_rl.utils import KeyboardInput

        keyboard = KeyboardInput()
        keyboard.start()
        keyboard.print_controls()

    obs, _ = env.reset()
    while simulation_app.is_running():
        if args_cli.fixed_velocity is None:
            env.unwrapped.set_commands_from_keyboard(keyboard.vel_x, keyboard.vel_y, keyboard.ang_z)
        obs, _ = env.step(torch.zeros(args_cli.num_envs, env.action_space.shape[0], device=env.device))

    if args_cli.fixed_velocity is None:
        keyboard.stop()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
