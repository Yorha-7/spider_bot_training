# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify the IMU double-rotation bug.

Compares imu.data.ang_vel_b * _imu_negate vs _robot.data.root_ang_vel_b
at every step. If they are identical (diff < 1e-6), the 180° Z rotation
in ImuCfg.offset cancels with the manual [-1,-1,1] negate — confirming
the double-rotation described in the review.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Verify IMU frame alignment for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import big_bertha.tasks  # noqa: F401
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Compare IMU-corrected angular velocity vs root angular velocity."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Device: {unwrapped.device}, Num envs: {unwrapped.num_envs}")
    print()
    print(f"{'Step':>6}  {'max|imu_raw|':>12}  {'max|imu_corr|':>13}  {'max|root_vel|':>13}  {'max|diff|':>10}  Status")
    print("-" * 80)

    # reset environment
    env.reset()
    step = 0
    # simulate environment
    while simulation_app.is_running():
        with torch.inference_mode():
            # sample actions from -1 to 1 (random flailing -> non-zero ang vel)
            actions = 2 * torch.rand(env.action_space.shape, device=unwrapped.device) - 1
            env.step(actions)

            # Read quantities of interest
            imu_raw = unwrapped.scene["imu"].data.ang_vel_b  # rotated sensor frame: (-wx, -wy, wz)
            imu_corrected = imu_raw * unwrapped._imu_negate  # after current [-1,-1,1] -> (wx, wy, wz)
            root_vel = unwrapped._robot.data.root_ang_vel_b  # base frame: (wx, wy, wz)

            diff = (imu_corrected - root_vel).abs().max().item()

            if step % 10 == 0:
                status = "IDENTICAL" if diff < 1e-6 else "DIFFERENT"
                print(
                    f"{step:6d}  {imu_raw.abs().max().item():>12.6f}  "
                    f"{imu_corrected.abs().max().item():>13.6f}  "
                    f"{root_vel.abs().max().item():>13.6f}  "
                    f"{diff:>10.2e}  {status}"
                )
            step += 1

            if step >= 500:
                print(f"\n[INFO] Reached {step} steps, stopping.")
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
