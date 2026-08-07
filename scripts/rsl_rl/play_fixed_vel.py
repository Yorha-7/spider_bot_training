# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint with fixed velocity commands."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint with fixed velocity commands.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--vx", type=float, default=0.0, help="Fixed forward velocity (m/s)")
parser.add_argument("--vy", type=float, default=0.0, help="Fixed lateral velocity (m/s)")
parser.add_argument("--omega", type=float, default=0.0, help="Fixed yaw rate (rad/s)")

# Canonical demo sequence for the seq GIF, as "vx,vy,wz:steps" at 50 Hz.
# forward -> right 90 -> forward -> REVERSE -> left 180 -> stop.
# Turn durations assume the measured ~0.45 rad/s achieved at a 0.5 command:
# 90 deg = 1.571 rad -> 3.5 s = 175 steps; 180 deg -> 7.0 s = 350 steps.
# Total is exactly 1000 steps = 20.0 s, which is episode_length_s, so the whole
# sequence plays inside one episode and no reset cuts it short.
DEMO_SEQ = (
    "0.30,0,0:150;"      # forward          3.0 s
    "0,0,-0.5:175;"      # turn right 90    3.5 s  (negative wz = clockwise)
    "0.30,0,0:150;"      # forward          3.0 s
    "-0.15,0,0:125;"     # REVERSE          2.5 s  (new in v2.0.0)
    "0,0,0.5:350;"       # turn left 180    7.0 s
    "0,0,0:50"           # stop             1.0 s
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

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
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent at fixed velocity."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Recording/eval must fit one uncut episode: the 20 s training timeout
    # resets the robot mid-video otherwise.
    env_cfg.episode_length_s = 120.0

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # override commands with fixed velocities
    fixed_cmd = torch.tensor([args_cli.vx, args_cli.vy, args_cli.omega], device=env.unwrapped.device)
    env.unwrapped._commands[:] = fixed_cmd
    # Register the override on the env so its _reset_idx re-applies these fixed
    # values instead of sampling random commands. Without this, an env that
    # resets mid-playback gets a random forward command baked into the very next
    # observation, so setting vx=vy=omega=0 still produced motion (issue #40).
    env.unwrapped._command_override = fixed_cmd
    print(f"[INFO] Fixed velocity commands: vx={args_cli.vx}, vy={args_cli.vy}, omega={args_cli.omega}")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_fixed_vel"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    # re-apply fixed commands after any reset that may have occurred during get_observations
    env.unwrapped._commands[:] = fixed_cmd
    timestep = 0
    # BB_EVAL_STEPS=<n>: after a 250-step settle, collect n steps of metrics
    # (vx/wz tracking, tilt, per-ankle amplitude), print one JSON line, exit.
    eval_steps = int(os.environ.get("BB_EVAL_STEPS", "0"))
    eval_skip = 250
    ev = {"vx": [], "wz": [], "tilt": [], "q_ankle": []} if eval_steps else None
    # BB_CMD_SEQ="vx,vy,wz:steps;...": step through a command sequence (for
    # the multi-command demo GIF); overrides the fixed command over time.
    # BB_CMD_SEQ="demo" expands to DEMO_SEQ below.
    seq = []
    raw_seq = os.environ.get("BB_CMD_SEQ")
    if raw_seq == "demo":
        raw_seq = DEMO_SEQ
    if raw_seq:
        t_acc = 0
        for part in raw_seq.split(";"):
            cmd_s, dur = part.split(":")
            t_acc += int(dur)
            seq.append((t_acc, torch.tensor([float(x) for x in cmd_s.split(",")], device=env.unwrapped.device)))
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # override commands for any environments that just reset
            if seq:
                for t_end, cmd in seq:
                    if timestep < t_end:
                        fixed_cmd = cmd
                        env.unwrapped._command_override = cmd
                        break
            env.unwrapped._commands[:] = fixed_cmd
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)
        if ev is not None:
            timestep += 1
            if timestep > eval_skip:
                robot = env.unwrapped._robot
                ev["vx"].append(robot.data.root_lin_vel_b[:, 0].mean().item())
                ev["wz"].append(robot.data.root_ang_vel_b[:, 2].mean().item())
                g = robot.data.projected_gravity_b
                tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
                ev["tilt"].append(tilt.mean().item())
                # type-grouped joint order: ankles are [8:12] = legs 1..4 (FR/FL/RL/RR)
                ev["q_ankle"].append(robot.data.joint_pos[:, 8:12].mean(dim=0).cpu())
            if timestep >= eval_skip + eval_steps:
                import json

                q = torch.stack(ev["q_ankle"])  # (T, 4)
                amp = (q.max(dim=0).values - q.min(dim=0).values).tolist()
                front = 0.5 * (amp[0] + amp[1])
                rear = 0.5 * (amp[2] + amp[3])
                tilts = torch.tensor(ev["tilt"])
                print(
                    "BB_EVAL "
                    + json.dumps(
                        {
                            "cmd": [args_cli.vx, args_cli.vy, args_cli.omega],
                            "vx_mean": sum(ev["vx"]) / len(ev["vx"]),
                            "wz_mean": sum(ev["wz"]) / len(ev["wz"]),
                            "tilt_mean_deg": tilts.mean().item(),
                            "tilt_p95_deg": tilts.quantile(0.95).item(),
                            "ankle_amp": amp,
                            "ankle_front_rear_ratio": front / max(rear, 1e-6),
                        }
                    )
                )
                break
        elif args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
