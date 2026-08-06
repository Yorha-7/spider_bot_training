# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from big_bertha.tasks.direct.big_bertha.symmetry import compute_symmetry

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for the Big Bertha velocity-control task."""

    # 48 steps ~ 1 gait cycle at 0.667 Hz / 50 Hz control (24 covered only a
    # third of a cycle, starving cycle-level credit assignment).
    num_steps_per_env = 48
    max_iterations = 15000
    save_interval = 100
    experiment_name = "big_bertha"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.01 drove an unbounded std ratchet (the env clamps actions, so the
        # return is std-insensitive past ~1 while the bonus keeps paying;
        # v1.0.0 ended at std ~2400 with LR pinned at the adaptive cap).
        entropy_coef=1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=8,  # keeps minibatch size (VRAM) flat after 24->48 steps
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.997,  # ~2 gait cycles of horizon (0.99 was ~1.3)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Mirror-symmetry about the forward (x) axis (Mittal 2024). coeff sized
        # against the MEASURED Loss/symmetry ~3e4 (the old 1e-8 was calibrated
        # to a stale 2.7e8 figure and contributed ~1e-4 -- a numerical no-op).
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0e-4,
            data_augmentation_func=compute_symmetry,
        ),
    )
