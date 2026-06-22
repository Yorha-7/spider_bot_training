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

    num_steps_per_env = 24
    max_iterations = 5000
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
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # Mirror-symmetry about the forward (x) axis -> a laterally-unbiased gait
        # that walks straight under DART's contact too (the principled cure for
        # the PhysX->DART crab; Mittal 2024). use_mirror_loss directly minimizes
        # the asymmetry of the policy's MEAN actions, which bypasses the action
        # saturation (std ~1676) that left the data-augmentation alone unable to
        # push the crab below ~0.5 m. coeff is sized so the term (~coeff*sym_loss
        # with sym_loss ~2.7e8) is comparable to the PPO losses. See symmetry.py.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            mirror_loss_coeff=1.0e-8,
            data_augmentation_func=compute_symmetry,
        ),
    )
