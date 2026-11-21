import gymnasium as gym

from . import agents

gym.register(
    id="big_bertha_v2",
    entry_point=f"{__name__}.big_bertha_v2_env:BigberthaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.big_bertha_v2_env_cfg:BigberthaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
