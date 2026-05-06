# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents
from .quadcopter_env import QuadcopterEnv, QuadcopterEnvCfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Quadcopter-Direct-v0",
    entry_point="crazyflie.tasks.direct.quadcopter.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "crazyflie.tasks.direct.quadcopter.quadcopter_env:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": "crazyflie.tasks.direct.quadcopter.agents:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": "crazyflie.tasks.direct.quadcopter.agents.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": "crazyflie.tasks.direct.quadcopter.agents:skrl_ppo_cfg.yaml",
    },
)

print(f"[INFO] Registered Isaac-Quadcopter-Direct-v0 using custom env from: {__file__}")


gym.register(
    id="Isaac-Quadcopter-Direct-v01",
    entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Quadcopter-DomainRand-v0",
    entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_domain_rand_cfg:QuadcopterDomainRandEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


