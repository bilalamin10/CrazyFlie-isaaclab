# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Quadcopter environment."""

import gymnasium as gym

from . import agents
from . import quadcopter_env_cfg as cfgs

# Per-shape task IDs for the baseline matrix
_SHAPES = ["Hover", "Static", "Circle", "Lemniscate", "Lissajous"]
for shape in _SHAPES:
    gym.register(
        id=f"Isaac-Quadcopter-{shape}-Direct-v0",
        entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(cfgs, f"Quadcopter{shape}Cfg"),
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        },
    )

# Backward-compatible original task ID — keep so old training scripts still work
gym.register(
    id="Isaac-Quadcopter-Direct-v0",
    entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_cfg:QuadcopterEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Quadcopter-StaticAH-Direct-v0",
    entry_point=f"{__name__}.quadcopter_env:QuadcopterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quadcopter_env_cfg:QuadcopterStaticAHCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:QuadcopterPPORunnerCfg",
    },
)

print(f"[INFO] Registered Isaac-Quadcopter-* tasks from: {__file__}")