"""BDX_Awd IsaacLab - Bipedal Locomotion Learning with IsaacLab."""

__version__ = "1.0.0"
__author__ = "BDX Robotics Team"

from .envs.duckling_base_env import DucklingBaseEnv, DucklingBaseCfg
from .envs.duckling_command_env import DucklingCommandEnv, DucklingCommandCfg
from .configs.robots.mini_bdx_cfg import MINI_BDX_CFG, MINI_BDX_PARAMS
from .configs.robots.go_bdx_cfg import GO_BDX_CFG, GO_BDX_PARAMS

__all__ = [
    "DucklingBaseEnv",
    "DucklingBaseCfg",
    "DucklingCommandEnv",
    "DucklingCommandCfg",
    "MINI_BDX_CFG",
    "MINI_BDX_PARAMS",
    "GO_BDX_CFG",
    "GO_BDX_PARAMS",
]
