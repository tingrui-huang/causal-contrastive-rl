"""Custom MiniGrid environments (Phase 7+). Registers Gymnasium ids on import."""

from __future__ import annotations

from gymnasium.envs.registration import register


def register_causal_envs() -> None:
    register(
        id="CausalContrastive-HiddenFork-15x15-v0",
        entry_point="envs.hidden_regime_fork:HiddenRegimeForkEnv",
        kwargs={"size": 15, "confound": True, "map_variant": "branch_wall"},
    )
    register(
        id="CausalContrastive-HiddenFork-15x15-Clean-v0",
        entry_point="envs.hidden_regime_fork:HiddenRegimeForkEnv",
        kwargs={"size": 15, "confound": False, "map_variant": "branch_wall"},
    )
    register(
        id="CausalContrastive-HiddenForkHiddenTrap-15x15-v0",
        entry_point="envs.hidden_regime_fork:HiddenRegimeForkEnv",
        kwargs={"size": 15, "confound": True, "map_variant": "hidden_trap"},
    )
    register(
        id="CausalContrastive-HiddenForkHiddenTrap-15x15-Clean-v0",
        entry_point="envs.hidden_regime_fork:HiddenRegimeForkEnv",
        kwargs={"size": 15, "confound": False, "map_variant": "hidden_trap"},
    )
    register(
        id="CausalContrastive-WindyCorridor-15x15-v0",
        entry_point="envs.windy_corridor:WindyCorridorEnv",
        kwargs={"size": 15, "confound": True},
    )
    register(
        id="CausalContrastive-WindyCorridor-15x15-Clean-v0",
        entry_point="envs.windy_corridor:WindyCorridorEnv",
        kwargs={"size": 15, "confound": False},
    )
    register(
        id="CausalContrastive-HiddenForkHiddenTrap-15x15-Wind-v0",
        entry_point="envs.windy_hidden_regime_fork:WindyHiddenRegimeForkEnv",
        kwargs={"size": 15, "confound": True, "map_variant": "hidden_trap"},
    )
    register(
        id="CausalContrastive-HiddenForkHiddenTrap-15x15-Wind-Clean-v0",
        entry_point="envs.windy_hidden_regime_fork:WindyHiddenRegimeForkEnv",
        kwargs={"size": 15, "confound": False, "map_variant": "hidden_trap"},
    )


register_causal_envs()
