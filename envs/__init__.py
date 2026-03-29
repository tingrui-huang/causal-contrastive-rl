"""Custom MiniGrid environments — WindyCorridor only. Registers Gymnasium ids on import."""

from __future__ import annotations

from gymnasium.envs.registration import register


def register_causal_envs() -> None:
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
        id="CausalContrastive-WindyCorridor-15x15-Lethal-v0",
        entry_point="envs.windy_corridor:WindyCorridorEnv",
        kwargs={"size": 15, "confound": True, "lethal_boundaries": True},
    )
    register(
        id="CausalContrastive-WindyCorridor-15x15-Lethal-Clean-v0",
        entry_point="envs.windy_corridor:WindyCorridorEnv",
        kwargs={"size": 15, "confound": False, "lethal_boundaries": True},
    )


register_causal_envs()
