"""Custom MiniGrid environments (Phase 7+). Registers Gymnasium ids on import."""

from __future__ import annotations

from gymnasium.envs.registration import register


def register_causal_envs() -> None:
    register(
        id="CausalContrastive-HiddenFork-9x9-v0",
        entry_point="envs.hidden_regime_fork:HiddenRegimeForkEnv",
        kwargs={"size": 9, "confound": True},
    )
    register(
        id="CausalContrastive-HiddenFork-9x9-Clean-v0",
        entry_point="envs.hidden_regime_fork:HiddenRegimeForkEnv",
        kwargs={"size": 9, "confound": False},
    )


register_causal_envs()
