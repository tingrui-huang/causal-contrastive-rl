"""
Sanity checks for HiddenRegimeForkEnv (Phase 7).

1) Initial obs: clean vs fixed U=0 vs fixed U=1 (same seed).
2) After k forward steps: alignment across regimes (legacy block for k=1,2).
3) **Scan:** for each k = 0..MAX_SCAN, when does **U0 vs U1** ``obs["image"]`` first differ?
   Use this to line up **contrastive horizon** ``k`` (see ``utils/contrastive_sampling.DEFAULT_K``).

Run: ``python experiments/inspect_hidden_fork.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.core.actions import Actions

import envs  # noqa: F401
from configs.training_defaults import TRAIN_SEED
from utils.contrastive_sampling import DEFAULT_K
from utils.preprocess import extract_state

ENV_CLEAN = "CausalContrastive-HiddenFork-15x15-Clean-v0"
ENV_CONF = "CausalContrastive-HiddenFork-15x15-v0"
MAX_SCAN = 12


def _cmp(name_a: str, obs_a, name_b: str, obs_b) -> None:
    img_eq = np.array_equal(obs_a["image"], obs_b["image"])
    dir_eq = int(obs_a["direction"]) == int(obs_b["direction"])
    print(f"  [{name_a}] vs [{name_b}]  image equal: {img_eq}  direction equal: {dir_eq}")
    if not img_eq:
        da = obs_a["image"].astype(np.int32) - obs_b["image"].astype(np.int32)
        print(f"    |image diff| max = {np.abs(da).max()}")


def _obs_after_k_forward(env_id: str, seed: int, k: int, **kwargs):
    env = gym.make(env_id, render_mode="rgb_array", **kwargs)
    obs, _ = env.reset(seed=seed)
    for _ in range(k):
        obs, _, _, _, _ = env.step(int(Actions.forward))
    return obs


def main() -> None:
    seed = TRAIN_SEED

    print(f"=== Initial reset (same seed) ===  seed={seed} (from configs/training_defaults.TRAIN_SEED)\n")

    e_clean = gym.make(ENV_CLEAN, render_mode="rgb_array")
    o0, i0 = e_clean.reset(seed=seed)
    print(f"Clean env  info[confounder] = {i0.get('confounder')}")

    e_u0 = gym.make(ENV_CONF, render_mode="rgb_array", fixed_u=0)
    o1, i1 = e_u0.reset(seed=seed)
    print(f"Confound+fixed_u=0  info[confounder] = {i1.get('confounder')}")

    e_u1 = gym.make(ENV_CONF, render_mode="rgb_array", fixed_u=1)
    o2, i2 = e_u1.reset(seed=seed)
    print(f"Confound+fixed_u=1  info[confounder] = {i2.get('confounder')}")

    _cmp("clean", o0, "fixed_u=0", o1)
    _cmp("clean", o0, "fixed_u=1", o2)
    _cmp("fixed_u=0", o1, "fixed_u=1", o2)

    s0, s1, s2 = extract_state(o0), extract_state(o1), extract_state(o2)
    print(
        f"\n  Full processed state equal: clean vs U0 = {np.array_equal(s0, s1)}, "
        f"clean vs U1 = {np.array_equal(s0, s2)}, U0 vs U1 = {np.array_equal(s1, s2)}"
    )

    print("\n=== After 1x / 2x forward (sanity) ===\n")
    for k in (1, 2):
        oa = _obs_after_k_forward(ENV_CLEAN, seed, k)
        ob = _obs_after_k_forward(ENV_CONF, seed, k, fixed_u=0)
        oc = _obs_after_k_forward(ENV_CONF, seed, k, fixed_u=1)
        print(f"--- k = {k} forward ---")
        _cmp("clean", oa, "U0", ob)
        _cmp("U0", ob, "U1", oc)

    print("\n=== Scan: U0 vs U1 image match after k forward steps ===\n")
    first_diverge: int | None = None
    saw_diverge = False
    for k in range(0, MAX_SCAN + 1):
        o_u0 = _obs_after_k_forward(ENV_CONF, seed, k, fixed_u=0)
        o_u1 = _obs_after_k_forward(ENV_CONF, seed, k, fixed_u=1)
        match = np.array_equal(o_u0["image"], o_u1["image"])
        state_match = np.array_equal(extract_state(o_u0), extract_state(o_u1))
        mark = ""
        if not match and first_diverge is None:
            first_diverge = k
            mark = "  <-- FIRST divergence (obs image)"
            saw_diverge = True
        elif match and saw_diverge:
            mark = "  (match again: often both reached goal or symmetric state)"
        print(f"  k={k:2d}  U0 vs U1 image equal: {match}  processed state equal: {state_match}{mark}")

    print()
    if first_diverge is None:
        print(
            f"  No divergence within k=0..{MAX_SCAN} (raise MAX_SCAN or check map)."
        )
    else:
        print(
            f"  Summary: U0 and U1 observations first differ after **{first_diverge}** forward step(s) "
            f"(i.e. at the post-step observation following k={first_diverge})."
        )
        print(
            f"  Contrastive default k = {DEFAULT_K} (positive = obs at t+{DEFAULT_K} on same trajectory)."
        )
        if first_diverge > DEFAULT_K:
            print(
                f"  Hint: if anchors are early in the episode, s_{{t+{DEFAULT_K}}} may still "
                f"lie **before** the regime affects the view (divergence starts at step {first_diverge}). "
                f"Consider increasing k in ``utils/contrastive_sampling.py`` or moving the fork closer."
            )
        else:
            print(
                f"  Hint: with DEFAULT_K={DEFAULT_K}, positives s_{{t+{DEFAULT_K}}} for small anchor t "
                f"typically lie **after** the first observation divergence (step {first_diverge}), "
                f"so regimes can affect contrastive pairs; validate with training / separation metrics."
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
