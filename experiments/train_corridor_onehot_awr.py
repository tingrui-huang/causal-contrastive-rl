"""
Formalized AWR actor for the one-hot contrastive critic (SUMMARY open-item #1).

This lifts the working ``train_awr`` / ``actor_rollout`` recipe out of the scratch
``tmp_repro_correct.py`` into a real, importable experiment that reuses the formal
modules instead of re-implementing dynamics inline.

Pipeline
--------
1. Load a one-hot critic checkpoint (from ``train_corridor_onehot.py``) — its weights,
   its ``sa_index`` / ``goal_index`` one-hot tables, and its config.
2. Load the worst-case offline data it was trained on (HER-style anchors).
3. Train a goal-conditioned actor by Advantage-Weighted Regression:
     adv(s,a) = f(s,a,s_g) - mean_a' f(s,a',s_g)        [critic advantage]
     w        = clip(exp(adv / beta), 0, w_max)          [AWR weight]
     loss     = -mean( w * log pi(a_data | s, s_g) )     [weighted BC over data acts]
   The actor uses RAW (x,y,dir) features with a dir-agnostic goal and a valid-action
   logit mask — exactly the tmp recipe — so it never clones the no-op MiniGrid actions.
4. Save the actor checkpoint and evaluate by rolling out in the REAL WindyCorridor
   SCM under three wind regimes (natural / forced-south / still).

Why the actor (not the critic) drives: the one-hot critic ranks future GOAL CELLS
well but does not by itself rank ACTIONS (greedy-critic just spins) — the AWR actor
is what turns the critic's value signal into a route-following policy.

Usage::
    python experiments/train_corridor_onehot_awr.py \\
        --critic checkpoints/corridor_onehot_formal.npz \\
        --data data/corridor_worst_case_formal.npz \\
        --output checkpoints/corridor_onehot_awr_seed0.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy
from envs import make_windy_corridor_scm
from envs.windy_corridor import GOAL_POS, START_POS
from experiments.train_corridor_onehot import encode_sa, encode_goal, GAMMA, VALID_ACTIONS

# Forced-wind presets mirror eval_corridor_critic_greedy.py (0=E,1=S,2=W,3=N,4=still).
FORCED_WIND_PRESETS = {
    "south": (0.0, 1.0, 0.0, 0.0, 0.0),
    "still": (0.0, 0.0, 0.0, 0.0, 1.0),
}


# --------------------------------------------------------------------------- #
#  Critic loading (one-hot)                                                   #
# --------------------------------------------------------------------------- #
def load_onehot_critic(path: Path):
    """Rebuild a one-hot ContrastiveCriticNumpy + its index tables from a ckpt."""
    ckpt = np.load(path, allow_pickle=True)
    cfg = json.loads(str(ckpt["config_json"]))
    critic = ContrastiveCriticNumpy(
        state_dim=cfg["state_dim"], n_actions=cfg["n_actions"],
        hidden=cfg["hidden"], emb_dim=cfg["emb_dim"], tau=cfg["tau"], seed=cfg["seed"],
        goal_dims=tuple(cfg["goal_dims"]) if cfg.get("goal_dims") is not None else None,
        goal_feat_dim=cfg.get("goal_feat_dim"),
    )
    for k in ("sa_W1", "sa_b1", "sa_W2", "sa_b2", "g_W1", "g_b1", "g_W2", "g_b2"):
        getattr(critic, k)[...] = ckpt[k]
    sa_index = {
        tuple(int(v) for v in k.split(",")): idx
        for k, idx in json.loads(str(ckpt["sa_index_json"])).items()
    }
    goal_index = {
        tuple(int(v) for v in k.split(",")): idx
        for k, idx in json.loads(str(ckpt["goal_index_json"])).items()
    }
    return critic, sa_index, goal_index, cfg


# --------------------------------------------------------------------------- #
#  AWR training                                                               #
# --------------------------------------------------------------------------- #
def _geom_offset(rng: np.random.Generator, max_val: int) -> int:
    off = int(rng.geometric(1.0 - GAMMA))
    return min(max(off, 1), max_val)


def _critic_scores(critic, sa_index, goal_index, s_raw, g_raw) -> np.ndarray:
    """f(s, a_i, g) for all actions over a raw-state batch, via one-hot encoding."""
    s_enc = encode_sa(s_raw, sa_index)
    g_enc = encode_goal(g_raw, goal_index)
    return critic.score_actions(s_enc, g_enc)


def train_awr(
    *,
    critic, sa_index, goal_index,
    episode_states, episode_actions, n_actions,
    beta: float = 0.5, steps: int = 15000, lr: float = 2e-3,
    batch_size: int = 64, w_max: float = 20.0, seed: int = 1,
    balanced: bool = True,
    log_interval: int = 2000, verbose: bool = True,
) -> GoalConditionedActorNumpy:
    actor = GoalConditionedActorNumpy(
        state_dim=3, n_actions=n_actions, hidden=128, seed=seed,
        valid_actions=VALID_ACTIONS,
    )
    anchors = [(ei, t) for ei, tr in enumerate(episode_states) for t in range(len(tr) - 1)]

    # Balanced sampling over distinct (state, action) pairs (SUMMARY pathology #5,
    # "rare-route undertraining"). Without it, AWR clones in proportion to data
    # frequency, so the 85%-NEAR data drowns the critic's FAR advantage. With it,
    # each distinct (s,a) gets equal weight: pick a key uniformly, then a datapoint
    # within it — so the rare FAR-initiating (start, right) is sampled as often as
    # the common (start, forward). Mirrors the critic's bykey scheme.
    def _k3(s):
        return (int(round(s[0])), int(round(s[1])), int(round(s[2])))
    by_key: dict[tuple, list[tuple[int, int]]] = {}
    for ep, t in anchors:
        key = (_k3(episode_states[ep][t]), int(episode_actions[ep][t]))
        by_key.setdefault(key, []).append((ep, t))
    keys = list(by_key.keys())
    if verbose:
        mode = f"balanced over {len(keys)} (s,a) keys" if balanced else "uniform over anchors"
        print(f"[awr] sampling: {mode}  ({len(anchors)} anchors)")

    rng = np.random.default_rng(seed)
    losses: list[float] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        if balanced:
            ksel = [keys[int(i)] for i in rng.integers(0, len(keys), size=batch_size)]
            pairs = [by_key[k][int(rng.integers(0, len(by_key[k])))] for k in ksel]
        else:
            ids = rng.integers(0, len(anchors), size=batch_size)
            pairs = [anchors[int(i)] for i in ids]
        s = np.stack([episode_states[ep][t] for ep, t in pairs]).astype(np.float64)
        a = np.array([int(episode_actions[ep][t]) for ep, t in pairs], dtype=np.int64)
        sf = np.stack([
            episode_states[ep][t + _geom_offset(rng, len(episode_states[ep]) - t - 1)]
            for ep, t in pairs
        ]).astype(np.float64)

        # HER goal = sampled future state; goal encoder is dir-agnostic by design.
        scores = _critic_scores(critic, sa_index, goal_index, s, sf)
        q = scores[np.arange(batch_size), a]
        v = scores.mean(axis=1)
        adv = q - v
        w = np.clip(np.exp(np.clip(adv / beta, -10.0, 10.0)), 0.0, w_max)

        # Actor sees raw (x,y,dir); goal is the (dir-zeroed) future state.
        g_actor = sf.copy()
        g_actor[:, 2] = 0.0
        loss, grads = actor.awr_loss_and_grads(s, g_actor, a, w)
        actor.apply_sgd(grads, lr=lr)
        losses.append(loss)

        if verbose and (step % log_interval == 0 or step == 1):
            print(f"  step {step:6d}  awr_loss={np.mean(losses[-log_interval:]):.4f}  "
                  f"mean_w={w.mean():.3f}")

    if verbose:
        print(f"[awr] done in {time.time() - t0:.1f}s")
    return actor


# --------------------------------------------------------------------------- #
#  Real-env rollout evaluation                                                #
# --------------------------------------------------------------------------- #
def _classify_route(visited_ys: set[int]) -> str:
    return "far" if 13 in visited_ys else "near"


def eval_actor(actor, *, num_episodes: int, max_steps: int, seed: int,
               forced_wind: str | None, verbose: bool = True) -> dict:
    wind_dist = FORCED_WIND_PRESETS[forced_wind] if forced_wind else None
    goal = np.array([GOAL_POS[0], GOAL_POS[1], 0], dtype=np.float64)
    outcomes: Counter[str] = Counter()
    route_outcome = {"near": Counter(), "far": Counter()}
    for ep in range(num_episodes):
        env = make_windy_corridor_scm(wind_dist=wind_dist) if wind_dist else make_windy_corridor_scm()
        obs, _ = env.reset(seed=seed + ep)
        visited_ys: set[int] = set()
        terminated = truncated = False
        for _ in range(max_steps):
            state = np.array([int(obs[0]), int(obs[1]), int(env.agent_dir)], dtype=np.float64)
            visited_ys.add(int(state[1]))
            action = actor.greedy_action(state, goal, valid_actions=VALID_ACTIONS)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        final_pos = (int(obs[0]), int(obs[1]))
        if final_pos == GOAL_POS:
            outcome = "goal"
        elif terminated:
            outcome = "lava"
        elif truncated:
            outcome = "timeout"
        else:
            outcome = "incomplete"
        outcomes[outcome] += 1
        route_outcome[_classify_route(visited_ys)][outcome] += 1
    if verbose:
        far = sum(route_outcome["far"].values())
        print(f"  success={outcomes['goal']/num_episodes:.2%}  "
              f"lava={outcomes['lava']/num_episodes:.2%}  "
              f"timeout={outcomes.get('timeout',0)/num_episodes:.2%}  "
              f"incomplete={outcomes.get('incomplete',0)/num_episodes:.2%}")
        print(f"  FAR rate: {far}/{num_episodes} = {far/num_episodes:.0%}")
    return {"outcomes": dict(outcomes), "routes": {k: dict(v) for k, v in route_outcome.items()}}


def save_actor(actor, output_path: Path, config: dict) -> None:
    sd = actor.state_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        actor_W1=sd["actor_W1"], actor_b1=sd["actor_b1"],
        actor_W2=sd["actor_W2"], actor_b2=sd["actor_b2"],
        config_json=np.array(json.dumps(config)),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train one-hot AWR actor on corridor worst-case data.")
    p.add_argument("--critic", type=Path, default=ROOT / "checkpoints" / "corridor_onehot_seed0.npz")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "corridor_worst_case_n1000.npz")
    p.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "corridor_onehot_awr_seed0.npz")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--actor-lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--balanced", dest="balanced", action="store_true", default=True,
                   help="balanced sampling over distinct (s,a) pairs (default on)")
    p.add_argument("--no-balanced", dest="balanced", action="store_false",
                   help="uniform sampling over anchors (frequency-weighted, old behavior)")
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()

    critic, sa_index, goal_index, ccfg = load_onehot_critic(args.critic)
    data = np.load(args.data, allow_pickle=True)
    episode_states = [np.asarray(s, dtype=np.float64) for s in data["episodes_states"]]
    episode_actions = [np.asarray(a, dtype=np.int64) for a in data["episodes_actions"]]
    n_actions = int(data["n_actions"])

    print(f"[awr] critic={args.critic.name}  data={args.data.name}  "
          f"episodes={len(episode_states)}  beta={args.beta}  steps={args.steps}")
    actor = train_awr(
        critic=critic, sa_index=sa_index, goal_index=goal_index,
        episode_states=episode_states, episode_actions=episode_actions,
        n_actions=n_actions, beta=args.beta, steps=args.steps,
        lr=args.actor_lr, seed=args.seed, balanced=args.balanced,
    )

    config = {
        "critic_checkpoint": str(args.critic), "data_path": str(args.data),
        "beta": args.beta, "steps": args.steps, "actor_lr": args.actor_lr,
        "seed": args.seed, "gamma": GAMMA, "valid_actions": VALID_ACTIONS,
        "balanced": args.balanced,
        "encoding": "raw_state_onehot_critic_awr",
    }
    save_actor(actor, args.output, config)
    print(f"[awr] checkpoint -> {args.output}\n")

    print(f"[AWR-actor eval] N={args.eval_episodes}  start={START_POS} goal={GOAL_POS}")
    for regime in ["natural", "south", "still"]:
        print(f"[regime={regime}]")
        eval_actor(
            actor, num_episodes=args.eval_episodes, max_steps=args.max_steps,
            seed=args.seed, forced_wind=None if regime == "natural" else regime,
            verbose=True,
        )
        print()


if __name__ == "__main__":
    main()
