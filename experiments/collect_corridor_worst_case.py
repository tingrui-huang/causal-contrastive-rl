r"""
Collect worst-case trajectories under $\underline{T}_\pi$ (Theorem 2 revised).

For each episode:
  S_0 = start state (1, 1, east)
  X_0 ~ pi(. | S_0)
  Repeat:
    (ii)  w.p. P_b(X_t | S_t):    S_{t+1} ~ T_e(. | S_t, X_t)    [observational]
    (iii) w.p. 1 - P_b(X_t | S_t): S_{t+1} = argmin V_lower over N(S_t, X_t)
    X_{t+1} ~ pi(. | S_{t+1})

Output schema matches collect_corridor_data.py so train_corridor.py can consume
it unchanged: ``episodes_states / _actions / _rewards / _winds / _routes /
_outcomes`` + scalars ``goal_state, state_dim, n_actions, n_episodes, p_near``.

Two policy options for pi:
  * --policy expert     : CorridorExpertPolicy with p_near (wind-blind; we feed
                          wind=4 since pi has no wind input in this framework)
  * --policy actor      : load a trained checkpoint (e.g. corridor baseline) and
                          sample from its action_probs

Usage::
    python experiments/collect_corridor_worst_case.py \
        --obs-data data/corridor_expert_n1000.npz \
        --policy actor --actor-checkpoint checkpoints/corridor_smoke.npz \
        --num-episodes 1000 --seed 0 \
        --output data/corridor_worst_case_n1000.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from minigrid.core.actions import Actions

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy  # noqa: F401 (kept for checkpoint format)
from agents.expert_policy import CorridorExpertPolicy
from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy
from configs import corridor_defaults as C
from envs.windy_corridor import GOAL_POS, START_POS
from utils.obs_transition import estimate_obs_transition, sample_obs_transition
from utils.propensity import estimate_propensity, lookup
from utils.v_lower_oracle import v_lower
from utils.wind_dynamics import is_goal, is_lava
from utils.worst_case_kernel import WorstCaseKernel


N_ACTIONS = 7
STATE_DIM = 3
GOAL_STATE_DIR = 0
START_DIR = 0
VALID_ACTIONS = [
    int(Actions.left),
    int(Actions.right),
    int(Actions.forward),
    int(Actions.done),
]
WIND_DUMMY = 4  # "still" — used when pi needs a wind input but no real wind was drawn


def _load_actor(checkpoint_path: Path) -> GoalConditionedActorNumpy:
    ckpt = np.load(checkpoint_path, allow_pickle=True)
    cfg = json.loads(str(ckpt["config_json"]))
    actor = GoalConditionedActorNumpy(
        state_dim=int(cfg["state_dim"]),
        n_actions=int(cfg["n_actions"]),
        hidden=int(cfg["hidden"]),
        seed=int(cfg["seed"]) + 1,
    )
    actor.W1[...] = ckpt["actor_W1"]; actor.b1[...] = ckpt["actor_b1"]
    actor.W2[...] = ckpt["actor_W2"]; actor.b2[...] = ckpt["actor_b2"]
    return actor


def _make_policy_step(
    policy_name: str,
    *,
    expert: CorridorExpertPolicy | None,
    actor: GoalConditionedActorNumpy | None,
    goal_state: np.ndarray,
    temperature: float,
):
    """Return ``policy_step(state, rng) -> action``."""
    if policy_name == "expert":
        assert expert is not None
        def step(state, rng):
            pos = (int(state[0]), int(state[1]))
            return int(expert(pos, WIND_DUMMY))
        return step
    assert actor is not None
    def step(state, rng):
        return int(actor.sample_action(
            np.asarray(state, dtype=np.float64),
            goal_state,
            valid_actions=VALID_ACTIONS,
            rng=rng,
            temperature=temperature,
        ))
    return step


def _rollout_episode(
    *,
    kernel: WorstCaseKernel,
    policy_step,
    expert: CorridorExpertPolicy | None,
    rng: np.random.Generator,
    max_steps: int,
) -> dict:
    state = (START_POS[0], START_POS[1], START_DIR)
    if expert is not None:
        expert.reset(seed=int(rng.integers(0, 2**31 - 1)), agent_dir=START_DIR)

    states: list[list[int]] = [[state[0], state[1], state[2]]]
    actions: list[int] = []
    rewards: list[float] = []
    winds: list[int] = []
    branches: list[str] = []

    terminated = False
    for _ in range(max_steps):
        action = policy_step(state, rng)
        next_state, term, wind, branch = kernel.step(state, action, rng)

        actions.append(int(action))
        winds.append(int(wind))
        branches.append(branch)
        # Reward: 0 on goal, -1 on lava, 0 elsewhere; per-step penalty -0.1
        # (matches WindyMiniGridSCM._action_sequence broadly; sufficient for our
        # training pipeline which doesn't use reward in the contrastive loss)
        if is_goal((next_state[0], next_state[1])):
            r = -0.1
        elif is_lava((next_state[0], next_state[1])):
            r = -1.1
        else:
            r = -0.1
        rewards.append(float(r))

        states.append([next_state[0], next_state[1], next_state[2]])
        state = next_state
        if term:
            terminated = True
            break

    final_pos = (state[0], state[1])
    if is_goal(final_pos):
        outcome = "goal"
    elif is_lava(final_pos):
        outcome = "lava"
    elif not terminated:
        outcome = "timeout"
    else:
        outcome = "incomplete"

    route = "far" if any(s[1] == 13 for s in states) else "near"

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "winds": np.array(winds, dtype=np.int64),
        "route": route,
        "outcome": outcome,
        "branches": branches,
    }


def collect(
    *,
    obs_data_path: Path,
    policy_name: str,
    actor_checkpoint: Path | None,
    num_episodes: int,
    max_steps: int,
    seed: int,
    p_near: float,
    temperature: float,
    propensity_default: float,
    obs_branch: str = "data",
    verbose: bool = True,
) -> dict:
    obs = np.load(obs_data_path, allow_pickle=True)
    obs_states = list(obs["episodes_states"])
    obs_actions = list(obs["episodes_actions"])
    propensity_table = estimate_propensity(
        obs_states, obs_actions, n_actions=int(obs["n_actions"]),
    )
    if verbose:
        print(f"[Collect] propensity table: {len(propensity_table)} unique states")

    def propensity_fn(state, action):
        return lookup(propensity_table, state, action, default_p=propensity_default)

    # (i) branch: replay empirical P_obs(s'|s,x) from the observational data
    # (the theoretically correct observational transition). obs_branch="analytical"
    # restores the legacy marginal-wind roll for comparison.
    obs_transition_fn = None
    if obs_branch == "data":
        obs_trans_table = estimate_obs_transition(obs_states, obs_actions)
        if verbose:
            print(f"[Collect] P_obs table: {len(obs_trans_table)} unique (s,a) pairs")

        def obs_transition_fn(state, action, rng):  # noqa: F811
            return sample_obs_transition(obs_trans_table, state, action, rng)

    kernel = WorstCaseKernel(
        propensity_fn=propensity_fn,
        v_lower_fn=v_lower,
        obs_transition_fn=obs_transition_fn,
    )

    expert = CorridorExpertPolicy(p_near=p_near) if policy_name == "expert" else None
    actor = None
    goal_state = np.array(
        [GOAL_POS[0], GOAL_POS[1], GOAL_STATE_DIR], dtype=np.float64
    )
    if policy_name == "actor":
        assert actor_checkpoint is not None
        actor = _load_actor(actor_checkpoint)

    policy_step = _make_policy_step(
        policy_name, expert=expert, actor=actor,
        goal_state=goal_state, temperature=temperature,
    )

    rng = np.random.default_rng(seed)
    states_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[np.ndarray] = []
    winds_list: list[np.ndarray] = []
    routes: list[str] = []
    outcomes: list[str] = []
    branch_counter: Counter[str] = Counter()
    outcome_counter: Counter[str] = Counter()
    route_counter: Counter[str] = Counter()
    route_outcome: dict[str, Counter[str]] = {"near": Counter(), "far": Counter()}

    for ep in range(num_episodes):
        traj = _rollout_episode(
            kernel=kernel,
            policy_step=policy_step,
            expert=expert,
            rng=rng,
            max_steps=max_steps,
        )
        states_list.append(traj["states"])
        actions_list.append(traj["actions"])
        rewards_list.append(traj["rewards"])
        winds_list.append(traj["winds"])
        routes.append(traj["route"])
        outcomes.append(traj["outcome"])
        for b in traj["branches"]:
            branch_counter[b] += 1
        outcome_counter[traj["outcome"]] += 1
        route_counter[traj["route"]] += 1
        if traj["route"] in route_outcome:
            route_outcome[traj["route"]][traj["outcome"]] += 1

    total_transitions = sum(len(a) for a in actions_list)

    if verbose:
        print(f"[Collect] {num_episodes} episodes ({total_transitions} transitions)")
        print(f"  branches: {dict(branch_counter)}")
        print(f"  routes:   {dict(route_counter)}")
        print(f"  outcomes: {dict(outcome_counter)}")
        for r, c in route_outcome.items():
            print(f"    {r:<4}: {dict(c)}")

    return {
        "states_list": states_list,
        "actions_list": actions_list,
        "rewards_list": rewards_list,
        "winds_list": winds_list,
        "routes": routes,
        "outcomes": outcomes,
        "total_transitions": total_transitions,
    }


def save_npz(out: dict, output_path: Path, *, p_near: float, num_episodes: int) -> None:
    goal_state = np.array([GOAL_POS[0], GOAL_POS[1], GOAL_STATE_DIR], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        episodes_states=np.array(out["states_list"], dtype=object),
        episodes_actions=np.array(out["actions_list"], dtype=object),
        episodes_rewards=np.array(out["rewards_list"], dtype=object),
        episodes_winds=np.array(out["winds_list"], dtype=object),
        episodes_routes=np.array(out["routes"], dtype=object),
        episodes_outcomes=np.array(out["outcomes"], dtype=object),
        goal_state=goal_state,
        state_dim=np.array(STATE_DIM),
        n_actions=np.array(N_ACTIONS),
        n_episodes=np.array(num_episodes),
        p_near=np.array(p_near, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect WindyCorridor worst-case data.")
    parser.add_argument("--obs-data", type=Path, default=ROOT / "data" / "corridor_expert_n1000.npz",
                        help="Observational dataset used to estimate P_b(x|s).")
    parser.add_argument("--policy", choices=["expert", "actor"], default="actor")
    parser.add_argument("--actor-checkpoint", type=Path,
                        default=ROOT / "checkpoints" / "corridor_smoke.npz")
    parser.add_argument("--num-episodes", type=int, default=C.NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--p-near", type=float, default=C.P_NEAR,
                        help="Only used when --policy expert.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature when --policy actor.")
    parser.add_argument("--propensity-default", type=float, default=C.PROPENSITY_DEFAULT,
                        help="P_b(x|s) fallback for unseen states (low = more (ii) firings).")
    parser.add_argument("--obs-branch", choices=["data", "analytical"], default="data",
                        help="(i) branch transition: 'data' replays empirical P_obs(s'|s,x) "
                             "(correct); 'analytical' re-rolls wind from WIND_DIST (legacy/D1-buggy).")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "corridor_worst_case_n1000.npz")
    args = parser.parse_args()

    out = collect(
        obs_data_path=args.obs_data,
        policy_name=args.policy,
        actor_checkpoint=args.actor_checkpoint if args.policy == "actor" else None,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        p_near=args.p_near,
        temperature=args.temperature,
        propensity_default=args.propensity_default,
        obs_branch=args.obs_branch,
    )
    save_npz(out, args.output, p_near=args.p_near, num_episodes=args.num_episodes)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
