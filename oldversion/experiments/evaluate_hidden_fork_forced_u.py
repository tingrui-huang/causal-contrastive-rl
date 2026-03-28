"""
Forced-regime policy evaluation for HiddenFork checkpoints.

The evaluator uses the same policy-extraction rule for every compared method:
- action subset fixed once for the whole run
- build a successful oracle reference trajectory for each fixed-U regime
- at each state, align to a reference-trajectory position
- score each action by similarity to the reference state's k-step future window

This matches the training setup more closely than terminal goal-bank scoring.
By default, reference alignment is done by episode time-step (t -> t+k),
which is more stable than nearest-observation matching in the early aliased
HiddenFork observations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs  # noqa: F401

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from utils.offline_data import build_policy
from utils.contrastive_sampling import DEFAULT_K
from utils.preprocess import extract_state

ENV_CONF = "CausalContrastive-HiddenForkHiddenTrap-15x15-v0"
ACTION_NAME_TO_ID = {
    "left": 0,
    "right": 1,
    "forward": 2,
}
ACTION_ID_TO_NAME = {v: k for k, v in ACTION_NAME_TO_ID.items()}

CSV_FIELDS = [
    "method",
    "loss_family",
    "checkpoint",
    "seed",
    "env_id",
    "eval_env_id",
    "collector_mode",
    "num_episodes",
    "num_train_steps",
    "w",
    "goal_mode",
    "action_subset",
    "goal_bank_size",
    "episodes_per_regime",
    "success_u0",
    "success_u1",
    "mean_success",
    "worst_case_success",
    "regime_gap",
    "fork_visit_rate_u0",
    "fork_action_prob_left_u0",
    "fork_action_prob_right_u0",
    "fork_action_prob_forward_u0",
    "fork_visit_rate_u1",
    "fork_action_prob_left_u1",
    "fork_action_prob_right_u1",
    "fork_action_prob_forward_u1",
]


def _load_numpy_checkpoint(path: Path) -> tuple[ContrastiveCriticNumpy, dict[str, Any]]:
    ckpt = np.load(path, allow_pickle=True)
    state_dim = int(ckpt["state_dim"])
    n_actions = int(ckpt["n_actions"])
    tau = float(ckpt["tau"])
    hidden = int(ckpt["W1"].shape[1])
    emb_dim = int(ckpt["W2"].shape[1])
    model = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=hidden,
        emb_dim=emb_dim,
        tau=tau,
        seed=0,
    )
    model.W1[...] = ckpt["W1"]
    model.b1[...] = ckpt["b1"]
    model.W2[...] = ckpt["W2"]
    model.b2[...] = ckpt["b2"]

    cfg = json.loads(str(ckpt["train_config_json"]))
    meta = {
        "method": str(ckpt["method"]) if "method" in ckpt else str(cfg.get("method", "unknown")),
        "loss_family": str(ckpt["loss_family"]) if "loss_family" in ckpt else str(cfg.get("loss_family", "unknown")),
        "w": float(ckpt["w"]) if "w" in ckpt else cfg.get("w"),
        "train_config": cfg,
    }
    return model, meta


def _reference_trajectory_from_fixed_u(
    *,
    env_id: str,
    fixed_u: int,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    env = gym.make(env_id, render_mode="rgb_array", fixed_u=fixed_u)
    env.action_space.seed(seed)
    policy = build_policy(
        env,
        collector_mode="oracle_eps",
        oracle_epsilon=0.0,
        seed=seed,
    )
    obs, _ = env.reset(seed=seed)
    states = [extract_state(obs)]
    positions = [tuple(int(v) for v in env.unwrapped.agent_pos)]
    for _ in range(env.unwrapped.max_steps):
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        states.append(extract_state(next_obs))
        positions.append(tuple(int(v) for v in env.unwrapped.agent_pos))
        obs = next_obs
        if terminated or truncated:
            break
    return np.stack(states, axis=0).astype(np.float64, copy=False), positions


def _score_action(
    model: ContrastiveCriticNumpy,
    state: np.ndarray,
    action_id: int,
    future_bank: np.ndarray,
) -> float:
    bank_count = future_bank.shape[0]
    s_batch = np.repeat(state[None, :], bank_count, axis=0)
    a_batch = np.full(bank_count, action_id, dtype=np.int64)
    h_sa, _ = model._embed(s_batch, a_batch)
    h_goal, _ = model._embed(future_bank, a_batch)
    return float(np.max(np.sum(h_sa * h_goal, axis=1)))


def _future_window_bank(
    ref_states: np.ndarray,
    *,
    ref_idx: int,
    lookahead_k: int,
    future_window: int,
) -> np.ndarray:
    start = min(ref_idx + lookahead_k, len(ref_states) - 1)
    stop = min(start + future_window + 1, len(ref_states))
    return ref_states[start:stop]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _empty_fork_stats() -> dict[str, float]:
    return {
        "fork_visit_rate": 0.0,
        "fork_action_prob_left": 0.0,
        "fork_action_prob_right": 0.0,
        "fork_action_prob_forward": 0.0,
    }


def _stack_states_by_positions(
    ref_states: np.ndarray,
    ref_positions: list[tuple[int, int]],
    *,
    predicate,
) -> np.ndarray:
    selected = [state for state, pos in zip(ref_states, ref_positions) if predicate(pos)]
    if not selected:
        raise RuntimeError("No reference states matched the requested shared-goal predicate.")
    return np.stack(selected, axis=0).astype(np.float64, copy=False)


def _shared_goal_banks(
    *,
    ref_states_u0: np.ndarray,
    ref_positions_u0: list[tuple[int, int]],
    ref_states_u1: np.ndarray,
    ref_positions_u1: list[tuple[int, int]],
    cx: int,
    fork_row: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    decision_cell = (cx, fork_row + 1)
    decision_bank = _stack_states_by_positions(
        ref_states_u0,
        ref_positions_u0,
        predicate=lambda pos: pos == decision_cell,
    )
    merge_bank_u0 = _stack_states_by_positions(
        ref_states_u0,
        ref_positions_u0,
        predicate=lambda pos: pos[0] == cx and pos[1] <= fork_row - 1,
    )
    merge_bank_u1 = _stack_states_by_positions(
        ref_states_u1,
        ref_positions_u1,
        predicate=lambda pos: pos[0] == cx and pos[1] <= fork_row - 1,
    )
    merge_bank = np.concatenate([merge_bank_u0, merge_bank_u1], axis=0)
    return decision_bank, merge_bank, decision_cell


def _reference_index(
    *,
    step_idx: int,
    state: np.ndarray,
    ref_states: np.ndarray,
    alignment_mode: str,
) -> int:
    if alignment_mode == "time":
        return min(step_idx, len(ref_states) - 1)
    if alignment_mode == "nearest":
        d2 = np.sum((ref_states - state[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))
    raise ValueError(f"Unknown alignment_mode: {alignment_mode!r}")


def _run_regime(
    *,
    model: ContrastiveCriticNumpy,
    eval_env_id: str,
    fixed_u: int,
    episodes: int,
    action_ids: list[int],
    ref_states: np.ndarray,
    ref_positions: list[tuple[int, int]],
    decision_bank: np.ndarray | None,
    merge_bank: np.ndarray | None,
    decision_cell: tuple[int, int] | None,
    fork_row: int,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_mode: str,
    plan_depth: int,
    max_eval_steps: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
    debug: bool,
    debug_max_steps: int,
) -> dict[str, float]:
    successes = 0
    fork_action_counts = {"left": 0, "right": 0, "forward": 0}
    fork_reached_episodes = 0
    for ep in range(episodes):
        env = gym.make(eval_env_id, render_mode="rgb_array", fixed_u=fixed_u)
        obs, _ = env.reset(seed=ep)
        done = False
        step_idx = 0
        counted_fork_action = False
        while not done and step_idx < max_eval_steps:
            raw = env.unwrapped
            state = extract_state(obs).astype(np.float64, copy=False)
            ref_idx = _reference_index(
                step_idx=step_idx,
                state=state,
                ref_states=ref_states,
                alignment_mode=alignment_mode,
            )
            pos = tuple(int(v) for v in raw.agent_pos)
            if goal_mode == "open_book":
                future_bank = _future_window_bank(
                    ref_states,
                    ref_idx=ref_idx,
                    lookahead_k=lookahead_k,
                    future_window=future_window,
                )
            elif goal_mode == "merge_shared":
                if decision_bank is None or merge_bank is None or decision_cell is None:
                    raise RuntimeError("merge_shared goal mode requires shared decision/merge banks.")
                if pos[1] > fork_row + 1:
                    future_bank = decision_bank
                else:
                    future_bank = merge_bank
            else:
                raise ValueError(f"Unknown goal_mode: {goal_mode!r}")
            scores = []
            best_sequences: list[tuple[int, ...]] = []
            for action_id in action_ids:
                best_value = -np.inf
                best_seq = (action_id,)
                for suffix in itertools.product(action_ids, repeat=max(0, plan_depth - 1)):
                    seq = (action_id, *suffix)
                    sim_env = copy.deepcopy(env)
                    sim_obs = obs
                    seq_value = 0.0
                    for depth_idx, seq_action in enumerate(seq):
                        sim_raw_before = sim_env.unwrapped
                        pos_before = tuple(int(v) for v in sim_raw_before.agent_pos)
                        sim_state = extract_state(sim_obs).astype(np.float64, copy=False)
                        sim_ref_idx = _reference_index(
                            step_idx=step_idx + depth_idx,
                            state=sim_state,
                            ref_states=ref_states,
                            alignment_mode=alignment_mode,
                        )
                        sim_fork_row = int(getattr(sim_raw_before, "_fork_row"))
                        if goal_mode == "open_book":
                            sim_future_bank = _future_window_bank(
                                ref_states,
                                ref_idx=sim_ref_idx,
                                lookahead_k=lookahead_k,
                                future_window=future_window,
                            )
                            sim_progress_target_pos: tuple[int, int] | None = ref_positions[
                                min(sim_ref_idx + lookahead_k, len(ref_positions) - 1)
                            ]
                        elif goal_mode == "merge_shared":
                            if decision_bank is None or merge_bank is None or decision_cell is None:
                                raise RuntimeError("merge_shared goal mode requires shared decision/merge banks.")
                            if pos_before[1] > sim_fork_row + 1:
                                sim_future_bank = decision_bank
                                sim_progress_target_pos = decision_cell
                            else:
                                sim_future_bank = merge_bank
                                sim_progress_target_pos = None
                        else:
                            raise ValueError(f"Unknown goal_mode: {goal_mode!r}")
                        seq_value += _score_action(model, sim_state, seq_action, sim_future_bank)
                        sim_next_obs, _, sim_terminated, sim_truncated, _ = sim_env.step(seq_action)
                        sim_raw_after = sim_env.unwrapped
                        pos_after = tuple(int(v) for v in sim_raw_after.agent_pos)
                        if sim_progress_target_pos is not None:
                            seq_value += progress_bonus * float(
                                _manhattan(pos_before, sim_progress_target_pos)
                                - _manhattan(pos_after, sim_progress_target_pos)
                            )
                        if seq_action == ACTION_NAME_TO_ID["forward"] and pos_after == pos_before:
                            seq_value -= collision_penalty
                        elif seq_action in (ACTION_NAME_TO_ID["left"], ACTION_NAME_TO_ID["right"]):
                            seq_value -= turn_penalty
                        if sim_terminated:
                            seq_value += success_bonus
                        sim_obs = sim_next_obs
                        if sim_terminated or sim_truncated:
                            break
                    if seq_value > best_value:
                        best_value = seq_value
                        best_seq = seq
                scores.append(float(best_value))
                best_sequences.append(best_seq)
            best_idx = int(np.argmax(scores))
            action = action_ids[best_idx]
            decision_cell = (int(raw.width // 2), int(raw._fork_row) + 1)
            pos = tuple(int(v) for v in raw.agent_pos)
            if not counted_fork_action and pos == decision_cell:
                fork_reached_episodes += 1
                fork_action_counts[ACTION_ID_TO_NAME.get(action, str(action))] += 1
                counted_fork_action = True
            if debug and ep == 0 and len(scores) > 0 and debug_max_steps > 0:
                debug_step = getattr(_run_regime, "_debug_step", 0)
                if debug_step < debug_max_steps:
                    direction = int(raw.agent_dir)
                    hidden_u = int(getattr(raw, "hidden_u", -1))
                    pre_fork = pos[1] > fork_row
                    at_fork = pos[1] == fork_row
                    print(
                        f"[EvalDebug] fixed_u={fixed_u} ep={ep} step={step_idx} "
                        f"pos={pos} dir={direction} fork_row={fork_row} "
                        f"hidden_u={hidden_u} pre_fork={pre_fork} at_fork={at_fork} "
                        f"ref_idx={ref_idx} align={alignment_mode} goal_mode={goal_mode} "
                        f"scores={scores} chosen={ACTION_ID_TO_NAME.get(action, action)} "
                        f"best_seq={[ACTION_ID_TO_NAME.get(a, a) for a in best_sequences[best_idx]]}"
                    )
                    setattr(_run_regime, "_debug_step", debug_step + 1)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if debug and ep == 0 and getattr(_run_regime, "_debug_step", 0) <= debug_max_steps:
                next_raw = env.unwrapped
                next_pos = tuple(int(v) for v in next_raw.agent_pos)
                print(
                    f"[EvalDebugStep] fixed_u={fixed_u} ep={ep} step={step_idx} "
                    f"action={ACTION_ID_TO_NAME.get(action, action)} next_pos={next_pos} "
                    f"reward={float(reward):.1f} terminated={terminated} truncated={truncated}"
                )
            if terminated and reward > 0:
                successes += 1
            obs = next_obs
            step_idx += 1
        if debug:
            setattr(_run_regime, "_debug_step", 0)
    success = successes / float(episodes)
    if fork_reached_episodes == 0:
        return {"success": success, **_empty_fork_stats()}
    fork_total = float(fork_reached_episodes)
    return {
        "success": success,
        "fork_visit_rate": fork_total / float(episodes),
        "fork_action_prob_left": fork_action_counts["left"] / fork_total,
        "fork_action_prob_right": fork_action_counts["right"] / fork_total,
        "fork_action_prob_forward": fork_action_counts["forward"] / fork_total,
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    eval_env_id: str,
    episodes_per_regime: int,
    action_ids: list[int],
    action_subset_label: str,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_mode: str,
    plan_depth: int,
    max_eval_steps: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
    debug: bool,
    debug_max_steps: int,
) -> dict[str, object]:
    model, meta = _load_numpy_checkpoint(checkpoint_path)
    cfg = meta["train_config"]
    geom_env = gym.make(eval_env_id, render_mode="rgb_array", fixed_u=0)
    geom_env.reset(seed=0)
    cx = int(geom_env.unwrapped.width // 2)
    fork_row = int(geom_env.unwrapped._fork_row)
    geom_env.close()

    ref_states_u0, ref_positions_u0 = _reference_trajectory_from_fixed_u(
        env_id=eval_env_id,
        fixed_u=0,
        seed=0,
    )
    ref_states_u1, ref_positions_u1 = _reference_trajectory_from_fixed_u(
        env_id=eval_env_id,
        fixed_u=1,
        seed=0,
    )
    decision_bank = None
    merge_bank = None
    decision_cell = None
    if goal_mode == "merge_shared":
        decision_bank, merge_bank, decision_cell = _shared_goal_banks(
            ref_states_u0=ref_states_u0,
            ref_positions_u0=ref_positions_u0,
            ref_states_u1=ref_states_u1,
            ref_positions_u1=ref_positions_u1,
            cx=cx,
            fork_row=fork_row,
        )

    regime_u0 = _run_regime(
        model=model,
        eval_env_id=eval_env_id,
        fixed_u=0,
        episodes=episodes_per_regime,
        action_ids=action_ids,
        ref_states=ref_states_u0,
        ref_positions=ref_positions_u0,
        decision_bank=decision_bank,
        merge_bank=merge_bank,
        decision_cell=decision_cell,
        fork_row=fork_row,
        lookahead_k=lookahead_k,
        future_window=future_window,
        alignment_mode=alignment_mode,
        goal_mode=goal_mode,
        plan_depth=plan_depth,
        max_eval_steps=max_eval_steps,
        collision_penalty=collision_penalty,
        turn_penalty=turn_penalty,
        progress_bonus=progress_bonus,
        success_bonus=success_bonus,
        debug=debug,
        debug_max_steps=debug_max_steps,
    )
    regime_u1 = _run_regime(
        model=model,
        eval_env_id=eval_env_id,
        fixed_u=1,
        episodes=episodes_per_regime,
        action_ids=action_ids,
        ref_states=ref_states_u1,
        ref_positions=ref_positions_u1,
        decision_bank=decision_bank,
        merge_bank=merge_bank,
        decision_cell=decision_cell,
        fork_row=fork_row,
        lookahead_k=lookahead_k,
        future_window=future_window,
        alignment_mode=alignment_mode,
        goal_mode=goal_mode,
        plan_depth=plan_depth,
        max_eval_steps=max_eval_steps,
        collision_penalty=collision_penalty,
        turn_penalty=turn_penalty,
        progress_bonus=progress_bonus,
        success_bonus=success_bonus,
        debug=debug,
        debug_max_steps=debug_max_steps,
    )
    success_u0 = regime_u0["success"]
    success_u1 = regime_u1["success"]
    mean_success = 0.5 * (success_u0 + success_u1)
    worst_case_success = min(success_u0, success_u1)
    regime_gap = abs(success_u0 - success_u1)

    return {
        "method": meta["method"],
        "loss_family": meta["loss_family"],
        "checkpoint": str(checkpoint_path),
        "seed": cfg.get("seed"),
        "env_id": cfg.get("env_id"),
        "eval_env_id": eval_env_id,
        "collector_mode": cfg.get("collector_mode"),
        "num_episodes": cfg.get("num_episodes"),
        "num_train_steps": cfg.get("num_train_steps"),
        "w": meta["w"],
        "goal_mode": goal_mode,
        "action_subset": action_subset_label,
        "goal_bank_size": (future_window + 1) if goal_mode == "open_book" else int(merge_bank.shape[0]),
        "episodes_per_regime": episodes_per_regime,
        "success_u0": success_u0,
        "success_u1": success_u1,
        "mean_success": mean_success,
        "worst_case_success": worst_case_success,
        "regime_gap": regime_gap,
        "fork_visit_rate_u0": regime_u0["fork_visit_rate"],
        "fork_action_prob_left_u0": regime_u0["fork_action_prob_left"],
        "fork_action_prob_right_u0": regime_u0["fork_action_prob_right"],
        "fork_action_prob_forward_u0": regime_u0["fork_action_prob_forward"],
        "fork_visit_rate_u1": regime_u1["fork_visit_rate"],
        "fork_action_prob_left_u1": regime_u1["fork_action_prob_left"],
        "fork_action_prob_right_u1": regime_u1["fork_action_prob_right"],
        "fork_action_prob_forward_u1": regime_u1["fork_action_prob_forward"],
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Forced-U evaluation for HiddenTrap HiddenFork checkpoints")
    p.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit checkpoint paths (.npz)",
    )
    p.add_argument(
        "--checkpoint-glob",
        type=str,
        default=None,
        help="Optional glob under checkpoints/, e.g. 'robust_v1_seed0_*.npz'",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_hidden_trap_forced_u_eval.csv",
    )
    p.add_argument("--eval-env-id", type=str, default=ENV_CONF)
    p.add_argument("--episodes-per-regime", type=int, default=100)
    p.add_argument(
        "--action-subset",
        type=str,
        default="left,right,forward",
        help="Comma-separated subset from {left,right,forward}",
    )
    p.add_argument(
        "--lookahead-k",
        type=int,
        default=DEFAULT_K,
        help="Reference-trajectory lookahead offset used for action scoring",
    )
    p.add_argument(
        "--future-window",
        type=int,
        default=2,
        help="Extra future candidates after t+k (0 => only exact t+k)",
    )
    p.add_argument(
        "--alignment-mode",
        type=str,
        default="time",
        choices=["time", "nearest"],
        help="How to align current state with the oracle reference trajectory",
    )
    p.add_argument(
        "--goal-mode",
        type=str,
        default="open_book",
        choices=["open_book", "merge_shared"],
        help="Use branch-specific future targets or a shared merge target near the fork",
    )
    p.add_argument(
        "--plan-depth",
        type=int,
        default=1,
        help="Number of greedy-planning steps to score before taking the first action",
    )
    p.add_argument(
        "--max-eval-steps",
        type=int,
        default=60,
        help="Hard cap on executed steps per evaluation episode",
    )
    p.add_argument(
        "--collision-penalty",
        type=float,
        default=3.0,
        help="Penalty for forward actions that keep the agent in the same cell",
    )
    p.add_argument(
        "--turn-penalty",
        type=float,
        default=0.05,
        help="Small penalty for left/right turns during multi-step planning",
    )
    p.add_argument(
        "--progress-bonus",
        type=float,
        default=0.75,
        help="Bonus per cell of Manhattan progress toward the aligned reference future position",
    )
    p.add_argument(
        "--success-bonus",
        type=float,
        default=5.0,
        help="Bonus for simulated sequences that already reach the goal",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print step-level action scores for the first episode of each regime",
    )
    p.add_argument(
        "--debug-max-steps",
        type=int,
        default=20,
        help="Maximum debug steps to print per regime when --debug is set",
    )
    p.add_argument("--stdout-only", action="store_true")
    args = p.parse_args()

    ckpts: list[Path] = []
    if args.checkpoints:
        ckpts.extend(args.checkpoints)
    if args.checkpoint_glob:
        ckpts.extend(sorted((ROOT / "checkpoints").glob(args.checkpoint_glob)))
    if not ckpts:
        raise SystemExit("No checkpoints provided. Use --checkpoints or --checkpoint-glob.")

    action_names = [part.strip() for part in args.action_subset.split(",") if part.strip()]
    unknown = [name for name in action_names if name not in ACTION_NAME_TO_ID]
    if unknown:
        raise SystemExit(f"Unknown action names: {unknown}")
    action_ids = [ACTION_NAME_TO_ID[name] for name in action_names]
    action_subset_label = ",".join(action_names)

    rows = [
        evaluate_checkpoint(
            checkpoint_path=ckpt,
            eval_env_id=args.eval_env_id,
            episodes_per_regime=args.episodes_per_regime,
            action_ids=action_ids,
            action_subset_label=action_subset_label,
            lookahead_k=args.lookahead_k,
            future_window=args.future_window,
            alignment_mode=args.alignment_mode,
            goal_mode=args.goal_mode,
            plan_depth=args.plan_depth,
            max_eval_steps=args.max_eval_steps,
            collision_penalty=args.collision_penalty,
            turn_penalty=args.turn_penalty,
            progress_bonus=args.progress_bonus,
            success_bonus=args.success_bonus,
            debug=args.debug,
            debug_max_steps=args.debug_max_steps,
        )
        for ckpt in ckpts
    ]

    if not args.stdout_only:
        _write_csv(rows, args.output)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()
