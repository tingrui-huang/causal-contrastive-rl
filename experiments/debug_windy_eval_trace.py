"""Trace one evaluation episode step-by-step for rollout vs forced-U planners."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.core.actions import Actions

import envs  # noqa: F401

import experiments.evaluate_windy_corridor_forced_u as forced_eval
import experiments.evaluate_windy_corridor_rollout as rollout_eval


ACTION_NAME = {
    int(Actions.left): "left",
    int(Actions.right): "right",
    int(Actions.forward): "forward",
    int(Actions.done): "done",
}


def _trace_rollout(
    *,
    checkpoint: Path,
    eval_env_id: str,
    seed: int,
    max_steps: int,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_bank_mode: str,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
) -> None:
    model, _ = rollout_eval._load_numpy_checkpoint(checkpoint)
    ref_states, ref_positions = rollout_eval._reference_trajectory(env_id=eval_env_id, seed=seed)

    env = gym.make(eval_env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    print(f"[Trace] mode = rollout")
    print(f"[Trace] seed = {seed}")
    print(f"[Trace] goal_bank_mode = {goal_bank_mode}")
    print(f"[Trace] ref_len = {len(ref_positions)}")
    print(
        f"[Trace] reset pos = {tuple(int(v) for v in env.unwrapped.agent_pos)} "
        f"dir = {int(env.unwrapped.agent_dir)} wind = {int(info.get('wind_direction', -1))}"
    )

    for step_idx in range(max_steps):
        pos_before = tuple(int(v) for v in env.unwrapped.agent_pos)
        dir_before = int(env.unwrapped.agent_dir)
        wind_before = int(getattr(env.unwrapped, "wind_direction", -1))
        action = rollout_eval._plan_action(
            env=env,
            obs=obs,
            model=model,
            ref_states=ref_states,
            ref_positions=ref_positions,
            step_idx=step_idx,
            lookahead_k=lookahead_k,
            future_window=future_window,
            alignment_mode=alignment_mode,
            goal_bank_mode=goal_bank_mode,
            plan_depth=plan_depth,
            collision_penalty=collision_penalty,
            turn_penalty=turn_penalty,
            progress_bonus=progress_bonus,
            success_bonus=success_bonus,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        pos_after = tuple(int(v) for v in env.unwrapped.agent_pos)
        dir_after = int(env.unwrapped.agent_dir)
        wind_after = int(info.get("wind_direction", -1))
        print(
            f"[Trace] step = {step_idx:02d} pos = {pos_before} dir = {dir_before} wind = {wind_before} "
            f"action = {ACTION_NAME.get(int(action), str(int(action)))} -> pos = {pos_after} "
            f"dir = {dir_after} next_wind = {wind_after} reward = {float(reward):.3f} "
            f"terminated = {terminated} truncated = {truncated}"
        )
        if terminated or truncated:
            break
    env.close()


def _trace_forced(
    *,
    checkpoint: Path,
    eval_env_id: str,
    fixed_u: int,
    seed: int,
    max_steps: int,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_bank_mode: str,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
    fatal_penalty: float,
) -> None:
    model, cfg = forced_eval._load_numpy_checkpoint(checkpoint)
    ref_seed = int(cfg.get("seed", 0))
    ref_states, ref_positions = forced_eval._reference_trajectory_from_fixed_u(
        env_id=eval_env_id,
        fixed_u=fixed_u,
        seed=ref_seed,
    )

    env = gym.make(eval_env_id, render_mode="rgb_array", fixed_u=fixed_u)
    obs, info = env.reset(seed=seed)
    print(f"[Trace] mode = forced")
    print(f"[Trace] fixed_u = {fixed_u}")
    print(f"[Trace] env_seed = {seed}")
    print(f"[Trace] ref_seed = {ref_seed}")
    print(f"[Trace] goal_bank_mode = {goal_bank_mode}")
    print(f"[Trace] ref_len = {len(ref_positions)}")
    print(
        f"[Trace] reset pos = {tuple(int(v) for v in env.unwrapped.agent_pos)} "
        f"dir = {int(env.unwrapped.agent_dir)} wind = {int(info.get('wind_direction', -1))}"
    )

    for step_idx in range(max_steps):
        pos_before = tuple(int(v) for v in env.unwrapped.agent_pos)
        dir_before = int(env.unwrapped.agent_dir)
        wind_before = int(getattr(env.unwrapped, "wind_direction", -1))
        action = forced_eval._plan_action(
            env=env,
            obs=obs,
            model=model,
            ref_states=ref_states,
            ref_positions=ref_positions,
            step_idx=step_idx,
            lookahead_k=lookahead_k,
            future_window=future_window,
            alignment_mode=alignment_mode,
            goal_bank_mode=goal_bank_mode,
            plan_depth=plan_depth,
            collision_penalty=collision_penalty,
            turn_penalty=turn_penalty,
            progress_bonus=progress_bonus,
            success_bonus=success_bonus,
            fatal_penalty=fatal_penalty,
        )
        obs, reward, terminated, truncated, info = env.step(action)
        pos_after = tuple(int(v) for v in env.unwrapped.agent_pos)
        dir_after = int(env.unwrapped.agent_dir)
        wind_after = int(info.get("wind_direction", -1))
        cell = env.unwrapped.grid.get(*pos_after)
        cell_type = None if cell is None else cell.type
        print(
            f"[Trace] step = {step_idx:02d} pos = {pos_before} dir = {dir_before} wind = {wind_before} "
            f"action = {ACTION_NAME.get(int(action), str(int(action)))} -> pos = {pos_after} "
            f"dir = {dir_after} next_wind = {wind_after} cell = {cell_type} "
            f"reward = {float(reward):.3f} terminated = {terminated} truncated = {truncated}"
        )
        if terminated or truncated:
            break
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one WindyCorridor evaluation episode.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-env-id", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["rollout", "forced"])
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--fixed-u", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--lookahead-k", type=int, default=4)
    parser.add_argument("--future-window", type=int, default=2)
    parser.add_argument("--alignment-mode", type=str, default="time", choices=["time", "nearest"])
    parser.add_argument(
        "--goal-bank-mode",
        type=str,
        default="waypoint",
        choices=["waypoint", "final_only"],
    )
    parser.add_argument("--plan-depth", type=int, default=2)
    parser.add_argument("--collision-penalty", type=float, default=2.0)
    parser.add_argument("--turn-penalty", type=float, default=0.05)
    parser.add_argument("--progress-bonus", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--fatal-penalty", type=float, default=999.0)
    args = parser.parse_args()

    if args.mode == "rollout":
        _trace_rollout(
            checkpoint=args.checkpoint,
            eval_env_id=args.eval_env_id,
            seed=args.seed,
            max_steps=args.max_steps,
            lookahead_k=args.lookahead_k,
            future_window=args.future_window,
            alignment_mode=args.alignment_mode,
            goal_bank_mode=args.goal_bank_mode,
            plan_depth=args.plan_depth,
            collision_penalty=args.collision_penalty,
            turn_penalty=args.turn_penalty,
            progress_bonus=args.progress_bonus,
            success_bonus=args.success_bonus,
        )
    else:
        _trace_forced(
            checkpoint=args.checkpoint,
            eval_env_id=args.eval_env_id,
            fixed_u=args.fixed_u,
            seed=args.seed,
            max_steps=args.max_steps,
            lookahead_k=args.lookahead_k,
            future_window=args.future_window,
            alignment_mode=args.alignment_mode,
            goal_bank_mode=args.goal_bank_mode,
            plan_depth=args.plan_depth,
            collision_penalty=args.collision_penalty,
            turn_penalty=args.turn_penalty,
            progress_bonus=args.progress_bonus,
            success_bonus=args.success_bonus,
            fatal_penalty=args.fatal_penalty,
        )


if __name__ == "__main__":
    main()
