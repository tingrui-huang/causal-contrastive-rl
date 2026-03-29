"""Grid search over planner parameters to maximize Oracle success_u1."""
import sys
import subprocess
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints" / "sigmoid_oracle_seed0_CausalContrastive_WindyCorridor_15x15_Lethal_v0_oracle_eps.npz"
EVAL_SCRIPT = ROOT / "experiments" / "evaluate_windy_corridor_forced_u.py"
ENV_ID = "CausalContrastive-WindyCorridor-15x15-Lethal-v0"

GRID = {
    "plan_depth": [2, 3],
    "lookahead_k": [4, 6, 8],
    "future_window": [2, 4, 6],
    "collision_penalty": [0.5, 1.0, 2.0],
}

FIXED = {
    "episodes_per_regime": 20,
    "turn_penalty": 0.05,
    "progress_bonus": 0.5,
    "success_bonus": 5.0,
    "fatal_penalty": 999.0,
    "alignment_mode": "nearest",
    "goal_bank_mode": "waypoint",
}


def run_eval(params: dict) -> dict | None:
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--checkpoint", str(CKPT),
        "--eval-env-id", ENV_ID,
        "--stdout-only",
    ]
    for k, v in {**FIXED, **params}.items():
        flag = "--" + k.replace("_", "-")
        cmd.extend([flag, str(v)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None

    for line in result.stdout.splitlines():
        if line.startswith("[Eval] success_u1"):
            u1 = float(line.split("=")[1].strip())
        if line.startswith("[Eval] success_u0"):
            u0 = float(line.split("=")[1].strip())
        if line.startswith("[Eval] mean_success"):
            mean = float(line.split("=")[1].strip())

    try:
        return {"u0": u0, "u1": u1, "mean": mean}
    except UnboundLocalError:
        print(f"  FAILED: {result.stderr[-200:]}")
        return None


def main():
    keys = list(GRID.keys())
    values = list(GRID.values())
    results = []

    total = 1
    for v in values:
        total *= len(v)
    print(f"Running {total} configurations...")
    print()

    for combo in product(*values):
        params = dict(zip(keys, combo))
        tag = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"[Sweep] {tag} ...", end=" ", flush=True)
        r = run_eval(params)
        if r is None:
            print("FAILED/TIMEOUT")
            continue
        print(f"u0={r['u0']:.2f} u1={r['u1']:.2f} mean={r['mean']:.2f}")
        results.append((params, r))

    print()
    print("=" * 80)
    print("TOP 10 by success_u1:")
    print("=" * 80)
    results.sort(key=lambda x: x[1]["u1"], reverse=True)
    for i, (params, r) in enumerate(results[:10]):
        tag = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"  #{i+1}: u0={r['u0']:.2f} u1={r['u1']:.2f} mean={r['mean']:.2f}  | {tag}")


if __name__ == "__main__":
    main()
