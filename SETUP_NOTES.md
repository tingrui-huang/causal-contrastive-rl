# Environment Setup Notes

## Current state
- **Venv**: `.venv\` — Python 3.12.7 (`py -3.12 -m venv .venv`)
- **causal_gym**: editable install from `D:\Users\trhua\Research\Causal-Gymnasium`
- **Other deps**: `requirements.txt` (gymnasium, minigrid, torch 2.12.0 CPU, numpy, matplotlib)
- **Smoke test**: `smoke_test_causal_gym.py` → `OK: (1, 1) {'wind': 4}`

## Local patches in `Causal-Gymnasium` (uncommitted)
1. `setup.py` — dropped `gymnasium[all]`, `box2d-py`, `multiprocess`, `highway-env`; added missing `minigrid`; moved heavy deps to `extras_require`.
2. `causal_gym/envs/__init__.py` — wrapped `lunar_lander`, `mnist`, `highway*`, `antmaze`, `race`, `masked_atari`, `random_friction_ant` imports in `try/except ImportError`.

## Why the original install failed
- Default `python` is 3.14.4 — no wheels for `numpy<=1.26.4`, pillow, pygame.
- `py -3.11` launcher entry is stale (`E:\...\Python311\` was deleted).
- `box2d-py` has no Windows wheel — needs SWIG + MSVC Build Tools (~6GB). Only LunarLander uses it.

## Re-enabling heavy envs (if needed later)
```bash
pip install -e D:\Users\trhua\Research\Causal-Gymnasium[box2d]   # needs MSVC + SWIG
pip install -e D:\Users\trhua\Research\Causal-Gymnasium[mujoco]  # mujoco has cp312 wheels
pip install -e D:\Users\trhua\Research\Causal-Gymnasium[atari]   # ale-py has wheels
pip install -e D:\Users\trhua\Research\Causal-Gymnasium[highway] # pure Python
pip install -e D:\Users\trhua\Research\Causal-Gymnasium[mnist]   # torch + torchvision
```
