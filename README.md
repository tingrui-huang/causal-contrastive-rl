# causal-contrastive-rl

Minimal experimental framework for **contrastive RL** and (later) **causal / pessimistic** variants under confounding.

## Execution protocol (canonical)

> **This repository’s rules for phased work, log contracts, and verification live in [`DEV_SPEC.md`](DEV_SPEC.md).**  
> That file is the execution protocol—not a research essay. Research motivation is in `Contrastive319.pdf` (scope for later phases only).

- Follow phases **0 → 1 → 2 → …** in order.
- Each phase has a **verification command** and **exact-prefix** log requirements—see `DEV_SPEC.md`.

**Legacy path:** `excuate.md` redirects here so older links keep working.

## Directory layout

```text
causal-contrastive-rl/
├── envs/              # MiniGrid modifications (confounding later)
├── agents/            # contrastive RL / causal variants
├── buffers/           # replay + contrastive sampling
├── experiments/       # runnable scripts (smoke tests, training)
├── configs/           # hyperparameter configs
├── utils/             # collector, preprocessing, helpers
├── main.py
├── requirements.txt
├── README.md
├── DEV_SPEC.md        # execution protocol (read this for Cursor / contributors)
└── Contrastive319.pdf # research notes (reference only)
```

## Quick start

```bash
python experiments/run_random.py
```

(Phase 0 sanity check; see `DEV_SPEC.md` for the full command list per phase.)
