# causal-contrastive-rl

Contrastive RL with **causal pessimism** for decision-making under unobserved confounding.

## Research context

Standard contrastive RL critics learn reachability scores from observational data, but
these scores conflate the causal effect of actions with hidden confounders. The **causal
pessimistic** variant trusts the critic only for observed actions and applies a pessimistic
lower bound for counterfactual actions, preventing the actor from being optimistic about
untested actions whose apparent value may be an artifact of confounding.

See `CAUSAL_PESSIMISM_SPEC.md` for the full algorithm specification and
`references/` for related papers.

## Directory layout

```text
causal-contrastive-rl/
├── envs/              # WindyCorridor environment (wind-based confounding)
├── agents/            # contrastive critic + baseline actor + causal pessimistic actor
├── buffers/           # replay buffer
├── configs/           # run_config.py (experiment params), training_defaults.py
├── experiments/       # training, evaluation, smoke tests
├── utils/             # collector, preprocessing, contrastive sampling, domain knowledge
├── figures/           # WindyCorridor schematic
├── references/        # research papers (PDF)
├── oldversion/        # archived Phase 0–8 code (HiddenFork era)
├── DEV_SPEC.md        # execution protocol (phased development rules)
├── ENV_DEV_SPEC.md    # environment specification (wind mechanics)
├── CAUSAL_PESSIMISM_SPEC.md  # Phase 9 algorithm spec
└── requirements.txt
```

## Quick start

1. Edit `configs/run_config.py` to set hyperparameters.
2. Edit `experiments/run.py` — change the `TASK` variable to the desired task.
3. Run:

```bash
python experiments/run.py
```

### Available tasks

| TASK | Description |
|------|-------------|
| `"train_oracle"` | Train actor-critic with oracle U info (upper bound) |
| `"train_confounded"` | Train actor-critic without U info (baseline) |
| `"train_causal_pessimistic"` | Train causal pessimistic actor (our method) |
| `"eval_oracle"` | Forced-U evaluation of oracle checkpoint |
| `"eval_confounded"` | Forced-U evaluation of confounded checkpoint |
| `"eval_causal_pessimistic"` | Forced-U evaluation of causal pessimistic checkpoint |

## Environment

**WindyCorridor** (`envs/windy_corridor.py`): a 15x15 MiniGrid with multi-segment corridors,
regime-dependent wind, and optional lethal boundaries. The hidden confounder `U` controls
wind distribution — calm vs strong lateral/downward wind — affecting transition dynamics
at every step, not just at a single decision point.

Registered Gymnasium ids (import `envs` before `gym.make`):

| Env id | Description |
|--------|-------------|
| `CausalContrastive-WindyCorridor-15x15-v0` | Confounded (random U) |
| `CausalContrastive-WindyCorridor-15x15-Clean-v0` | Clean (fixed U=0) |
| `CausalContrastive-WindyCorridor-15x15-Lethal-v0` | Confounded + lethal boundaries |
| `CausalContrastive-WindyCorridor-15x15-Lethal-Clean-v0` | Clean + lethal boundaries |

## Agents

| Module | Class | Role |
|--------|-------|------|
| `agents/contrastive_critic_numpy.py` | `ContrastiveCriticNumpy` | Twin InfoNCE critics (frozen during actor training) |
| `agents/goal_conditioned_actor_numpy.py` | `GoalConditionedActorNumpy` | Baseline actor (BC-regularized advantage) |
| `agents/causal_pessimistic_actor.py` | `CausalPessimisticActorNumpy` | Pessimistic actor — trusts critic only for observed actions |

## Evaluation

Forced-U evaluation (`experiments/evaluate_actor_forced_u.py`) runs the trained policy
under `fixed_u=0` and `fixed_u=1` separately and reports:

| Metric | Meaning |
|--------|---------|
| `success_u0`, `success_u1` | Per-regime goal-reaching rate |
| `mean_success` | Average of both |
| `worst_case_success` | `min(success_u0, success_u1)` — primary metric |
| `regime_gap` | `abs(success_u0 - success_u1)` — lower is better |

## Archived code

Previous phases (HiddenRegimeFork environment, contrastive baseline/robust sweeps,
diagnostics, and one-off experiment scripts) are preserved in `oldversion/` for reference.
They are not part of the active pipeline.
