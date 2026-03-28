# Environment Development Specification (ENV_DEV_SPEC)

> **WARNING FOR CODE GENERATION (VIBE CODING):**
> This document is the strict execution protocol for building and evolving confounded
> environments. Do NOT invent new mechanics outside the phased plan. Do NOT optimize for
> creativity. Follow phases strictly. A phase is NOT complete unless its verification
> script passes and all required log lines match.

**Repository:** `causal-contrastive-rl`
**Companion specs:** `DEV_SPEC.md` (learning pipeline), `Contrastive319.pdf` (research motivation).

This document defines **how environments are built, extended, and verified**. It is the
single source of truth for environment-side phased development, mechanism contracts, and
confounding-strength control. It does **not** govern training or evaluation — those live
in `DEV_SPEC.md`.

---

## 1. Purpose

- **Goal:** Build **minimal, controllable, verifiable** confounded environments where the
  hidden variable provably affects transition dynamics, not just geometry.
- **Non-goals:** Visual richness, game complexity, novel environment design (early stage).

---

## 2. Development Philosophy

We optimize for:

| Property | Meaning |
|----------|---------|
| **Controllability** | Every mechanism (wind, trap, geometry) can be toggled on/off via constructor args. |
| **Observability** | Hidden variables are visible in `info` dicts and logs — never in `obs`. |
| **Verifiability** | Every mechanism has a deterministic smoke script that prints pass/fail. |
| **Composability** | New dynamics (e.g. wind) layer on top of existing map variants without rewriting. |

We explicitly reject:

- "Looks correct" — must be verified by script.
- "Seems harder" — must show `p(s'|s,a,U=0) ≠ p(s'|s,a,U=1)` in logs.
- "Probably confounded" — must degrade baseline in a controlled experiment.

---

## 3. Core Environment Contract

### 3.1 Hidden Confounder (`hidden_u`)

- `hidden_u ∈ {0, 1}` sampled at `reset` (when `confound=True`).
- MUST NOT appear in `obs` (the image tensor or any field the agent receives).
- MAY appear in:
  - `env.unwrapped.hidden_u`
  - `info["confounder"]`
  - stdout logs

### 3.2 Information Wall (STRICT)

```text
Agent observation MUST NOT contain any direct or indirect signal of hidden_u.
```

Violation → environment is **INVALID**. This includes visible geometry asymmetry at the
decision point that correlates perfectly with `U` (the motivation for `hidden_trap` over
`branch_wall`).

### 3.3 Toggleability

Every confounding mechanism MUST support:

```python
confound=False   # deterministic U=0, no stochastic dynamics
fixed_u=0 | 1    # force a specific regime for diagnostics
```

### 3.4 Transition Requirement (STRICT)

```text
There MUST exist (s, a) such that:
    p(s' | s, a, U=0)  ≠  p(s' | s, a, U=1)
```

If this cannot be demonstrated by a verification script → the confounder does **not**
exist in the dynamics and the environment is not research-ready.

---

## 4. Existing Environment Inventory

### 4.1 `HiddenRegimeForkEnv` (Phase 7, implemented)

| Property | Value |
|----------|-------|
| File | `envs/hidden_regime_fork.py` |
| Grid | 15×15, odd, center spine + fork |
| Map variants | `branch_wall`, `hidden_trap` |
| Confounder channel | Geometry only (which branch is open / safe) |
| Dynamics channel | None — `step()` uses default MiniGrid forward |
| Registered ids | `CausalContrastive-HiddenFork-15x15-v0`, `…-Clean-v0`, `…HiddenTrap…-v0`, `…HiddenTrap…-Clean-v0` |

**Known limitation:** The confounder only affects **which cells are walkable or trapped**
(decided at `reset`). Once the agent is on the center spine, the transition function
`p(s'|s,a)` is identical for both `U` values. A baseline that memorizes "go straight" can
still succeed without ever needing to distinguish regimes. This limits the **confounding
strength** available for experiments.

---

## 5. Next Target: Dynamic Wind Confounder

> This section defines the **next planned extension**. The protocol itself (§3, §6–§12)
> remains valid for future designs.

### 5.1 Motivation

Inspired by [Causal-Gymnasium `WindyMiniGridPCH`](https://github.com/CausalAILab/Causal-Gymnasium),
we add **regime-dependent wind** that modifies the **transition dynamics** of `step()`,
not just the map geometry. This creates confounding that is:

- **Continuous in strength** — tunable via `wind_strength` and `P(wind | U)`.
- **Active every step** — not only at the fork decision point.
- **Invisible** — wind direction / presence is not in `obs`.

### 5.2 Wind Mechanics (Specification)

Wind is a **discrete direction** sampled each step (or each episode — see §5.3):

| Wind value | Direction | Effect on MiniGrid `forward` action |
|------------|-----------|-------------------------------------|
| 0 | Right (+x) | If agent faces right → move 2; if agent faces left → move 0; otherwise normal |
| 1 | Down (+y) | If agent faces down → move 2; if agent faces up → move 0; otherwise normal |
| 2 | Left (−x) | If agent faces left → move 2; if agent faces right → move 0; otherwise normal |
| 3 | Up (−y) | If agent faces up → move 2; if agent faces down → move 0; otherwise normal |
| 4 | No wind | Normal movement (move 1) |

"Move 2" means the agent attempts to advance two cells in one step (stopping at the first
wall / obstacle). "Move 0" means the forward action has no displacement effect.

**Only the `forward` action (action 2 in MiniGrid) is affected.** Turn left, turn right,
and all other actions are unmodified.

### 5.3 Wind Sampling

Wind is drawn from a **categorical distribution** that depends on `hidden_u`:

```python
wind_dist: dict[int, tuple[float, ...]]
# wind_dist[u] = (p_right, p_down, p_left, p_up, p_none)  len=5, sums to 1
```

**Default (proposed):**

```python
wind_dist = {
    0: (0.05, 0.05, 0.05, 0.05, 0.80),   # U=0: mostly calm
    1: (0.25, 0.25, 0.05, 0.05, 0.40),    # U=1: strong lateral/downward wind
}
```

**Sampling frequency options (constructor arg `wind_per`):**

| Value | Meaning |
|-------|---------|
| `"step"` | Re-sample wind direction every `step()` call (default). |
| `"episode"` | Sample once at `reset()`, fixed for the episode. |

### 5.4 Confounding Strength Control

The environment MUST expose a **`wind_strength`** parameter in `[0.0, 1.0]`:

- `wind_strength = 0.0` → wind has no effect (equivalent to `p_none = 1.0` regardless of
  `wind_dist`). Useful as a control.
- `wind_strength = 1.0` → full effect as specified by `wind_dist`.
- Intermediate values interpolate: effective distribution =
  `(1 - wind_strength) * (0,0,0,0,1) + wind_strength * wind_dist[u]`.

This allows **sweeping confounding strength** to find the regime where baseline degrades
but our method still works.

### 5.5 Composability with Map Variants

Wind is **orthogonal** to the map variant:

| Combination | Geometry | Dynamics |
|-------------|----------|----------|
| `hidden_trap` + no wind | Symmetric branches, latent trap | Default MiniGrid forward |
| `hidden_trap` + wind | Symmetric branches, latent trap | Regime-dependent stochastic forward |
| `branch_wall` + wind | Asymmetric branches | Regime-dependent stochastic forward |

All combinations MUST be testable. The wind extension MUST NOT break existing map variant
behavior when `wind_strength = 0.0`.

### 5.6 Information Contract

| Field | In `obs`? | In `info`? | In logs? |
|-------|-----------|------------|----------|
| `hidden_u` | NO | YES (`info["confounder"]`) | YES |
| `wind_direction` (current step) | NO | YES (`info["wind_direction"]`) | YES |
| `wind_strength` (config) | NO | YES (`info["wind_strength"]`) | YES |
| `wind_dist` (config) | NO | NO (constructor only) | YES (at reset) |

---

## 6. Phase-Based Construction (MANDATORY)

Environment extensions MUST be built in phases. Each phase:

- Has **ONE** change.
- Has **ONE** expected effect.
- Has **ONE** verification method (script + required log lines).

No multi-feature phases. No skipping.

---

## 7. Logging Contract (Exact Prefix Match)

Follows the same rules as `DEV_SPEC.md` §6.1:

- Each required line MUST match the documented string as an **exact prefix**
  (case-sensitive), from the first character through the last fixed character before
  dynamic content.
- **Dynamic segments** (numbers, tuples, floats) may vary after the prefix.
- **Forbidden:** paraphrasing labels, changing bracket style, changing spacing around `=`,
  different tag names, lowercasing tags, or omitting punctuation that is part of the
  prefix.

---

## 8. Phased Execution Protocol — Wind Extension

### Phase W0 — Baseline Snapshot

**Task:** Verify existing `hidden_trap` environment still passes all `DEV_SPEC.md` Phase 4
and Phase 7 contracts before any modification.

**Verification command:**

```bash
python experiments/run_pipeline.py --env CausalContrastive-HiddenForkHiddenTrap-15x15-v0
```

**Success criteria (STRICT):**

- All Phase 4 log lines from `DEV_SPEC.md` appear.
- No uncaught exceptions.

**Gate:** Do NOT proceed to Phase W1 until this passes.

---

### Phase W1 — Wind Sampling Only (No Effect)

**Task:**

- Add constructor args: `wind_dist`, `wind_strength`, `wind_per`.
- At each `step()` (or `reset()` if `wind_per="episode"`), sample `wind_direction` from
  `wind_dist[hidden_u]`.
- Store in `info["wind_direction"]` and `info["wind_strength"]`.
- **DO NOT** modify movement. Wind is sampled but has zero effect.

**Verification command:**

```bash
python experiments/smoke_env_wind.py --phase w1
```

**Required logging:**

```text
[EnvWind] reset hidden_u = <int>
[EnvWind] reset wind_strength = <float>
[EnvWind] step = <int>, wind_direction = <int>, agent_pos = <tuple>
```

**Success criteria (STRICT):**

- `wind_direction` values appear in `{0,1,2,3,4}`.
- Agent movement is **identical** to no-wind baseline (compare trajectories with
  `wind_strength=0.0` and `wind_strength=1.0` under the same seed — positions must match
  because wind has no effect yet).
- All existing `DEV_SPEC.md` Phase 4 log lines still pass.

---

### Phase W2 — Wind Affects Forward (CRITICAL)

**Task:**

- Override `step()` so that when `action == 2` (forward) and `wind_strength > 0`:
  - If wind direction is **same** as agent facing → attempt to move **2 cells**.
  - If wind direction is **opposite** to agent facing → move **0 cells** (stay in place).
  - Otherwise → normal move (1 cell).
- "Move 2" stops at first wall/obstacle. "Move 0" means no displacement.
- All other actions unchanged.

**Verification command (MANDATORY SCRIPT):**

```bash
python experiments/smoke_env_wind.py --phase w2
```

**Script MUST:**

1. Fix agent position and direction.
2. Set `fixed_u=0`, run one `forward` step, record `next_pos`.
3. Reset to same position, set `fixed_u=1`, run one `forward` step, record `next_pos`.
4. Print both.

**Required logging:**

```text
[Dynamics] U=0 wind_dir=<int> next_pos = <tuple>
[Dynamics] U=1 wind_dir=<int> next_pos = <tuple>
```

**Success criteria (STRICT):**

```text
The two next_pos MUST be different for at least one tested (position, direction) pair.
```

If equal for ALL tested pairs → **FAIL** → do NOT proceed.

Additional check:

- With `wind_strength=0.0`, behavior MUST be identical to Phase W1 (no wind effect).

---

### Phase W3 — Strength Sweep Smoke Test

**Task:**

- Run a short episode under `wind_strength ∈ {0.0, 0.3, 0.6, 1.0}` with `confound=True`.
- Collect trajectory lengths and success rates.

**Verification command:**

```bash
python experiments/smoke_env_wind.py --phase w3
```

**Required logging:**

```text
[WindSweep] wind_strength = <float>, mean_traj_len = <float>, success_rate = <float>
```

**Success criteria (STRICT):**

- `wind_strength=0.0` trajectory stats match non-wind baseline.
- Higher `wind_strength` shows **different** (not necessarily worse) trajectory stats.
- No crashes at any strength level.

---

### Phase W4 — Gymnasium Registration

**Task:**

- Register new env ids in `envs/__init__.py` for the wind-enabled variants.
- Naming convention: append `-Wind` before version, e.g.:
  - `CausalContrastive-HiddenForkHiddenTrap-15x15-Wind-v0`
  - `CausalContrastive-HiddenForkHiddenTrap-15x15-Wind-Clean-v0`

**Verification command:**

```bash
python -c "import gymnasium as gym; import envs; env = gym.make('CausalContrastive-HiddenForkHiddenTrap-15x15-Wind-v0'); print('OK')"
```

**Success criteria (STRICT):**

- `gym.make(...)` succeeds for all new ids.
- Existing (non-wind) ids still work.

---

### Phase W5 — Pipeline Integration

**Task:**

- Run the full `DEV_SPEC.md` Phase 4 pipeline with a wind-enabled env id.
- Verify all pipeline log lines still pass.
- Verify `state_shape` is unchanged (wind must not alter observation dimensionality).

**Verification command:**

```bash
python experiments/run_pipeline.py --env CausalContrastive-HiddenForkHiddenTrap-15x15-Wind-v0
```

**Success criteria (STRICT):**

- All `DEV_SPEC.md` Phase 4 log lines appear.
- `[Pipeline] processed state shape = ` matches the non-wind variant exactly.
- No `hidden_u` or `wind_direction` leakage into processed state.

**Gate:** Do NOT start training experiments until Phase W5 passes.

---

### Phase W6 — Baseline Degradation Test

**Task:**

- Train the Phase 6 / Phase 8 baseline on the wind-enabled confounded env.
- Compare forced-regime evaluation (`fixed_u=0`, `fixed_u=1`) against the non-wind
  baseline results.

**Verification command:**

```bash
python experiments/train_contrastive_baseline.py --env CausalContrastive-HiddenForkHiddenTrap-15x15-Wind-v0
```

(Or the robust-v1 equivalent with `w=1.0`.)

**Required logging:**

```text
[BaselineDegradation] env = <string>
[BaselineDegradation] wind_strength = <float>
[BaselineDegradation] mean_success_no_wind = <float>
[BaselineDegradation] mean_success_with_wind = <float>
[BaselineDegradation] worst_case_success_no_wind = <float>
[BaselineDegradation] worst_case_success_with_wind = <float>
```

**Success criteria (STRICT):**

- Baseline `worst_case_success` degrades under wind compared to no-wind.
- If baseline does NOT degrade → increase `wind_strength` or adjust `wind_dist` and
  re-run. If no setting causes degradation → the wind mechanism is too weak and must be
  redesigned before claiming confounding.

---

## 9. Dataset Collection Contract (Separate from Env)

Collector MAY access:

```python
env.unwrapped.hidden_u
```

Collector MUST NOT:

- Modify environment state.
- Inject `hidden_u` or `wind_direction` into stored data.

Stored transitions MUST remain:

```python
{"obs": ..., "action": ..., "reward": ..., "next_obs": ..., "done": ...}
```

This contract is inherited from `DEV_SPEC.md` §7 (Phase 8 data-generation contract) and
applies identically to wind-enabled environments.

---

## 10. Sanity Checks (MANDATORY, run after any env change)

### Check 1 — Mechanism

```text
Same (s, a) → different next states under U=0 vs U=1
```

### Check 2 — Aliasing

```text
Same observation at decision point → different future trajectories under U=0 vs U=1
```

### Check 3 — Toggle

```text
confound=False → deterministic behavior (no regime randomness)
wind_strength=0.0 → identical to non-wind variant
```

### Check 4 — Observation Purity

```text
obs tensor is byte-identical between wind_strength=0.0 and wind_strength=1.0
at the same (agent_pos, agent_dir, grid_state) — wind does not leak into pixels.
```

---

## 11. Definition of Done

### 11.1 Implementation Done

An environment extension is **complete** if:

- [ ] `hidden_u` exists and is logged.
- [ ] `hidden_u` not in `obs`.
- [ ] New dynamic variable (e.g. `wind_direction`) not in `obs`.
- [ ] Transition differs across `U` (verified by smoke script).
- [ ] `fixed_u` works.
- [ ] `confound=False` disables randomness.
- [ ] `wind_strength=0.0` recovers non-wind behavior exactly.
- [ ] All logging contracts satisfied.
- [ ] All existing `DEV_SPEC.md` pipeline phases still pass.

### 11.2 Research-Ready Done

An environment extension is **validated** if:

- [ ] Clean env (no confounding) → baseline performs well.
- [ ] Confounded env → baseline degrades OR shows regime gap.
- [ ] Strength sweep shows monotonic (or at least non-trivial) degradation curve.

Metrics (from `DEV_SPEC.md` Phase 8 evaluation protocol):

- `mean_success`
- `worst_case_success`
- `regime_gap`

---

## 12. Debug Contract (MANDATORY)

Every environment module MUST:

- Print shapes or positions at boundaries.
- Print at least one concrete value per mechanism per step (when verbose/debug mode is on).
- Never silently swallow exceptions.

---

## 13. Forbidden

- Adding new mechanics without a phase entry in this document.
- Mixing multiple mechanism changes in one phase.
- `hidden_u` leakage into `obs`.
- `wind_direction` leakage into `obs`.
- Skipping verification scripts.
- Silent exception handling (`except: pass`).
- Changing observation shape/dtype as a side effect of adding dynamics.

---

## 14. Key Principle

```text
If a confounder cannot be verified through controlled tests and logs,
it does not exist.
```

---

## 15. Relationship to DEV_SPEC

```text
ENV_DEV_SPEC.md  → controls environment correctness and confounding strength
DEV_SPEC.md      → controls learning pipeline correctness and evaluation integrity
```

Both must pass before research claims are valid.

---

## 16. Document Maintenance

- **Canonical file:** `ENV_DEV_SPEC.md`. Do not fork competing env specs; update this
  file.
- When you add a new mechanism or environment variant, **append** a new phase block with a
  new number (e.g. Phase W7, W8, …); do not renumber completed phases.
- If a log string must change, update **Required logging** here and in the corresponding
  smoke script in the **same** commit.
