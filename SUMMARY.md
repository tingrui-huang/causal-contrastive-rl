# Causal Contrastive RL on WindyCorridor — Status Summary

## 1. Theoretical Result

The method is the per-step Manski / pessimistic Bellman framework from `references/526_Causal_Contrastive (1).pdf`.

**Pessimistic Bellman operator** with local neighborhood $\mathcal{N}(s,x)$ (the physically reachable next-cell set under wind):

$$
(\underline{T}f)(s,x) \;=\; (1-\gamma)\mathbb{I}[s=s_g] \;+\; \gamma\, P(x\mid s)\,\mathbb{E}_{P_{\text{obs}}}\!\bigl[V_f(s')\bigr] \;+\; \gamma\bigl(1-P(x\mid s)\bigr)\min_{s''\in\mathcal{N}(s,x)} V_f(s'')
$$

**Theorems used:**
- **Lemma 1** — $\underline{T}$ is a $\gamma$-contraction → unique fixed point $\underline{d}^\pi$.
- **Theorem 1** — $\underline{d}^\pi \le d^\pi$ pointwise (valid lower bound).
- **Theorem 2** — A recursive sampler realizes $\underline{d}^\pi$ as the future-state distribution given $(s,x)$.
- **Theorem 3** — Under $p^+ = \mu\underline{d}^\pi$, $p^- = \mu p$, the Bayes-optimal contrastive critic is $f^\star(s,x,s_f) = \log\bigl(\underline{d}^\pi(s_f\mid s,x) / p(s_f)\bigr)$.

**Key property (verified, see §3):** the conditional $\underline{d}^\pi(g \mid s,a)$ does **not** depend on how often $(s,a)$ appears in the data — only on the dynamics from $(s,a)$. So even a small fraction of safe-route data is enough for the method to recommend the safe route, as long as the pessimism penalises the unsafe alternative.

## 2. What was done — diagnosis and fix path

**Starting state.** Critic margin ≈ 0.16. The trained critic ranked the lethal NEAR route ABOVE the safe FAR route at the fork. Actor walked NEAR and died.

**Method-is-sound check (oracle).** Solved the tabular pessimistic Bellman equation by value iteration on the empirical model (`P_b` and `P_obs` estimated from observational data, `V_lower` with lava absorbing). At the start:

$$
\underline{d}(\text{goal} \mid \text{start},\text{forward/NEAR}) = 0.455
\qquad
\underline{d}(\text{goal} \mid \text{start},\text{right/FAR}) = 0.637
$$

Greedy w.r.t. exact $\underline{d}$ achieves **88% goal / 87% far** under natural wind and **0% lava death** under worst-case forced-south wind. This isolates every observed failure to the **contrastive estimator**, not to the principle.

**Six estimator pathologies — each diagnosed concretely and fixed:**

| # | Pathology | Why it breaks here | Fix |
|---|---|---|---|
| 1 | Goal direction confound | NEAR arrives at goal facing east, FAR facing north — different goal states under any encoding; querying a fixed direction silently favours NEAR | g-encoder ignores direction (`goal_dims=(0,1)`) |
| 2 | False negatives | Goal cell is ~70% of all HER positives → a batch of 64 contains ~45 copies of the goal; in-batch off-diagonals labelled "not future" contradict the diagonals' "future" → margin collapses | Mask same-cell off-diagonals in the in-batch BCE |
| 3 | Coordinate smoothness | Raw `[x,y,dir]` lets the MLP score "(12,1) ≈ (13,1)" by coordinate distance; NEAR cells inherit a free bias toward the goal | One-hot state encoding over `(x,y,dir)` |
| 4 | $q(\text{goal})$ domination | In-batch negatives → $q(\text{goal})\approx 0.71$ → the useful log-ratio is compressed into a ~1.4-logit window; at $\tau=0.07$ this is below the resolvable cosine gap | Uniform-over-cells negatives, $\tau$ retuned to 0.1 |
| 5 | Rare-route undertraining | Far-route $(s,a)$ pairs are 16% of data → noisy action-value estimates on the long FAR route, e.g. $(1,1,\text{south})+\text{forward}$ scored *below* turning back | Balanced sampling over distinct $(s,a)$ pairs |
| 6 | OOD-action overestimation | At start, "left" (toward wall) never appears in data → its critic value is unconstrained and happens to be high → critic-greedy walks into the wall | AWR actor (advantage-weighted cloning over data actions only) + valid-action logit mask |

**Code changes (in `main` repo):**
- `agents/contrastive_critic_numpy.py` — added `goal_dims`, `goal_feat_dim` (decoupled goal-encoder width); false-negative mask built into `loss_and_grads` and `margin`.
- `agents/goal_conditioned_actor_numpy.py` — added `valid_actions` (additive logit mask); decoupled state/goal dimensions.
- `experiments/train_corridor.py` — passes `goal_dims=(0,1)` to the critic.
- `experiments/train_corridor_onehot.py` (new) — one-hot critic training + verification (covers fixes 1, 2, 3).

**Working recipe (still in `/tmp`, not yet formalised into the repo):**
- **Critic:** one-hot $(x,y,\text{dir})$ for `sa`, one-hot $(x,y)$ for `g` (dir-agnostic) + false-neg mask + **uniform-over-cells negatives** + **balanced $(s,a)$ sampling** + $\tau=0.1$.
- **Actor:** AWR weighted BC ($\beta=0.5$) over raw `[x,y,dir]` features, dir-agnostic goal, valid-action mask.

## 3. Current Result

| Setting | Natural wind | Worst-case forced-south wind |
|---|---:|---:|
| **Tabular oracle** (exact $\underline{d}^\pi$, planning) | 88% goal / 87% far | **0% lava** (100% timeout, 100% far) |
| **Learned full recipe** (strong critic + AWR β=0.5) | **86% goal / 87% far** | **0% lava** (100% timeout, 100% far) |
| Critic-only after fixes 1–3 (no AWR, no uniform-neg, no balance) | 54% goal / 46% lava / 100% near | 100% lava / 100% near |
| Original broken baseline | walks NEAR / dies | walks NEAR / dies |

The learned contrastive pipeline **matches the tabular oracle within 2% on natural wind and exactly on worst-case lava avoidance**.

Critic margin trajectory across the fix path:
`0.16` (broken) → `0.36` (just fix 2) → `0.69` (fixes 1+2 on raw coords) → `1.69` (fixes 1+2+3, one-hot) → strong critic with uniform-neg + balanced sampling adds the deep-route action-value correctness needed for the policy.

## 4. Honest framing relative to the original contrastive_rl

Fixes (4), (5), (6) **deviate** from the standard Eysenbach setup. The original repo uses uniform replay sampling and in-batch negatives — sufficient for *online* control with *diverse* goals, both of which this testbed violates (offline + single absorbing goal). The deviations are not arbitrary: each addresses a specific assumption the testbed breaks.

For the long-horizon / rare-route issue (#5), there is a more principled in-method alternative: the **C-learning TD variant** in the same repo, which bootstraps by re-weighting negatives with `next_v / (1-next_v)`. It would handle the long FAR route the way tabular VI does (backward value propagation) and might let us drop the `(s,a)`-balancing band-aid. This is the most promising next direction to align the working recipe with the published method.

## 5. Open items

1. **Formalise the working recipe in the repo** — upgrade `experiments/train_corridor_onehot.py` to add uniform negatives + balanced sampling; add `experiments/train_corridor_onehot_awr.py` for the AWR actor; clean up the `/tmp` scratch scripts.
2. **Try the C-learning TD critic** as a principled replacement for fixes (4)+(5), keeping the rest. Goal: match or beat the current strong critic without ad-hoc reweighting.
3. **Forced-U evaluation table** — produce the same `success / mean / worst-case / gap` numbers as the references PDF (baseline, ours, oracle), seed-averaged.
4. **Multi-goal caveat** — the in-batch + balanced-sampling combination introduces a $g$-only additive shift in $f^\star$; harmless at the single fixed goal here, but would distort any cross-goal comparison. Re-check before any subgoal/HER-eval extension.
