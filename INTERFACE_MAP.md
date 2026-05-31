# Interface Map for Claude Code / Skills

This file records the real function names, signatures, return shapes, and call
patterns in the current WindyCorridor pessimistic pipeline. Use it to avoid
guessing interfaces when writing skills or follow-up scripts.

## State and Action Conventions

- Raw state is a length-3 tuple or array: `(x, y, dir)`.
- Position is a length-2 tuple: `(x, y)`.
- Direction encoding follows MiniGrid / causal_gym:
  - `0`: east
  - `1`: south
  - `2`: west
  - `3`: north
- Wind encoding in `utils.wind_dynamics`:
  - `0`: east
  - `1`: south
  - `2`: west
  - `3`: north
  - `4`: still
- Valid control actions used in the corridor experiments:
  - `0`: left
  - `1`: right
  - `2`: forward
  - `6`: done
- Full MiniGrid action count in saved data is `n_actions = 7`.

## Environment Constants

Source: `envs/windy_corridor.py`

```python
SIZE = 15
START_POS = (1, 1)
GOAL_POS = (13, 1)
LETHAL_X = frozenset({2, 3, 4, 7, 8, 9})
SAFE_WAIT_X = frozenset({5, 6, 10, 11})
```

`LETHAL_X` is not the lava set. It is the set of near-row `x` coordinates where
`forward` under south wind can drift into lava.

Source: `utils.wind_dynamics.py`

```python
LAVA_CELLS
WALKABLE_CELLS
WIND_STILL = 4
```

`LAVA_CELLS` is derived from `WindyCorridorEnv._y2_pattern()` and should be
treated as the source of truth for lava in pure-function dynamics.

## Pure Dynamics

Source: `utils.wind_dynamics.py`

### `analytical_step`

```python
def analytical_step(
    pos: tuple[int, int],
    dir_: int,
    action: int,
    wind: int,
) -> tuple[tuple[int, int], int, bool]:
```

Applies one action under one wind value without mutating an environment.

Returns:

```python
(new_pos, new_dir, terminated)
```

where `new_pos` is `(x, y)`, `new_dir` is an int direction, and `terminated`
means goal or lava was reached.

### `enumerate_neighbors`

```python
def enumerate_neighbors(
    pos: tuple[int, int],
    dir_: int,
    action: int,
) -> list[tuple[tuple[int, int], int, bool]]:
```

This is the Thm2 local neighborhood enumerator `N(s, x)`. It returns the result
of `analytical_step(pos, dir_, action, wind)` for all five wind values
`wind in range(5)`.

Return element shape:

```python
(new_pos, new_dir, terminated)
```

For non-forward actions, all wind values collapse to the same transition. This
is expected.

### Predicates

```python
def is_goal(pos: tuple[int, int]) -> bool
def is_lava(pos: tuple[int, int]) -> bool
```

Both take a position `(x, y)`, not a full `(x, y, dir)` state.

## Thm2 Worst-Case Kernel

Source: `utils/worst_case_kernel.py`

### Constructor

```python
class WorstCaseKernel:
    def __init__(
        self,
        propensity_fn: Callable[[tuple[int, int, int], int], float],
        v_lower_fn: Callable[[tuple[int, int, int]], float],
        wind_dist: tuple[float, ...] = CORRIDOR_WIND_DIST,
        obs_transition_fn: Callable[
            [tuple[int, int, int], int, np.random.Generator],
            tuple[tuple[int, int, int], bool] | None,
        ] | None = None,
    ) -> None:
```

The kernel does not estimate `P_b`, `P_obs`, or `V_lower` internally. Pass them
in as callables.

### Step

```python
def step(
    self,
    state: tuple[int, int, int],
    action: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int, int], bool, int, str]:
```

Returns:

```python
(next_state, terminated, sampled_wind, branch_label)
```

where:

- `next_state` is `(x, y, dir)`.
- `terminated` is `True` for goal or lava.
- `sampled_wind` is the actual wind for analytical fallback branches, and `-1`
  when no wind was sampled.
- `branch_label` is one of:
  - `"obs"`: empirical observational branch succeeded
  - `"obs_fallback"`: unseen `(s, a)` in empirical `P_obs`; analytical marginal
    wind fallback was used
  - `"adv"`: adversarial branch selected `argmin V_lower` over neighbors

### Argmin Logic

The adversarial branch uses:

```python
neighbors = enumerate_neighbors(pos, dir_, action)
val = v_lower_fn((new_pos[0], new_pos[1], new_dir))
```

It minimizes over the `v_lower_fn` values. `v_lower_fn` receives a full
`(x, y, dir)` state even if the current oracle implementation ignores direction.

## Propensity `P_b(a | s)`

Source: `utils/propensity.py`

### Estimate

```python
def estimate_propensity(
    episodes_states: list[np.ndarray],
    episodes_actions: list[np.ndarray],
    *,
    n_actions: int,
    laplace_alpha: float = 1.0,
) -> dict[tuple[int, int, int], np.ndarray]:
```

Returns a table:

```python
{(x, y, dir): probs}
```

where `probs` is a length-`n_actions` `np.ndarray`.

### Lookup

```python
def lookup(
    table: dict[tuple[int, int, int], np.ndarray],
    state,
    action: int,
    *,
    default_p: float = 0.0,
) -> float:
```

Typical wrapper:

```python
propensity_table = estimate_propensity(obs_states, obs_actions, n_actions=n_actions)

def propensity_fn(state, action):
    return lookup(propensity_table, state, action, default_p=propensity_default)
```

Use a low `propensity_default` for unseen states if you want conservative
behavior: lower default means more adversarial branch firings.

## Observational Transition `P_obs(s' | s, a)`

Source: `utils/obs_transition.py`

### Estimate

```python
def estimate_obs_transition(
    episodes_states: list[np.ndarray],
    episodes_actions: list[np.ndarray],
) -> dict[tuple[State, int], tuple[list[State], np.ndarray]]:
```

`State` is:

```python
State = tuple[int, int, int]
```

The returned table maps:

```python
((x, y, dir), action) -> (next_states, probs)
```

### Sample

```python
def sample_obs_transition(
    table: dict[tuple[State, int], tuple[list[State], np.ndarray]],
    state,
    action: int,
    rng: np.random.Generator,
) -> tuple[State, bool] | None:
```

Returns:

```python
(next_state, terminated)
```

or `None` when `(state, action)` was never observed.

Typical wrapper:

```python
obs_trans_table = estimate_obs_transition(obs_states, obs_actions)

def obs_transition_fn(state, action, rng):
    return sample_obs_transition(obs_trans_table, state, action, rng)
```

Important: this empirical branch is not the same as re-sampling wind from
`WIND_DIST`. It is conditioned on the observed action, so it preserves the
action-wind confounding in the data.

## Oracle `V_lower`

Source: `utils/v_lower_oracle.py`

### `v_lower`

```python
def v_lower(state) -> float:
```

Input is a full state-like object `(x, y, dir)`. Direction is ignored.

Returns:

- `-1e9` for lava cells
- `-1e6` for unreachable cells
- `-BFS_distance_to_goal` otherwise

This is the current `V_lower` used by the formal `WorstCaseKernel`. It is an
oracle / domain-knowledge lower value, not a learned critic and not the tabular
fixed point from VI.

### `bfs_distance`

```python
def bfs_distance(pos: tuple[int, int]) -> int | None:
```

Takes a position `(x, y)`, not a full state.

## Formal Worst-Case Data Collection

Source: `experiments/collect_corridor_worst_case.py`

Important imports:

```python
from utils.obs_transition import estimate_obs_transition, sample_obs_transition
from utils.propensity import estimate_propensity, lookup
from utils.v_lower_oracle import v_lower
from utils.wind_dynamics import is_goal, is_lava
from utils.worst_case_kernel import WorstCaseKernel
```

The kernel is instantiated as:

```python
kernel = WorstCaseKernel(
    propensity_fn=propensity_fn,
    v_lower_fn=v_lower,
    obs_transition_fn=obs_transition_fn,
)
```

The rollout helper signature is:

```python
def _rollout_episode(
    *,
    kernel: WorstCaseKernel,
    policy_step,
    expert: CorridorExpertPolicy | None,
    rng: np.random.Generator,
    max_steps: int,
) -> dict:
```

The policy wrapper signature is:

```python
def _make_policy_step(
    policy_name: str,
    *,
    expert: CorridorExpertPolicy | None,
    actor: GoalConditionedActorNumpy | None,
    goal_state: np.ndarray,
    temperature: float,
):
```

It returns:

```python
policy_step(state, rng) -> int_action
```

## One-Hot Critic Training Interfaces

Source: `experiments/train_corridor_onehot.py`

### Key Constants

```python
GAMMA = 0.99
GOAL_CELL = (13, 1)
VALID_ACTIONS = [0, 1, 2, 6]
```

### Encoding Helpers

```python
def build_indices(episode_states):
    -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int], int]]

def encode_sa(states_raw: np.ndarray, sa_index: dict) -> np.ndarray
def encode_goal(states_raw: np.ndarray, goal_index: dict) -> np.ndarray
```

- `encode_sa` maps raw `(x, y, dir)` states to one-hot features over
  `(x, y, dir)`.
- `encode_goal` maps raw states to one-hot features over `(x, y)` only.

### Training Entry Point

```python
def train(
    *,
    data_path: Path,
    output_path: Path,
    seed: int = 0,
    num_steps: int = 30000,
    batch_size: int = 64,
    critic_lr: float = 2e-3,
    hidden: int = 128,
    emb_dim: int = 64,
    tau: float = 0.07,
    log_interval: int = 2000,
    verbose: bool = True,
) -> dict:
```

Returns a dict containing:

```python
{
    "critic": ContrastiveCriticNumpy,
    "sa_index": dict,
    "goal_index": dict,
    "episode_states": list[np.ndarray],
    "final_margin": float,
}
```

### Verification Entry Point

```python
def verify(result: dict) -> None:
```

This prints:

- fork ranking: `f(near-next 2,1)` vs `f(far-next 1,2)`
- route mean scores
- correlation between learned `f` and empirical discounted goal occupancy

## Contrastive Critic Interface

Source: `agents/contrastive_critic_numpy.py`

### Constructor

```python
class ContrastiveCriticNumpy:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        emb_dim: int = 64,
        tau: float = 0.07,
        seed: int = 0,
        goal_dims: tuple[int, ...] | None = None,
        goal_feat_dim: int | None = None,
    ) -> None:
```

Use one of these goal modes:

- Raw state goals with direction ignored: `goal_dims=(0, 1)`.
- Pre-encoded one-hot goal vectors: `goal_feat_dim=n_goal`.

If `goal_feat_dim` is set, `goal_dims` is ignored.

### Embeddings

```python
def _embed_sa(self, s: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, tuple]
def _embed_g(self, g: np.ndarray) -> tuple[np.ndarray, tuple]
```

Both expect batch dimensions:

- `s`: `(B, state_dim)`
- `a`: `(B,)`
- `g`: `(B, goal_dim)`

### Scoring

```python
def score_actions(self, s: np.ndarray, goals: np.ndarray) -> np.ndarray:
```

Returns:

```python
scores.shape == (B, n_actions)
```

where `scores[i, a] = f(s_i, a, goal_i)`.

### In-Batch BCE Loss

```python
def loss_and_grads(
    self,
    s: np.ndarray,
    a: np.ndarray,
    s_future: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
```

This uses in-batch negatives:

```python
logits[i, j] = phi(s_i, a_i)^T psi(s_future_j) / tau
```

It includes the same-goal false-negative mask. If positives come from
worst-case trajectories, the off-diagonal negative marginal is the worst-case
future marginal, not the original observational future marginal.

### Monitoring

```python
def margin(self, s: np.ndarray, a: np.ndarray, s_future: np.ndarray) -> float
```

Mean positive logit minus mean true-negative logit, excluding same-goal false
negatives.

## Goal-Conditioned Actor Interface

Source: `agents/goal_conditioned_actor_numpy.py`

### Constructor

```python
class GoalConditionedActorNumpy:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        seed: int = 0,
        valid_actions: list[int] | None = None,
        goal_dim: int | None = None,
    ) -> None:
```

Use `valid_actions=[0, 1, 2, 6]` to mask unused MiniGrid actions.

Use `goal_dim=n_goal` when the actor state and goal feature widths differ, such
as one-hot `(x, y, dir)` state features with one-hot `(x, y)` goal features.

### Policy Methods

```python
def action_probs(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray
def greedy_action(
    self,
    state: np.ndarray,
    goal: np.ndarray,
    valid_actions: list[int] | None = None,
) -> int
def sample_action(
    self,
    state: np.ndarray,
    goal: np.ndarray,
    valid_actions: list[int] | None = None,
    rng: np.random.Generator | None = None,
    temperature: float = 1.0,
) -> int
```

For single-state calls, pass unbatched `state` and `goal`; the methods add a
batch dimension internally where needed.

### Baseline Actor Loss

```python
def loss_and_grads(
    self,
    state: np.ndarray,
    goal: np.ndarray,
    a_orig: np.ndarray,
    critic_scores: np.ndarray,
    lam: float = 0.5,
) -> tuple[float, dict[str, np.ndarray]]:
```

`critic_scores` must have shape `(B, n_actions)`.

## Tmp Repro Helpers Are Not Formal Interfaces

Source: `tmp_repro_correct.py`

The following helpers exist only inside `tmp_repro_correct.py`; do not import
them from a module unless they are first formalized:

```python
def Vget(cell, dr)
def qval(s, a)
def tab_greedy(s)
def true_rollout(policy_fn, force=None, seed=0, n=2000)
def sa1(s)
def g1cell(cell)
def g1(s)
def scores_at(x, y, dd)
def train_awr(beta=0.5, steps=15000, lr=2e-3, seed=1)
def actor_rollout(act, force=None, seed=0, n=2000)
```

The VI check in that file is local scratch code:

```python
V = {s: 0.0 for s in states}

def Vget(cell, dr):
    ...

def qval(s, a):
    ...
    worst = min(Vget((p[0], p[1]), dd) for (p, dd, _t) in neighbors(...))
    return GAMMA * (pb * ov + (1 - pb) * worst)
```

There is no formal public `run_vi()` or `qval()` function in `experiments/` or
`utils/` at the time this map was written. If a skill needs VI, either keep it
inline or first formalize it into a real module.

The final strong critic in `tmp_repro_correct.py` does not call
`ContrastiveCriticNumpy.loss_and_grads`. It manually implements a positive
logistic loss plus `K=48` uniform-over-observed-cells negative goals:

```python
negc = [gcells_list[i] for i in rng.integers(0, len(gcells_list), K)]
neg = np.stack([g1cell(c) for c in negc])
```

This negative sampler is not in-batch and is not the theorem's original
`p(s_f)` marginal. It is uniform over goal cells observed in the generated
worst-case data.

## Common Pitfalls for Agents

1. Do not confuse `LETHAL_X` with `LAVA_CELLS`.
2. Do not call `enumerate_neighbors(state, action)`. The real signature is
   `enumerate_neighbors(pos, dir_, action)`.
3. Do not pass a position into `v_lower`; pass `(x, y, dir)`.
4. Do not re-sample wind for the observational branch when empirical `P_obs` is
   available. Use `sample_obs_transition`.
5. Do not assume VI helpers are formal APIs. In current code, VI lives inside
   `tmp_repro_correct.py`.
6. Do not assume `train_corridor_onehot.py` implements the final strong critic.
   It uses in-batch negatives. The final scratch repro uses uniform cell
   negatives and balanced `(s, a)` sampling inside `tmp_repro_correct.py`.
7. Do not report the final scratch result as fully formalized. `SUMMARY.md`
   says the working recipe still needs to be moved out of tmp scripts.
