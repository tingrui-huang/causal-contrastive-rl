import numpy as np


def extract_state(obs, info=None):
    """
    Convert observation into a compact numeric state.

    Supports two formats:
    * Legacy MiniGrid dict: ``image``, ``agent_pos``, ``direction``.
    * causal_gym SCM tuple ``(x, y)``: agent direction is read from ``info``
      under the key ``agent_dir`` when available, else defaulted to 0.
    """
    if isinstance(obs, dict):
        image = obs["image"]
        agent_pos = obs.get("agent_pos", (0, 0))
        x = float(agent_pos[0])
        y = float(agent_pos[1])
        direction = float(obs.get("direction", 0))
        flat_image = np.asarray(image, dtype=np.float32).reshape(-1)
        pose = np.array([x, y, direction], dtype=np.float32)
        return np.concatenate([flat_image, pose])

    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.size == 2:
        direction = 0.0
        if info is not None:
            direction = float(info.get("agent_dir", 0))
        return np.array([arr[0], arr[1], direction], dtype=np.float32)
    return arr


def extract_state_oracle(obs, info=None):
    """Like extract_state but appends the hidden confounder U from info.

    This gives the model access to the causal variable that the standard
    baseline cannot see, serving as an upper-bound oracle.

    U is repeated as a block of constant features so that its gradient
    signal is not drowned out by the 147-dim image.
    """
    base = extract_state(obs)
    u = 0.0
    if info is not None:
        u = float(info.get("confounder", 0))
    u_block = np.full(16, u, dtype=np.float32)
    return np.concatenate([base, u_block])


def get_u_from_info(info) -> int:
    """Extract the hidden regime label from an info dict."""
    if info is None:
        return 0
    return int(info.get("confounder", 0))