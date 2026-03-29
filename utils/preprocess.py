import numpy as np


def extract_state(obs, info=None):
    """
    Convert MiniGrid observation dict into a compact numeric state.
    Include absolute agent pose for navigation-heavy custom environments.
    """
    if isinstance(obs, dict):
        image = obs["image"]  # shape like (7, 7, 3)
        agent_pos = obs.get("agent_pos", (0, 0))
        x = float(agent_pos[0])
        y = float(agent_pos[1])
        direction = float(obs.get("direction", 0))
        flat_image = np.asarray(image, dtype=np.float32).reshape(-1)
        pose = np.array([x, y, direction], dtype=np.float32)
        state = np.concatenate([flat_image, pose])
        return state
    else:
        return np.asarray(obs, dtype=np.float32).reshape(-1)


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