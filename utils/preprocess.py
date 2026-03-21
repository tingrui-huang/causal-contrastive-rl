import numpy as np


def extract_state(obs):
    """
    Convert MiniGrid observation dict into a compact numeric state.
    First version: only use image + direction.
    """
    if isinstance(obs, dict):
        image = obs["image"]           # shape like (7, 7, 3)
        direction = obs.get("direction", 0)
        flat_image = np.asarray(image, dtype=np.float32).reshape(-1)
        state = np.concatenate([flat_image, np.array([direction], dtype=np.float32)])
        return state
    else:
        return np.asarray(obs, dtype=np.float32).reshape(-1)