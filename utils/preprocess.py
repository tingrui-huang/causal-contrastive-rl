import numpy as np


def extract_state(obs):
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