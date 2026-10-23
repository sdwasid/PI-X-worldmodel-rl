import json
from pathlib import Path

import numpy as np


def load_normalization_config(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    mean = np.asarray(config["joint_mean"], dtype=np.float32)
    std = np.asarray(config["joint_std"], dtype=np.float32)

    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError("Expected 6 joint normalization values")

    if not np.isfinite(mean).all():
        raise ValueError("Invalid joint mean")

    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid joint std")

    return config


def normalize_observation(obs, config):
    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim not in (1, 2) or obs.shape[-1] != 2054:
        raise ValueError(f"Invalid observation shape: {obs.shape}")

    if not np.isfinite(obs).all():
        raise ValueError("Observation contains NaN or Inf")

    mean = np.asarray(config["joint_mean"], dtype=np.float32)
    std = np.asarray(config["joint_std"], dtype=np.float32)

    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError("Invalid normalization config")

    if not np.isfinite(mean).all():
        raise ValueError("Invalid joint mean")

    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid joint std")

    result = obs.copy()

    result[..., :6] = (obs[..., :6] - mean) / std

    return result
