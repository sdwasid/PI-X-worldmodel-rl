"""Explicit mappings between policy and robot state/action spaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    """Per-dimension mean/std normalization parameters."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.ndim != 1 or mean.shape != std.shape:
            raise ValueError("Normalization mean/std must be same-sized vectors")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Normalization values must be finite and std must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def normalize(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape[-1] != len(self.mean):
            raise ValueError("Normalization dimension mismatch")
        return (array - self.mean) / self.std

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape[-1] != len(self.mean):
            raise ValueError("Denormalization dimension mismatch")
        return array * self.std + self.mean


@dataclass(frozen=True, slots=True)
class GripperMapping:
    """Linear model-to-robot range mapping for one gripper dimension."""

    model_index: int
    robot_index: int
    model_range: tuple[float, float]
    robot_range: tuple[float, float]


class ActionAdapter:
    """Handle dimension padding, joint order, gripper range, modes, and stats."""

    def __init__(
        self,
        model_dim: int,
        robot_dim: int,
        robot_from_model: Sequence[int | None] | None = None,
        state_normalization: NormalizationSpec | None = None,
        action_normalization: NormalizationSpec | None = None,
        gripper: GripperMapping | None = None,
        model_action_mode: Literal["delta_joint", "absolute_joint"] = "delta_joint",
        robot_action_mode: Literal["delta_joint", "absolute_joint"] = "delta_joint",
    ) -> None:
        if model_dim <= 0 or robot_dim <= 0:
            raise ValueError("model_dim and robot_dim must be positive")
        valid_action_modes = {"delta_joint", "absolute_joint"}
        if model_action_mode not in valid_action_modes:
            raise ValueError(
                f"model_action_mode must be 'delta_joint' or 'absolute_joint', "
                f"got {model_action_mode!r}"
            )
        if robot_action_mode not in valid_action_modes:
            raise ValueError(
                f"robot_action_mode must be 'delta_joint' or 'absolute_joint', "
                f"got {robot_action_mode!r}"
            )
        self.model_dim = model_dim
        self.robot_dim = robot_dim
        mapping = list(robot_from_model) if robot_from_model is not None else list(range(robot_dim))
        if len(mapping) != robot_dim:
            raise ValueError("robot_from_model must contain one entry per robot dimension")
        used = [index for index in mapping if index is not None]
        if any(index < 0 or index >= model_dim for index in used) or len(set(used)) != len(used):
            raise ValueError("robot_from_model contains invalid or duplicate model indices")
        if state_normalization is not None and len(state_normalization.mean) != model_dim:
            raise ValueError("state_normalization must match model_dim")
        if action_normalization is not None and len(action_normalization.mean) != model_dim:
            raise ValueError("action_normalization must match model_dim")
        if gripper is not None:
            for field, index, dimension in (
                ("model_index", gripper.model_index, model_dim),
                ("robot_index", gripper.robot_index, robot_dim),
            ):
                if (
                    isinstance(index, (bool, np.bool_))
                    or not isinstance(index, (int, np.integer))
                    or index < 0
                    or index >= dimension
                ):
                    raise ValueError(
                        f"gripper.{field} must be an integer in [0, {dimension})"
                    )
            for field, bounds in (
                ("model_range", gripper.model_range),
                ("robot_range", gripper.robot_range),
            ):
                invalid_range_message = (
                    f"gripper.{field} must contain two finite numeric endpoints"
                )
                try:
                    values = np.asarray(bounds)
                    numeric = np.issubdtype(values.dtype, np.integer) or np.issubdtype(
                        values.dtype, np.floating
                    )
                except (TypeError, ValueError):
                    raise ValueError(invalid_range_message) from None
                if values.shape != (2,) or not numeric or not np.all(np.isfinite(values)):
                    raise ValueError(invalid_range_message)
                if values[0] >= values[1]:
                    raise ValueError(f"gripper.{field} must be strictly increasing")
        self.robot_from_model = mapping
        self.state_normalization = state_normalization
        self.action_normalization = action_normalization
        self.gripper = gripper
        self.model_action_mode = model_action_mode
        self.robot_action_mode = robot_action_mode

    @staticmethod
    def pad_vector(value: np.ndarray, target_dim: int, pad_value: float = 0.0) -> np.ndarray:
        """Pad a 1D vector; refuse silent truncation."""
        vector = np.asarray(value, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("Expected a one-dimensional vector")
        if len(vector) > target_dim:
            raise ValueError(f"Cannot truncate dimension {len(vector)} to {target_dim}")
        output = np.full(target_dim, pad_value, dtype=np.float64)
        output[: len(vector)] = vector
        return output

    def robot_state_to_model(self, robot_state: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Reorder robot state into a padded model vector."""
        state = np.asarray(robot_state, dtype=np.float64)
        if state.shape != (self.robot_dim,):
            raise ValueError(f"robot_state must have shape ({self.robot_dim},)")
        model = np.zeros(self.model_dim, dtype=np.float64)
        for robot_index, model_index in enumerate(self.robot_from_model):
            if model_index is not None:
                model[model_index] = state[robot_index]
        if normalize and self.state_normalization is not None:
            model = self.state_normalization.normalize(model)
        return model

    def model_action_to_robot(
        self,
        model_action: np.ndarray,
        current_robot_state: np.ndarray | None = None,
        denormalize: bool = True,
    ) -> np.ndarray:
        """Convert one model action into the configured robot command mode."""
        action = self.pad_vector(model_action, self.model_dim)
        if denormalize and self.action_normalization is not None:
            action = self.action_normalization.denormalize(action)
        robot = np.zeros(self.robot_dim, dtype=np.float64)
        for robot_index, model_index in enumerate(self.robot_from_model):
            if model_index is not None:
                robot[robot_index] = action[model_index]
        if self.gripper is not None:
            robot[self.gripper.robot_index] = self._map_gripper(action[self.gripper.model_index])
        if self.model_action_mode != self.robot_action_mode:
            if current_robot_state is None:
                raise ValueError("current_robot_state is required when converting action modes")
            state = np.asarray(current_robot_state, dtype=np.float64)
            if state.shape != (self.robot_dim,):
                raise ValueError(f"current_robot_state must have shape ({self.robot_dim},)")
            if self.model_action_mode == "absolute_joint":
                robot = robot - state
            else:
                robot = state + robot
        return robot

    def model_chunk_to_robot(
        self, model_chunk: np.ndarray, current_robot_state: np.ndarray | None = None
    ) -> np.ndarray:
        """Convert a [chunk, model_dim] action sequence."""
        chunk = np.asarray(model_chunk, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError("model_chunk must have shape [chunk_size, action_dim]")
        return np.stack(
            [self.model_action_to_robot(row, current_robot_state) for row in chunk], axis=0
        )

    def _map_gripper(self, value: float) -> float:
        assert self.gripper is not None
        model_min, model_max = self.gripper.model_range
        robot_min, robot_max = self.gripper.robot_range
        if model_min >= model_max or robot_min >= robot_max:
            raise ValueError("Gripper ranges must be increasing")
        fraction = (np.clip(value, model_min, model_max) - model_min) / (model_max - model_min)
        return float(robot_min + fraction * (robot_max - robot_min))
