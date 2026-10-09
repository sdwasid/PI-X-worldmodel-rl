"""YAML loading and explicit configuration validation."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing or unsafe."""


def project_root() -> Path:
    """Return the installed source tree root for editable/source installations."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Resolve a user path against the current directory, then the project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    rooted = project_root() / candidate
    return rooted.resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and report malformed or missing files clearly."""
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise ConfigError(f"Configuration file does not exist: {resolved}")
    try:
        value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration root must be a mapping: {resolved}")
    return value


def _required(config: dict[str, Any], section: str, fields: tuple[str, ...]) -> dict[str, Any]:
    value = config.get(section)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing mapping '{section}'")
    missing = [field for field in fields if field not in value]
    if missing:
        raise ConfigError(f"Missing required field(s) in '{section}': {', '.join(missing)}")
    return value


def _positive(mapping: dict[str, Any], fields: tuple[str, ...], section: str) -> None:
    for field in fields:
        value = mapping[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            or (isinstance(value, float) and not isfinite(value))
        ):
            raise ConfigError(f"'{section}.{field}' must be a positive finite number")


def _integer(mapping: dict[str, Any], fields: tuple[str, ...], section: str) -> None:
    for field in fields:
        value = mapping[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (isinstance(value, float) and (not isfinite(value) or not value.is_integer()))
        ):
            raise ConfigError(f"'{section}.{field}' must be a finite integer")


def _positive_integer(
    mapping: dict[str, Any], fields: tuple[str, ...], section: str
) -> None:
    for field in fields:
        value = mapping[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            or (isinstance(value, float) and (not isfinite(value) or not value.is_integer()))
        ):
            raise ConfigError(f"'{section}.{field}' must be a positive integer")


def validate_task_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate task, control, and safety sections without dangerous defaults."""
    task = _required(config, "task", ("name", "instruction", "max_episode_steps"))
    control = _required(
        config,
        "control",
        ("frequency_hz", "action_mode", "action_dim", "state_dim", "chunk_size", "execute_steps"),
    )
    safety = _required(config, "safety", ("max_action_delta", "max_velocity", "timeout_seconds"))
    if not str(task["name"]).strip() or not str(task["instruction"]).strip():
        raise ConfigError("'task.name' and 'task.instruction' must be non-empty")
    _positive_integer(task, ("max_episode_steps",), "task")
    _positive(control, ("frequency_hz",), "control")
    _positive_integer(
        control, ("action_dim", "state_dim", "chunk_size", "execute_steps"), "control"
    )
    _positive(safety, ("max_action_delta", "max_velocity", "timeout_seconds"), "safety")
    if control["action_mode"] not in {"delta_joint", "absolute_joint"}:
        raise ConfigError("'control.action_mode' must be 'delta_joint' or 'absolute_joint'")
    if int(control["execute_steps"]) > int(control["chunk_size"]):
        raise ConfigError("'control.execute_steps' cannot exceed 'control.chunk_size'")
    return config


def validate_robot_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate robot dimensions and joint limits."""
    robot = _required(
        config,
        "robot",
        ("type", "state_dim", "action_dim", "initial_state", "joint_min", "joint_max"),
    )
    _positive_integer(robot, ("state_dim", "action_dim"), "robot")
    state_dim = int(robot["state_dim"])
    for field in ("initial_state", "joint_min", "joint_max"):
        if not isinstance(robot[field], list) or len(robot[field]) != state_dim:
            raise ConfigError(f"'robot.{field}' must contain exactly {state_dim} values")
    if any(
        float(lo) >= float(hi)
        for lo, hi in zip(robot["joint_min"], robot["joint_max"], strict=True)
    ):
        raise ConfigError("Every robot joint_min value must be smaller than joint_max")
    return config


def validate_camera_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate common camera fields."""
    camera = _required(config, "camera", ("type", "name", "width", "height", "fps"))
    _positive_integer(camera, ("width", "height"), "camera")
    _positive(camera, ("fps",), "camera")
    if "seed" in camera:
        _integer(camera, ("seed",), "camera")
    return config


def validate_training_config(config: dict[str, Any], expected_policy: str) -> dict[str, Any]:
    """Validate fields needed to construct a LeRobot training command."""
    training = _required(
        config,
        "training",
        (
            "backend",
            "dataset_root",
            "dataset_repo_id",
            "output_dir",
            "job_name",
            "steps",
            "batch_size",
            "device",
            "seed",
        ),
    )
    policy = _required(config, "policy", ("type", "chunk_size", "n_action_steps"))
    _positive_integer(training, ("steps", "batch_size"), "training")
    _positive_integer(policy, ("chunk_size", "n_action_steps"), "policy")
    if training["backend"] != "lerobot-train":
        raise ConfigError("Only the verified 'lerobot-train' backend is supported")
    if policy["type"] != expected_policy:
        raise ConfigError(f"Expected policy.type='{expected_policy}', got '{policy['type']}'")
    return config
