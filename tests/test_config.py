from __future__ import annotations

from collections.abc import Callable
from math import inf, nan
from typing import Any

import pytest

from pi0fast_wm_rl.utils.config import (
    ConfigError,
    validate_camera_config,
    validate_robot_config,
    validate_task_config,
    validate_training_config,
)

Config = dict[str, Any]
ConfigFactory = Callable[[], Config]
ConfigValidator = Callable[[Config], Config]


def _task_config() -> Config:
    return {
        "task": {
            "name": "pick_place",
            "instruction": "put the cube in the bowl",
            "max_episode_steps": 20,
        },
        "control": {
            "frequency_hz": 10.0,
            "action_mode": "delta_joint",
            "action_dim": 2,
            "state_dim": 2,
            "chunk_size": 4,
            "execute_steps": 1,
        },
        "safety": {
            "max_action_delta": 0.05,
            "max_velocity": 0.5,
            "timeout_seconds": 2.0,
        },
    }


def _robot_config() -> Config:
    return {
        "robot": {
            "type": "mock",
            "state_dim": 3,
            "action_dim": 3,
            "initial_state": [0.0, 0.0, 0.0],
            "joint_min": [-1.0, -1.0, -1.0],
            "joint_max": [1.0, 1.0, 1.0],
        }
    }


def _camera_config() -> Config:
    return {
        "camera": {
            "type": "mock",
            "name": "front",
            "width": 640,
            "height": 480,
            "fps": 30.0,
        }
    }


def _training_config() -> Config:
    return {
        "training": {
            "backend": "lerobot-train",
            "dataset_root": "data/processed/example",
            "dataset_repo_id": "local/example",
            "output_dir": "outputs/example",
            "job_name": "example",
            "steps": 100,
            "batch_size": 2,
            "device": "cpu",
            "seed": 0,
        },
        "policy": {
            "type": "act",
            "chunk_size": 4,
            "n_action_steps": 2,
        },
    }


def _validate_training(config: Config) -> Config:
    return validate_training_config(config, "act")


VALIDATORS: tuple[tuple[ConfigFactory, ConfigValidator], ...] = (
    (_task_config, validate_task_config),
    (_robot_config, validate_robot_config),
    (_camera_config, validate_camera_config),
    (_training_config, _validate_training),
)

INVALID_POSITIVE_NUMBER_CASES: tuple[
    tuple[ConfigFactory, ConfigValidator, str, str, object], ...
] = (
    (_task_config, validate_task_config, "control", "frequency_hz", 0),
    (_task_config, validate_task_config, "safety", "max_action_delta", -1),
    (_task_config, validate_task_config, "safety", "max_velocity", nan),
    (_task_config, validate_task_config, "safety", "timeout_seconds", inf),
    (_camera_config, validate_camera_config, "camera", "fps", -inf),
    (_task_config, validate_task_config, "control", "frequency_hz", "1"),
    (_task_config, validate_task_config, "safety", "max_action_delta", True),
    (_camera_config, validate_camera_config, "camera", "fps", None),
)

POSITIVE_INTEGER_FIELDS: tuple[
    tuple[ConfigFactory, ConfigValidator, str, str], ...
] = (
    (_task_config, validate_task_config, "task", "max_episode_steps"),
    (_task_config, validate_task_config, "control", "action_dim"),
    (_task_config, validate_task_config, "control", "state_dim"),
    (_task_config, validate_task_config, "control", "chunk_size"),
    (_task_config, validate_task_config, "control", "execute_steps"),
    (_robot_config, validate_robot_config, "robot", "state_dim"),
    (_robot_config, validate_robot_config, "robot", "action_dim"),
    (_camera_config, validate_camera_config, "camera", "width"),
    (_camera_config, validate_camera_config, "camera", "height"),
    (_training_config, _validate_training, "training", "steps"),
    (_training_config, _validate_training, "training", "batch_size"),
    (_training_config, _validate_training, "policy", "chunk_size"),
    (_training_config, _validate_training, "policy", "n_action_steps"),
)


@pytest.mark.parametrize(("factory", "validator"), VALIDATORS)
def test_valid_config(factory: ConfigFactory, validator: ConfigValidator) -> None:
    config = factory()

    assert validator(config) == config


@pytest.mark.parametrize(
    ("factory", "validator", "section", "field", "value"), INVALID_POSITIVE_NUMBER_CASES
)
def test_positive_number_fields_reject_invalid_values(
    factory: ConfigFactory,
    validator: ConfigValidator,
    section: str,
    field: str,
    value: object,
) -> None:
    config = factory()
    config[section][field] = value

    with pytest.raises(ConfigError, match=rf"{section}\.{field}"):
        validator(config)


@pytest.mark.parametrize(
    ("factory", "validator", "section", "field"), POSITIVE_INTEGER_FIELDS
)
def test_positive_integer_fields_reject_fractional_values(
    factory: ConfigFactory,
    validator: ConfigValidator,
    section: str,
    field: str,
) -> None:
    config = factory()
    config[section][field] = 1.5

    with pytest.raises(ConfigError, match=rf"{section}\.{field}"):
        validator(config)


@pytest.mark.parametrize(
    ("factory", "validator", "section", "field", "value"),
    [
        (_task_config, validate_task_config, "task", "max_episode_steps", 1.0),
        (_robot_config, validate_robot_config, "robot", "state_dim", 3.0),
    ],
)
def test_positive_integer_fields_accept_whole_floats(
    factory: ConfigFactory,
    validator: ConfigValidator,
    section: str,
    field: str,
    value: float,
) -> None:
    config = factory()
    config[section][field] = value

    assert validator(config) == config


@pytest.mark.parametrize("value", [0, -1, nan, inf, -inf, 3.5, "3", True])
def test_positive_integer_rejects_invalid_types_and_values(value: object) -> None:
    config = _training_config()
    config["training"]["steps"] = value

    with pytest.raises(ConfigError, match=r"training\.steps"):
        _validate_training(config)


@pytest.mark.parametrize("seed", [7, 0, 3.0, -1])
def test_camera_seed_accepts_integer_values(seed: int | float) -> None:
    config = _camera_config()
    config["camera"]["seed"] = seed

    assert validate_camera_config(config) == config


@pytest.mark.parametrize("seed", [3.5, nan, inf, -inf, True, "3"])
def test_camera_seed_rejects_non_integer_values(seed: object) -> None:
    config = _camera_config()
    config["camera"]["seed"] = seed

    with pytest.raises(ConfigError, match=r"camera\.seed"):
        validate_camera_config(config)


def test_execute_steps_may_equal_chunk_size() -> None:
    config = _task_config()
    config["control"]["chunk_size"] = 3.0
    config["control"]["execute_steps"] = 3.0

    assert validate_task_config(config) == config


def test_execute_steps_cannot_exceed_chunk_size() -> None:
    config = _task_config()
    config["control"]["execute_steps"] = config["control"]["chunk_size"] + 1

    with pytest.raises(ConfigError, match="execute_steps.*cannot exceed.*chunk_size"):
        validate_task_config(config)


def test_n_action_steps_may_exceed_chunk_size() -> None:
    config = _training_config()
    config["policy"]["n_action_steps"] = config["policy"]["chunk_size"] + 1

    assert _validate_training(config) == config
