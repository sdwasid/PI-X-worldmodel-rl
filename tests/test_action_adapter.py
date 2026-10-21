import numpy as np
import pytest

from pi0fast_wm_rl.policies.action_adapter import ActionAdapter, GripperMapping, NormalizationSpec


def test_action_adapter_pads_and_reorders_dimensions() -> None:
    adapter = ActionAdapter(model_dim=5, robot_dim=3, robot_from_model=[2, 0, 4])
    np.testing.assert_allclose(
        adapter.robot_state_to_model(np.array([10, 20, 30])), [20, 0, 10, 0, 30]
    )
    np.testing.assert_allclose(adapter.model_action_to_robot(np.array([1, 2, 3])), [3, 1, 0])


def test_action_adapter_denormalizes_maps_gripper_and_converts_mode() -> None:
    normalization = NormalizationSpec(np.zeros(3), np.full(3, 2.0))
    adapter = ActionAdapter(
        model_dim=3,
        robot_dim=3,
        action_normalization=normalization,
        gripper=GripperMapping(2, 2, (-2.0, 2.0), (0.0, 1.0)),
        model_action_mode="absolute_joint",
        robot_action_mode="delta_joint",
    )
    result = adapter.model_action_to_robot(
        np.array([0.25, -0.25, 0.0]), np.array([0.1, -0.1, 0.25])
    )
    np.testing.assert_allclose(result, [0.4, -0.4, 0.25])


def test_action_adapter_accepts_valid_gripper_mapping() -> None:
    mapping = GripperMapping(1, 2, (-1.0, 1.0), (0.0, 1.0))
    adapter = ActionAdapter(model_dim=3, robot_dim=3, gripper=mapping)

    result = adapter.model_action_to_robot(np.array([0.0, 0.5, 0.0]))

    assert adapter.gripper is mapping
    np.testing.assert_allclose(result, [0.0, 0.5, 0.75])


@pytest.mark.parametrize(
    ("model_index", "robot_index", "field"),
    [
        (-1, 0, "model_index"),
        (0, -1, "robot_index"),
        (3, 0, "model_index"),
        (0, 3, "robot_index"),
    ],
)
def test_action_adapter_rejects_invalid_gripper_indices(
    model_index: int,
    robot_index: int,
    field: str,
) -> None:
    gripper = GripperMapping(model_index, robot_index, (-1.0, 1.0), (0.0, 1.0))

    with pytest.raises(ValueError, match=rf"gripper\.{field}"):
        ActionAdapter(model_dim=3, robot_dim=3, gripper=gripper)


@pytest.mark.parametrize(
    ("field", "bounds"),
    [
        pytest.param("model_range", (np.nan, 1.0), id="model-nan"),
        pytest.param("robot_range", (0.0, np.nan), id="robot-nan"),
        pytest.param("model_range", (0.0, np.inf), id="model-positive-inf"),
        pytest.param("robot_range", (-np.inf, 1.0), id="robot-negative-inf"),
        pytest.param("model_range", (1.0, 1.0), id="model-equal"),
        pytest.param("robot_range", (1.0, 1.0), id="robot-equal"),
        pytest.param("model_range", (1.0, 0.0), id="model-reversed"),
        pytest.param("robot_range", (1.0, 0.0), id="robot-reversed"),
        pytest.param("model_range", ([0.0], [1.0, 2.0]), id="model-ragged"),
        pytest.param("robot_range", ([0.0], [1.0, 2.0]), id="robot-ragged"),
    ],
)
def test_action_adapter_rejects_invalid_gripper_ranges(
    field: str,
    bounds: tuple[float, float],
) -> None:
    ranges = {
        "model_range": (-1.0, 1.0),
        "robot_range": (0.0, 1.0),
    }
    ranges[field] = bounds
    gripper = GripperMapping(
        1,
        2,
        model_range=ranges["model_range"],
        robot_range=ranges["robot_range"],
    )

    with pytest.raises(ValueError, match=rf"gripper\.{field}"):
        ActionAdapter(model_dim=3, robot_dim=3, gripper=gripper)


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("model_action_mode", {"model_action_mode": "typo"}),
        ("robot_action_mode", {"robot_action_mode": "typo"}),
    ],
)
def test_action_adapter_rejects_invalid_action_modes(
    field: str, kwargs: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match=field):
        ActionAdapter(model_dim=3, robot_dim=3, **kwargs)
