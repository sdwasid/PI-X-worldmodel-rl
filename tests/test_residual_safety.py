import numpy as np
import pytest
from lerobot.rollout import lateral_residual_control as lateral_rc
from lerobot.rollout import residual_control as lerobot_rc

from so101_residual import residual_control as project_rc


def test_lerobot_zero_active():
    assert lerobot_rc.ACTIVE_SCALE == 0.0
    assert lateral_rc.LATERAL_ACTIVE_SCALE == 0.0

    act = np.array(
        [10, -20, 15, 30, 5, 40],
        dtype=np.float32,
    )

    candidate = np.array(
        [0.5, -2.0, 1.0, -0.5, 0.3, -0.5],
        dtype=np.float32,
    )

    final, active = lerobot_rc.apply_residual(
        act,
        candidate,
    )

    np.testing.assert_array_equal(final, act)
    np.testing.assert_array_equal(active, np.zeros(6))


def test_lateral_zero_active():
    act = np.array(
        [10, -20, 15, 30, 5, 40],
        dtype=np.float32,
    )

    candidate = np.array(
        [0.5, -0.5, 0.5, -0.5, 0.2, -0.2],
        dtype=np.float32,
    )

    final, active = lateral_rc.apply_lateral_residual(
        act,
        candidate,
    )

    np.testing.assert_array_equal(final, act)
    np.testing.assert_array_equal(active, np.zeros(6))


def test_project_active_001():
    assert project_rc.ACTIVE_SCALE == 0.0

    act = np.array(
        [10, -20, 15, 30, 5, 40],
        dtype=np.float32,
    )

    candidate = np.array(
        [100, -100, 100, -100, 100, -100],
        dtype=np.float32,
    )

    final, active = project_rc.apply_residual(
        act,
        candidate,
    )

    clipped = np.clip(
        candidate,
        -project_rc.RESIDUAL_LIMITS,
        project_rc.RESIDUAL_LIMITS,
    )

    expected_active = clipped * project_rc.ACTIVE_SCALE

    np.testing.assert_allclose(
        active,
        expected_active,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        final,
        act + expected_active,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "invalid",
    [np.nan, np.inf, -np.inf],
)
def test_invalid_act_rejected(invalid):
    act = np.zeros(6, dtype=np.float32)
    candidate = np.zeros(6, dtype=np.float32)

    act[0] = invalid

    with pytest.raises(ValueError):
        project_rc.apply_residual(act, candidate)


@pytest.mark.parametrize(
    "invalid",
    [np.nan, np.inf, -np.inf],
)
def test_invalid_residual_rejected(invalid):
    act = np.zeros(6, dtype=np.float32)
    candidate = np.zeros(6, dtype=np.float32)

    candidate[0] = invalid

    with pytest.raises(ValueError):
        project_rc.apply_residual(act, candidate)
