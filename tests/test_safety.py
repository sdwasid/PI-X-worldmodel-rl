from collections.abc import Callable

import numpy as np
import pytest

from pi0fast_wm_rl.robots.safety import SafetyFilter, SafetyViolation


def make_filter(
    *,
    max_action_delta: float = 0.2,
    max_velocity: float = 1.0,
    control_hz: float = 10.0,
    timeout_seconds: float = 1.0,
    action_mode: str = "delta_joint",
    workspace_hook: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> SafetyFilter:
    return SafetyFilter(
        -np.ones(2),
        np.ones(2),
        max_action_delta=max_action_delta,
        max_velocity=max_velocity,
        control_hz=control_hz,
        timeout_seconds=timeout_seconds,
        action_mode=action_mode,
        workspace_hook=workspace_hook,
    )


@pytest.mark.parametrize(
    ("joint_min", "joint_max", "message"),
    [
        (np.zeros((1, 2)), np.ones((1, 2)), "same-sized vectors"),
        (np.zeros(2), np.ones(3), "same-sized vectors"),
        (np.array([-1.0, 0.0]), np.array([1.0, 0.0]), "smaller than joint_max"),
    ],
)
def test_safety_filter_rejects_invalid_joint_limits(
    joint_min: np.ndarray,
    joint_max: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SafetyFilter(joint_min, joint_max, 0.2, 1.0, 10.0, 1.0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_action_delta", 0.0),
        ("max_velocity", -1.0),
        ("control_hz", 0.0),
        ("timeout_seconds", -1.0),
    ],
)
def test_safety_filter_rejects_nonpositive_safety_values(name: str, value: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        make_filter(**{name: value})


def test_safety_filter_rejects_unsupported_action_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported action_mode"):
        make_filter(action_mode="cartesian")


def test_safety_filter_clips_delta_velocity_and_joint_limits() -> None:
    safety = make_filter()
    result = safety.filter(np.array([0.5, -0.5]), np.array([0.95, -0.95]))
    np.testing.assert_allclose(result.action, [0.05, -0.05])
    assert result.clipped
    assert result.clipped_values == 2
    assert len(result.reasons) == 3
    assert set(result.reasons) == {"max_action_delta", "max_velocity", "joint_limits"}
    assert safety.total_clipped_values == 2


def test_safety_filter_passes_through_safe_delta_without_clipping() -> None:
    safety = make_filter()

    result = safety.filter(np.array([0.05, -0.05]), np.zeros(2))

    np.testing.assert_allclose(result.action, [0.05, -0.05])
    assert result.clipped_values == 0
    assert not result.clipped
    assert result.reasons == ()
    assert safety.total_clipped_values == 0


def test_safety_filter_allows_values_exactly_at_safety_limits() -> None:
    safety = make_filter(max_action_delta=0.1)

    result = safety.filter(np.array([0.1, -0.1]), np.array([0.9, -0.9]))

    np.testing.assert_allclose(result.action, [0.1, -0.1])
    assert result.reasons == ()
    assert not result.clipped


def test_safety_filter_supports_absolute_joint_actions() -> None:
    safety = make_filter(action_mode="absolute_joint")

    result = safety.filter(np.array([0.9, -0.9]), np.array([0.4, -0.4]))

    np.testing.assert_allclose(result.action, [0.5, -0.5])
    assert result.clipped_values == 2
    assert len(result.reasons) == 2
    assert set(result.reasons) == {"max_action_delta", "max_velocity"}


@pytest.mark.parametrize(
    ("action", "state", "message"),
    [
        (np.array([np.nan, 0.0]), np.zeros(2), "Action contains NaN or Inf"),
        (np.array([np.inf, 0.0]), np.zeros(2), "Action contains NaN or Inf"),
        (np.zeros(2), np.array([np.nan, 0.0]), "Robot state contains NaN or Inf"),
        (np.zeros(2), np.array([np.inf, 0.0]), "Robot state contains NaN or Inf"),
    ],
)
def test_safety_filter_rejects_nonfinite_values(
    action: np.ndarray,
    state: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(SafetyViolation, match=message):
        make_filter().filter(action, state)


@pytest.mark.parametrize(
    ("action", "state"),
    [
        (np.zeros(3), np.zeros(2)),
        (np.zeros(2), np.zeros((1, 2))),
    ],
)
def test_safety_filter_rejects_invalid_action_or_state_shape(
    action: np.ndarray,
    state: np.ndarray,
) -> None:
    with pytest.raises(SafetyViolation, match=r"must have shape \(2,\)"):
        make_filter().filter(action, state)


def test_safety_filter_accumulates_clipped_values_and_reset_clears_accounting() -> None:
    safety = make_filter()
    first = safety.filter(np.array([0.5, 0.0]), np.zeros(2))
    second = safety.filter(np.array([0.0, -0.5]), np.zeros(2))

    assert first.clipped_values == 1
    assert second.clipped_values == 1
    assert safety.total_clipped_values == 2

    safety.reset()

    assert safety.total_clipped_values == 0
    result = safety.filter(np.zeros(2), np.zeros(2), now=100.0)
    assert not result.clipped


def test_safety_filter_workspace_hook_receives_copies_of_limited_target_and_state() -> None:
    received: list[tuple[np.ndarray, np.ndarray]] = []

    def accept_and_mutate(target: np.ndarray, state: np.ndarray) -> bool:
        received.append((target.copy(), state.copy()))
        target[:] = 99.0
        state[:] = 99.0
        return True

    safety = make_filter(workspace_hook=accept_and_mutate)
    state = np.array([0.95, 0.0])

    result = safety.filter(np.array([0.5, 0.0]), state)

    assert len(received) == 1
    np.testing.assert_allclose(received[0][0], [1.0, 0.0])
    np.testing.assert_allclose(received[0][1], state)
    np.testing.assert_allclose(result.action, [0.05, 0.0])
    np.testing.assert_allclose(state, [0.95, 0.0])


def test_safety_filter_workspace_hook_can_reject_limited_target() -> None:
    received: list[tuple[np.ndarray, np.ndarray]] = []

    def reject(target: np.ndarray, state: np.ndarray) -> bool:
        received.append((target, state))
        return False

    safety = make_filter(workspace_hook=reject)

    with pytest.raises(SafetyViolation, match="Workspace safety hook rejected"):
        safety.filter(np.array([0.5, 0.0]), np.array([0.95, 0.0]))

    np.testing.assert_allclose(received[0][0], [1.0, 0.0])
    np.testing.assert_allclose(received[0][1], [0.95, 0.0])
    assert safety.total_clipped_values == 0


def test_safety_filter_timeout_uses_strict_boundary_and_reset_clears_timeout() -> None:
    safety = make_filter(timeout_seconds=0.5)
    safety.heartbeat(now=1.0)

    result = safety.filter(np.zeros(2), np.zeros(2), now=1.5)
    assert not result.clipped
    assert not safety.timed_out

    with pytest.raises(SafetyViolation, match="timeout"):
        safety.filter(np.zeros(2), np.zeros(2), now=1.6)

    assert safety.timed_out
    with pytest.raises(SafetyViolation, match="timeout"):
        safety.filter(np.zeros(2), np.zeros(2), now=1.7)

    safety.reset()
    assert not safety.timed_out
    assert safety.last_communication_time is None
    result = safety.filter(np.zeros(2), np.zeros(2), now=10.0)
    assert not result.clipped


def test_safety_filter_emergency_stop_latches_until_reset() -> None:
    safety = make_filter()
    safety.emergency_stop()

    with pytest.raises(SafetyViolation, match="Emergency"):
        safety.filter(np.zeros(2), np.zeros(2))

    safety.reset()

    result = safety.filter(np.zeros(2), np.zeros(2))
    assert not result.clipped
