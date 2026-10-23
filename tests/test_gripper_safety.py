import pytest
from lerobot.rollout.gripper_safety import (
    GripperSafetyError,
    GripperSafetyGuard,
    GripperStatus,
)


def make_status(**overrides):
    values = {
        "load": 100,
        "current": 100,
        "temperature": 30.0,
        "voltage": 12.0,
        "overload_detected": False,
    }
    values.update(overrides)
    return GripperStatus(**values)


def test_valid_status_passes():
    guard = GripperSafetyGuard()
    assert guard.check(make_status()) is True


def test_missing_status_triggers_fault():
    guard = GripperSafetyGuard()

    with pytest.raises(GripperSafetyError):
        guard.check(None)

    assert guard.fault_latched is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("load", float("nan")),
        ("current", float("inf")),
        ("temperature", float("nan")),
        ("voltage", float("inf")),
    ],
)
def test_nonfinite_status_triggers_fault(field, value):
    guard = GripperSafetyGuard()

    with pytest.raises(GripperSafetyError):
        guard.check(make_status(**{field: value}))

    assert guard.fault_latched is True


def test_invalid_overload_flag_triggers_fault():
    guard = GripperSafetyGuard()

    with pytest.raises(GripperSafetyError):
        guard.check(make_status(overload_detected="false"))

    assert guard.fault_latched is True


def test_overload_triggers_fault():
    guard = GripperSafetyGuard()

    with pytest.raises(GripperSafetyError):
        guard.check(make_status(overload_detected=True))

    assert guard.fault_latched is True


def test_fault_remains_latched():
    guard = GripperSafetyGuard()

    with pytest.raises(GripperSafetyError):
        guard.check(None)

    with pytest.raises(GripperSafetyError, match="GRIPPER LOCKED"):
        guard.check(make_status())
