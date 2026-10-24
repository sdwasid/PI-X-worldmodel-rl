import math

JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


class UnsafeActionError(ValueError):
    pass


def validate_action(action):
    """Validate six-joint action before dispatch."""

    if not isinstance(action, dict):
        raise UnsafeActionError(
            "Action must be a dictionary"
        )

    if set(action) != set(JOINT_KEYS):
        raise UnsafeActionError(
            "Joint names mismatch"
        )

    validated = {}

    for joint in JOINT_KEYS:
        value = action[joint]

        if isinstance(value, bool):
            raise UnsafeActionError(
                f"Boolean action not allowed: {joint}"
            )

        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise UnsafeActionError(
                f"Invalid action value: {joint}"
            ) from exc

        if not math.isfinite(numeric):
            raise UnsafeActionError(
                f"Non-finite action: {joint}"
            )

        validated[joint] = numeric

    return validated


def validate_normalized_gripper(action):
    """Check normalized gripper command range."""

    validated = validate_action(action)

    gripper = validated["gripper.pos"]

    if not 0.0 <= gripper <= 100.0:
        raise UnsafeActionError(
            f"Gripper outside [0,100]: {gripper}"
        )

    return validated


if __name__ == "__main__":
    print("===== SAFETY-058 ACTION GUARD TEST =====")

    normal = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -20.0,
        "elbow_flex.pos": 15.0,
        "wrist_flex.pos": 30.0,
        "wrist_roll.pos": 5.0,
        "gripper.pos": 40.0,
    }

    validate_normalized_gripper(normal)
    print("NORMAL: PASS")

    test_cases = {
        "NAN": {
            **normal,
            "elbow_flex.pos": float("nan"),
        },
        "INF": {
            **normal,
            "wrist_flex.pos": float("inf"),
        },
        "MISSING_JOINT": {
            k: v
            for k, v in normal.items()
            if k != "gripper.pos"
        },
        "GRIPPER_OVER_MAX": {
            **normal,
            "gripper.pos": 120.0,
        },
        "GRIPPER_BELOW_MIN": {
            **normal,
            "gripper.pos": -10.0,
        },
    }

    for name, action in test_cases.items():
        try:
            validate_normalized_gripper(action)
        except UnsafeActionError as exc:
            print(f"{name}: PASS - REJECTED ({exc})")
        else:
            raise AssertionError(
                f"{name}: unsafe action accepted"
            )

    print()
    print("===== SAFETY-058 ALL TESTS PASSED =====")
    print("REAL ROBOT CONNECTED: NO")
    print("REAL ROBOT COMMANDS SENT: NONE")
