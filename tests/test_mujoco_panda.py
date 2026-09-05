from pathlib import Path

import numpy as np
import pytest

from pi0fast_wm_rl.robots.mujoco_panda import MujocoPandaRobot


def test_mujoco_panda_requires_connection_for_state() -> None:
    robot = MujocoPandaRobot()

    with pytest.raises(RuntimeError, match="not connected"):
        robot.get_state()


def test_mujoco_panda_requires_connection_for_action() -> None:
    robot = MujocoPandaRobot()

    with pytest.raises(RuntimeError, match="not connected"):
        robot.send_action(np.zeros(7))


def test_mujoco_panda_rejects_missing_model(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing_scene.xml"
    robot = MujocoPandaRobot(model_path=missing_model)

    with pytest.raises(FileNotFoundError, match="model not found"):
        robot.connect()


def test_mujoco_panda_metadata() -> None:
    robot = MujocoPandaRobot()

    metadata = robot.metadata()

    assert metadata["type"] == "mujoco_panda"
    assert metadata["state_dim"] == 7
    assert metadata["action_dim"] == 7
    assert metadata["action_mode"] == "absolute_joint"
    assert metadata["connected"] is False
    assert metadata["emergency_stopped"] is False