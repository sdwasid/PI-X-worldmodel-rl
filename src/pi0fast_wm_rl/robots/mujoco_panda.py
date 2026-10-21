"""MuJoCo Franka Panda robot adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseRobot


class MujocoPandaRobot(BaseRobot):
    """Franka Panda implementation backed by MuJoCo."""

    def __init__(
        self,
        model_path: str | Path = "~/mujoco_menagerie/franka_emika_panda/scene.xml",
    ) -> None:
        self.model_path = Path(model_path).expanduser()

        self._mujoco: Any | None = None
        self._model: Any | None = None
        self._data: Any | None = None
        self._qpos_ids: list[int] = []

        self._connected = False
        self._stopped = False
        self._emergency_stopped = False

    def connect(self) -> None:
        if self._connected:
            return

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo Panda model not found: {self.model_path}"
            )

        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MuJoCo is required to use MujocoPandaRobot"
            ) from exc

        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self._data = mujoco.MjData(self._model)

        self._qpos_ids = []
        for index in range(1, 8):
            joint_name = f"joint{index}"
            joint_id = mujoco.mj_name2id(
                self._model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"Missing Panda joint: {joint_name}")

            self._qpos_ids.append(
                int(self._model.jnt_qposadr[joint_id])
            )

        if self._model.nu < 7:
            raise ValueError(
                f"Panda model must expose at least 7 actuators, got {self._model.nu}"
            )

        self._connected = True
        self.reset()

    def disconnect(self) -> None:
        self._data = None
        self._model = None
        self._mujoco = None
        self._qpos_ids = []
        self._connected = False

    def reset(self) -> None:
        self._require_connected()

        mujoco = self._mujoco
        model = self._model
        data = self._data

        home_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            "home",
        )

        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_id)
        else:
            mujoco.mj_resetData(model, data)

        mujoco.mj_forward(model, data)

        self._stopped = False
        self._emergency_stopped = False

    def get_state(self) -> np.ndarray:
        self._require_connected()

        return np.asarray(
            [self._data.qpos[index] for index in self._qpos_ids],
            dtype=np.float64,
        )

    def send_action(self, action: np.ndarray) -> np.ndarray:
        self._require_connected()

        if self._emergency_stopped:
            raise RuntimeError(
                "MujocoPandaRobot emergency stop is active; reset is required"
            )

        if self._stopped:
            raise RuntimeError(
                "MujocoPandaRobot is stopped; reset is required"
            )

        command = np.asarray(action, dtype=np.float64)

        if command.shape != (7,):
            raise ValueError(
                f"action must have shape (7,), got {command.shape}"
            )

        if not np.all(np.isfinite(command)):
            raise ValueError("action contains NaN or Inf")

        self._data.ctrl[:7] = command
        self._mujoco.mj_step(self._model, self._data)

        return command.copy()

    def stop(self) -> None:
        self._stopped = True

    def emergency_stop(self) -> None:
        self._emergency_stopped = True
        self._stopped = True

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "mujoco_panda",
            "state_dim": 7,
            "action_dim": 7,
            "action_mode": "absolute_joint",
            "connected": self._connected,
            "emergency_stopped": self._emergency_stopped,
            "model_path": str(self.model_path),
        }

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MujocoPandaRobot is not connected")