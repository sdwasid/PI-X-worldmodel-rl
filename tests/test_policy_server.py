from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pi0fast_wm_rl.data.schema import Observation
from pi0fast_wm_rl.inference.policy_server import PolicyService
from pi0fast_wm_rl.policies.base import BasePolicy


class StubPolicy(BasePolicy):
    def __init__(self, action_chunk: np.ndarray, policy_type: str = "stub") -> None:
        self.action_chunk = action_chunk
        self.policy_type = policy_type

    def reset(self) -> None:
        return None

    def predict_action_chunk(self, observation: Observation) -> np.ndarray:
        return self.action_chunk

    def metadata(self) -> dict[str, Any]:
        return {"type": self.policy_type}


def make_observation() -> Observation:
    return Observation(images={}, state=np.zeros(2), timestamp=0.0)


def test_predict_returns_valid_action_chunk_as_float64_with_policy_type() -> None:
    action_chunk = np.array([[1, 2], [3, 4]], dtype=np.int32)
    service = PolicyService(StubPolicy(action_chunk, policy_type="test-policy"), state_dim=2)

    response = service.predict(make_observation())

    np.testing.assert_array_equal(response.action_chunk, action_chunk)
    assert response.action_chunk.dtype == np.float64
    assert response.policy_type == "test-policy"


@pytest.mark.parametrize(
    "action_chunk",
    [
        pytest.param(np.empty((0, 2)), id="empty-chunk-axis"),
        pytest.param(np.empty((3, 0)), id="empty-action-axis"),
        pytest.param(np.empty((0, 0)), id="empty-both-axes"),
        pytest.param(np.array([0.1, 0.2]), id="one-dimensional"),
        pytest.param(np.array([[np.nan, 0.0]]), id="nan"),
        pytest.param(np.array([[np.inf, 0.0]]), id="inf"),
    ],
)
def test_predict_rejects_invalid_action_chunk(action_chunk: np.ndarray) -> None:
    service = PolicyService(StubPolicy(action_chunk), state_dim=2)

    with pytest.raises(ValueError, match="invalid action chunk"):
        service.predict(make_observation())
