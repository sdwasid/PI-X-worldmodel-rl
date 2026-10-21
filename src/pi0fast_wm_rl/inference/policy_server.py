"""Transport-neutral policy service boundary for later remote deployment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pi0fast_wm_rl.data.schema import Observation
from pi0fast_wm_rl.policies.base import BasePolicy


@dataclass(frozen=True, slots=True)
class PolicyResponse:
    action_chunk: np.ndarray
    policy_type: str


class PolicyService:
    """Validate observations before forwarding them to a local policy.

    A network transport is deliberately not started in phase one. A future gRPC or
    HTTP layer should call this boundary and add authentication, size limits, and TLS.
    """

    def __init__(self, policy: BasePolicy, state_dim: int) -> None:
        self.policy = policy
        self.state_dim = state_dim

    def predict(self, observation: Observation) -> PolicyResponse:
        if observation.state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape ({self.state_dim},)")
        if not np.all(np.isfinite(observation.state)):
            raise ValueError("Observation state contains NaN or Inf")
        chunk = np.asarray(self.policy.predict_action_chunk(observation), dtype=np.float64)
        if (
            chunk.ndim != 2
            or chunk.shape[0] == 0
            or chunk.shape[1] == 0
            or not np.all(np.isfinite(chunk))
        ):
            raise ValueError("Policy returned an invalid action chunk")
        return PolicyResponse(chunk, str(self.policy.metadata().get("type", "unknown")))
