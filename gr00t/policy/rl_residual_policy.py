from __future__ import annotations

from typing import Any

import numpy as np

from gr00t.policy.policy import BasePolicy, PolicyWrapper
from gr00t.rl.residual_adapter import (
    ResidualInferenceModel,
    flatten_action_dict_batch,
    flatten_state_obs_batch,
    unflatten_action_batch,
)


class RLPPOResidualSimPolicyWrapper(PolicyWrapper):
    """Apply a trained PPO residual adapter on top of a sim-compatible GR00T policy.

    This wrapper expects the wrapped policy to use the flat Gr00T sim observation format
    (e.g. `video.*`, `state.*`, and text keys) and to return flat action keys like `action.*`.
    """

    def __init__(
        self,
        policy: BasePolicy,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        strict: bool = True,
    ):
        super().__init__(policy, strict=strict)
        self.model = ResidualInferenceModel(checkpoint_path=checkpoint_path, device=device)

    def check_observation(self, observation: dict[str, Any]) -> None:
        if hasattr(self.policy, "check_observation"):
            self.policy.check_observation(observation)

    def check_action(self, action: dict[str, Any]) -> None:
        if hasattr(self.policy, "check_action"):
            self.policy.check_action(action)

    def get_modality_config(self):
        if hasattr(self.policy, "get_modality_config"):
            return self.policy.get_modality_config()
        return {}

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_action, info = self.policy.get_action(observation, options)

        obs_batch = flatten_state_obs_batch(observation, self.model.obs_keys)
        residual_unit = self.model.predict_residual_unit(obs_batch)
        residual_delta = self.model.scale_and_clip_residual(residual_unit)

        base_action_vec = flatten_action_dict_batch(base_action, self.model.action_specs)
        adapted_action_vec = np.clip(
            base_action_vec + residual_delta,
            self.model.action_low[None, :],
            self.model.action_high[None, :],
        ).astype(np.float32)
        adapted_action = unflatten_action_batch(adapted_action_vec, self.model.action_specs)

        out_info = dict(info) if isinstance(info, dict) else {}
        out_info["rl_residual_enabled"] = True
        out_info["rl_residual_mean_abs"] = float(np.mean(np.abs(residual_delta)))
        return adapted_action, out_info
