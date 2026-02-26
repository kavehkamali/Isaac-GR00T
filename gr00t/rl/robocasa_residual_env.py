from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.rollout_policy import WrapperConfigs, create_eval_env
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.rl.residual_adapter import (
    action_bounds_from_specs,
    action_half_range,
    flatten_action_dict_single,
    flatten_state_obs_single,
    infer_action_specs_from_space,
    infer_state_obs_keys,
    obs_dim_from_keys,
    unflatten_action_single,
)


@dataclass
class RewardShapingConfig:
    base_reward_scale: float = 1.0
    success_bonus: float = 10.0
    height_coef: float = 25.0
    hold_progress_coef: float = 2.0
    action_l2_coef: float = 0.02


class RobocasaGr00tResidualPPOEnv(gym.Env):
    """Gym env for PPO residual training on top of GR00T on RoboCasa eval wrappers.

    The agent outputs a normalized residual action in [-1, 1]. This is scaled and added to
    GR00T's action chunk before stepping the same wrapped eval env used by rollout_policy.py.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_name: str,
        base_model_path: str,
        embodiment_tag: EmbodimentTag,
        device: str = "cuda",
        n_action_steps: int = 8,
        max_episode_steps: int = 600,
        residual_scale: float = 0.25,
        terminate_on_success: bool = True,
        strict_policy: bool = False,
        reward_shaping: RewardShapingConfig | None = None,
        env_idx: int = 0,
        total_n_envs: int = 1,
    ):
        super().__init__()
        self.env_name = env_name
        self.base_model_path = base_model_path
        self.embodiment_tag = embodiment_tag
        self.device = device
        self.residual_scale = float(residual_scale)
        self.reward_shaping = reward_shaping or RewardShapingConfig()

        wrapper_configs = WrapperConfigs()
        wrapper_configs.video.video_dir = None
        wrapper_configs.multistep.n_action_steps = int(n_action_steps)
        wrapper_configs.multistep.max_episode_steps = int(max_episode_steps)
        wrapper_configs.multistep.terminate_on_success = bool(terminate_on_success)
        self.inner_env = create_eval_env(
            env_name=env_name,
            env_idx=env_idx,
            total_n_envs=total_n_envs,
            wrapper_configs=wrapper_configs,
        )

        base_policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=base_model_path,
            device=device,
            strict=strict_policy,
        )
        self.policy = Gr00tSimPolicyWrapper(base_policy, strict=strict_policy)

        self.obs_keys = infer_state_obs_keys(self.inner_env.observation_space)
        self.action_specs = infer_action_specs_from_space(self.inner_env.action_space)
        low, high = action_bounds_from_specs(self.action_specs)
        self._action_low = low
        self._action_high = high
        self._action_scale = action_half_range(low, high)

        # Initialize a sample observation to infer observation dimensions.
        obs0, _info0 = self.inner_env.reset(seed=0)
        self.policy.reset()
        obs_dim = obs_dim_from_keys(obs0, self.obs_keys)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(low.size,),
            dtype=np.float32,
        )

        self._last_obs_raw: dict[str, Any] | None = obs0
        self._episode_step = 0
        self._episode_return = 0.0
        self._episode_success = False

    def _make_policy_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        batched: dict[str, Any] = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                batched[key] = np.expand_dims(value, axis=0)
            elif isinstance(value, str):
                batched[key] = [value]
            elif isinstance(value, (tuple, list)):
                if len(value) == 0:
                    batched[key] = [""]
                elif isinstance(value[0], str):
                    batched[key] = list(value)
                else:
                    batched[key] = [value]
            else:
                batched[key] = [value]
        return batched

    def _get_kitchen_env(self):
        base = self.inner_env.unwrapped
        return getattr(base, "env", None)

    def _dense_height_reward(self) -> tuple[float, float]:
        kitchen = self._get_kitchen_env()
        if kitchen is None or not hasattr(kitchen, "obj_body_id") or "obj" not in kitchen.obj_body_id:
            return 0.0, 0.0
        if not hasattr(kitchen, "_obj_init_z") or getattr(kitchen, "_obj_init_z") is None:
            return 0.0, 0.0
        obj_z = float(kitchen.sim.data.body_xpos[kitchen.obj_body_id["obj"]][2])
        init_z = float(kitchen._obj_init_z)
        height_delta = max(0.0, obj_z - init_z)
        hold_count = float(getattr(kitchen, "_lift_hold_count", 0.0))
        hold_steps = float(max(1, int(getattr(kitchen, "LIFT_HOLD_STEPS", 1))))
        hold_progress = min(1.0, hold_count / hold_steps)
        return height_delta, hold_progress

    @staticmethod
    def _info_success(info: dict[str, Any]) -> bool:
        success = info.get("success", False)
        if isinstance(success, np.ndarray):
            return bool(np.max(success))
        if isinstance(success, (list, tuple)):
            return any(bool(v) for v in success)
        return bool(success)

    def _shape_reward(self, base_reward: float, info: dict[str, Any], residual_delta: np.ndarray) -> tuple[float, dict[str, float]]:
        cfg = self.reward_shaping
        success = self._info_success(info)
        height_delta, hold_progress = self._dense_height_reward()
        action_penalty = float(np.mean(np.square(residual_delta), dtype=np.float32))
        shaped = (
            cfg.base_reward_scale * float(base_reward)
            + cfg.success_bonus * float(success)
            + cfg.height_coef * float(height_delta)
            + cfg.hold_progress_coef * float(hold_progress)
            - cfg.action_l2_coef * action_penalty
        )
        components = {
            "base": float(base_reward),
            "success": float(success),
            "height_delta": float(height_delta),
            "hold_progress": float(hold_progress),
            "residual_l2": float(action_penalty),
            "shaped": float(shaped),
        }
        return float(shaped), components

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.policy.reset()
        obs, info = self.inner_env.reset(seed=seed, options=options)
        self._last_obs_raw = obs
        self._episode_step = 0
        self._episode_return = 0.0
        self._episode_success = False
        flat_obs = flatten_state_obs_single(obs, self.obs_keys)
        return flat_obs, info

    def step(self, action: np.ndarray):
        if self._last_obs_raw is None:
            raise RuntimeError("Environment must be reset() before step()")

        residual_unit = np.asarray(action, dtype=np.float32).reshape(-1)
        residual_unit = np.clip(residual_unit, -1.0, 1.0)

        policy_obs = self._make_policy_obs(self._last_obs_raw)
        base_action_batched, _ = self.policy.get_action(policy_obs)
        base_action_single = {
            key: np.asarray(value[0], dtype=np.float32)
            for key, value in base_action_batched.items()
        }

        base_action_vec = flatten_action_dict_single(base_action_single, self.action_specs)
        residual_delta = residual_unit * self.residual_scale * self._action_scale
        adapted_action_vec = np.clip(
            base_action_vec + residual_delta,
            self._action_low,
            self._action_high,
        ).astype(np.float32)
        adapted_action = unflatten_action_single(adapted_action_vec, self.action_specs)

        base_abs = np.abs(base_action_vec)
        resid_abs = np.abs(residual_delta)
        adapted_abs = np.abs(adapted_action_vec)
        denom = float(np.mean(base_abs))
        residual_stats = {
            "base_action_mean_abs": float(np.mean(base_abs)),
            "base_action_max_abs": float(np.max(base_abs)),
            "residual_mean_abs": float(np.mean(resid_abs)),
            "residual_max_abs": float(np.max(resid_abs)),
            "residual_l2": float(np.sqrt(np.mean(np.square(residual_delta), dtype=np.float32))),
            "adapted_action_mean_abs": float(np.mean(adapted_abs)),
            "adapted_action_max_abs": float(np.max(adapted_abs)),
            "residual_to_base_ratio": float(np.mean(resid_abs) / max(denom, 1e-6)),
        }

        obs, reward, terminated, truncated, info = self.inner_env.step(adapted_action)
        shaped_reward, reward_components = self._shape_reward(reward, info, residual_delta)

        self._last_obs_raw = obs
        self._episode_step += 1
        self._episode_return += shaped_reward
        self._episode_success = self._episode_success or self._info_success(info)

        if terminated or truncated:
            info = dict(info)
            info["episode_return_shaped"] = float(self._episode_return)
            info["episode_success"] = bool(self._episode_success)
            info["episode_steps"] = int(self._episode_step)

        info = dict(info)
        info["rl_reward_components"] = reward_components
        info["rl_residual_stats"] = residual_stats

        flat_obs = flatten_state_obs_single(obs, self.obs_keys)
        return flat_obs, float(shaped_reward), bool(terminated), bool(truncated), info

    def close(self):
        self.inner_env.close()
