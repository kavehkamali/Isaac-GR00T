from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


EPS = 1e-6


@dataclass
class RunningMeanStd:
    """Numerically stable running mean / variance for observation normalization."""

    mean: np.ndarray
    var: np.ndarray
    count: float

    @classmethod
    def create(cls, shape: tuple[int, ...]) -> "RunningMeanStd":
        return cls(
            mean=np.zeros(shape, dtype=np.float64),
            var=np.ones(shape, dtype=np.float64),
            count=1e-4,
        )

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = float(total_count)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + EPS)).astype(np.float32)


@dataclass(frozen=True)
class ArraySpec:
    key: str
    shape: tuple[int, ...]


def _sorted_dict_keys(keys: Iterable[str]) -> list[str]:
    return sorted(list(keys))


def infer_state_obs_keys(observation_space: gym.Space) -> list[str]:
    if not isinstance(observation_space, gym.spaces.Dict):
        raise TypeError(f"Expected Dict observation space, got {type(observation_space)}")
    return [k for k in _sorted_dict_keys(observation_space.spaces.keys()) if k.startswith("state.")]


def infer_action_specs_from_space(action_space: gym.Space) -> list[dict[str, Any]]:
    if not isinstance(action_space, gym.spaces.Dict):
        raise TypeError(f"Expected Dict action space, got {type(action_space)}")
    specs: list[dict[str, Any]] = []
    for key in _sorted_dict_keys(action_space.spaces.keys()):
        space = action_space.spaces[key]
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(f"Only Box action spaces are supported, got {type(space)} for {key}")
        specs.append(
            {
                "key": key,
                "shape": tuple(int(v) for v in space.shape),
                "low": np.asarray(space.low, dtype=np.float32).reshape(-1),
                "high": np.asarray(space.high, dtype=np.float32).reshape(-1),
            }
        )
    return specs


def action_dim_from_specs(action_specs: list[dict[str, Any]]) -> int:
    return int(sum(int(np.prod(spec["shape"], dtype=np.int64)) for spec in action_specs))


def obs_dim_from_keys(observation: dict[str, Any], obs_keys: list[str]) -> int:
    total = 0
    for key in obs_keys:
        total += int(np.asarray(observation[key], dtype=np.float32).size)
    return total


def flatten_state_obs_single(observation: dict[str, Any], obs_keys: list[str]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in obs_keys], axis=0
    ).astype(np.float32)


def flatten_state_obs_batch(observation: dict[str, Any], obs_keys: list[str]) -> np.ndarray:
    chunks = []
    batch_size = None
    for key in obs_keys:
        arr = np.asarray(observation[key], dtype=np.float32)
        if arr.ndim == 0:
            raise ValueError(f"Observation key {key} must have a batch dimension")
        if batch_size is None:
            batch_size = int(arr.shape[0])
        elif int(arr.shape[0]) != batch_size:
            raise ValueError(f"Batch mismatch for key {key}: {arr.shape[0]} != {batch_size}")
        chunks.append(arr.reshape(batch_size, -1))
    if batch_size is None:
        raise ValueError("No state observation keys found")
    return np.concatenate(chunks, axis=1).astype(np.float32)


def flatten_action_dict_single(action: dict[str, np.ndarray], action_specs: list[dict[str, Any]]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(action[spec["key"]], dtype=np.float32).reshape(-1) for spec in action_specs], axis=0
    ).astype(np.float32)


def flatten_action_dict_batch(action: dict[str, np.ndarray], action_specs: list[dict[str, Any]]) -> np.ndarray:
    chunks = []
    batch_size = None
    for spec in action_specs:
        arr = np.asarray(action[spec["key"]], dtype=np.float32)
        if arr.ndim < 1:
            raise ValueError(f"Action key {spec['key']} must have batch dimension")
        if batch_size is None:
            batch_size = int(arr.shape[0])
        elif int(arr.shape[0]) != batch_size:
            raise ValueError(f"Batch mismatch for action {spec['key']}")
        chunks.append(arr.reshape(batch_size, -1))
    if batch_size is None:
        raise ValueError("No action specs provided")
    return np.concatenate(chunks, axis=1).astype(np.float32)


def unflatten_action_single(vec: np.ndarray, action_specs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    out: dict[str, np.ndarray] = {}
    idx = 0
    for spec in action_specs:
        size = int(np.prod(spec["shape"], dtype=np.int64))
        out[spec["key"]] = vec[idx : idx + size].reshape(spec["shape"]).astype(np.float32)
        idx += size
    if idx != vec.size:
        raise ValueError(f"Unused action dimensions: consumed {idx}, got {vec.size}")
    return out


def unflatten_action_batch(vec: np.ndarray, action_specs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim != 2:
        raise ValueError(f"Expected (B, A) residual action batch, got {vec.shape}")
    out: dict[str, np.ndarray] = {}
    idx = 0
    batch_size = vec.shape[0]
    for spec in action_specs:
        size = int(np.prod(spec["shape"], dtype=np.int64))
        out[spec["key"]] = vec[:, idx : idx + size].reshape((batch_size, *spec["shape"])).astype(
            np.float32
        )
        idx += size
    if idx != vec.shape[1]:
        raise ValueError(f"Unused action dimensions: consumed {idx}, got {vec.shape[1]}")
    return out


def action_bounds_from_specs(action_specs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    low = np.concatenate([np.asarray(spec["low"], dtype=np.float32) for spec in action_specs], axis=0)
    high = np.concatenate([np.asarray(spec["high"], dtype=np.float32) for spec in action_specs], axis=0)
    return low.astype(np.float32), high.astype(np.float32)


def action_half_range(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    span = (np.asarray(high, dtype=np.float32) - np.asarray(low, dtype=np.float32)) * 0.5
    span[np.abs(span) < 1e-6] = 1.0
    return span.astype(np.float32)


class ResidualActorCritic(nn.Module):
    """Small PPO actor-critic that predicts residuals over GR00T actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...] = (512, 512)):
        super().__init__()
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError(f"Invalid dims: {obs_dim=}, {act_dim=}")

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(in_dim, act_dim)
        self.value_head = nn.Linear(in_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Slightly conservative initial residuals.
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.zeros_(self.value_head.bias)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def get_dist_and_value(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        feats = self._features(obs)
        mean = self.actor_mean(feats)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        value = self.value_head(feats).squeeze(-1)
        return dist, value

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.get_dist_and_value(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


class ResidualInferenceModel:
    """Loads a trained residual PPO checkpoint and applies deterministic residual actions."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cpu"):
        payload = torch.load(checkpoint_path, map_location=device)
        self.payload = payload
        self.obs_keys: list[str] = list(payload["obs_keys"])
        self.action_specs: list[dict[str, Any]] = list(payload["action_specs"])
        self.obs_mean = np.asarray(payload["obs_mean"], dtype=np.float32)
        self.obs_std = np.asarray(payload["obs_std"], dtype=np.float32)
        self.residual_scale = float(payload.get("residual_scale", 0.25))

        low, high = action_bounds_from_specs(self.action_specs)
        self.action_low = low
        self.action_high = high
        self.action_scale = action_half_range(low, high)

        hidden_dims = tuple(int(v) for v in payload["hidden_dims"])
        obs_dim = int(payload["obs_dim"])
        act_dim = int(payload["act_dim"])
        self.net = ResidualActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims)
        self.net.load_state_dict(payload["model_state_dict"])
        self.net.to(device)
        self.net.eval()
        self.device = torch.device(device)

    def _normalize_obs(self, obs_batch: np.ndarray) -> np.ndarray:
        return ((obs_batch - self.obs_mean) / (self.obs_std + EPS)).astype(np.float32)

    def predict_residual_unit(self, obs_batch: np.ndarray) -> np.ndarray:
        norm_obs = self._normalize_obs(obs_batch)
        obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            action_t, _, _ = self.net.act(obs_t, deterministic=True)
        return action_t.cpu().numpy().astype(np.float32)

    def scale_and_clip_residual(self, residual_unit: np.ndarray) -> np.ndarray:
        residual_unit = np.asarray(residual_unit, dtype=np.float32)
        return (residual_unit * self.residual_scale * self.action_scale).astype(np.float32)


def save_residual_checkpoint(
    path: str | Path,
    *,
    model: ResidualActorCritic,
    obs_keys: list[str],
    action_specs: list[dict[str, Any]],
    obs_rms: RunningMeanStd,
    residual_scale: float,
    hidden_dims: tuple[int, ...],
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    low, high = action_bounds_from_specs(action_specs)
    payload: dict[str, Any] = {
        "version": 1,
        "obs_keys": list(obs_keys),
        "action_specs": [
            {
                "key": spec["key"],
                "shape": tuple(spec["shape"]),
                "low": np.asarray(spec["low"], dtype=np.float32),
                "high": np.asarray(spec["high"], dtype=np.float32),
            }
            for spec in action_specs
        ],
        "obs_mean": obs_rms.mean.astype(np.float32),
        "obs_std": np.sqrt(obs_rms.var + EPS).astype(np.float32),
        "obs_dim": int(obs_rms.mean.size),
        "act_dim": int(low.size),
        "hidden_dims": tuple(int(v) for v in hidden_dims),
        "residual_scale": float(residual_scale),
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
