"""Launch PPO-based RL residual fine-tuning for GR00T on RoboCasa.

This trains a small residual policy on top of a frozen GR00T policy using PPO. The trained
checkpoint can be loaded by `gr00t/eval/run_gr00t_server.py` via `--rl-residual-checkpoint`
while keeping the same `rollout_policy.py` client evaluation flow.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
import tyro

from gr00t.configs.rl_finetune_config import RLFineTuneConfig
from gr00t.rl.residual_adapter import RunningMeanStd, ResidualActorCritic, save_residual_checkpoint
from gr00t.rl.robocasa_residual_env import RewardShapingConfig, RobocasaGr00tResidualPPOEnv


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_nonterminal = 1.0 - float(dones[t])
            next_values = float(next_value)
        else:
            next_nonterminal = 1.0 - float(dones[t])
            next_values = float(values[t + 1])
        delta = float(rewards[t]) + gamma * next_values * next_nonterminal - float(values[t])
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def maybe_init_wandb(cfg: RLFineTuneConfig, output_dir: Path):
    if not cfg.use_wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError as e:
        raise ImportError(
            "wandb is not installed. Install it in your training env (e.g. pip install wandb) or set --use-wandb False."
        ) from e

    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        group=cfg.wandb_group,
        name=cfg.wandb_run_name,
        tags=cfg.wandb_tags,
        config=asdict(cfg),
        dir=str(output_dir),
        save_code=False,
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/update")
    wandb.define_metric("*", step_metric="train/global_step")
    return run


def wandb_log(run, metrics: dict[str, Any]) -> None:
    if run is None:
        return
    run.log(metrics)


def evaluate_deterministic(
    env: RobocasaGr00tResidualPPOEnv,
    model: ResidualActorCritic,
    obs_rms: RunningMeanStd,
    episodes: int,
    seed_start: int,
    device: torch.device,
) -> dict[str, float]:
    if episodes <= 0:
        return {}

    model.eval()
    successes = 0
    returns = []
    lengths = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_len = 0
        while not (done or truncated):
            obs_norm = obs_rms.normalize(obs)
            obs_t = torch.as_tensor(obs_norm[None, :], dtype=torch.float32, device=device)
            with torch.inference_mode():
                action_t, _, _ = model.act(obs_t, deterministic=True)
            action = action_t.cpu().numpy()[0].astype(np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
        successes += int(bool(info.get("episode_success", False)))
        returns.append(ep_ret)
        lengths.append(ep_len)
    model.train()
    return {
        "eval/success_rate": float(successes / max(1, episodes)),
        "eval/return_mean": float(np.mean(returns) if returns else 0.0),
        "eval/return_std": float(np.std(returns) if returns else 0.0),
        "eval/episode_len_mean": float(np.mean(lengths) if lengths else 0.0),
    }


def collect_info_metric(step_infos: list[dict[str, Any]], key: str, subkey: str) -> float | None:
    vals = []
    for info in step_infos:
        d = info.get(key)
        if isinstance(d, dict) and subkey in d:
            vals.append(float(d[subkey]))
    if not vals:
        return None
    return float(np.mean(vals))


def main(cfg: RLFineTuneConfig) -> None:
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    set_global_seed(cfg.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)
    hidden_dims = tuple([cfg.hidden_dim] * cfg.hidden_layers)

    reward_shaping = RewardShapingConfig(
        success_bonus=cfg.success_bonus,
        height_coef=cfg.height_coef,
        hold_progress_coef=cfg.hold_progress_coef,
        action_l2_coef=cfg.action_l2_coef,
    )

    print("Starting GR00T RL residual fine-tuning (PPO)...")
    print(f"  Env: {cfg.env_name}")
    print(f"  Base model: {cfg.base_model_path}")
    print(f"  Embodiment: {cfg.embodiment_tag}")
    print(f"  Device: {device}")
    print(f"  Output dir: {output_dir}")
    if cfg.use_wandb:
        print(f"  wandb: {cfg.wandb_project} (entity={cfg.wandb_entity})")

    env = RobocasaGr00tResidualPPOEnv(
        env_name=cfg.env_name,
        base_model_path=cfg.base_model_path,
        embodiment_tag=cfg.embodiment_tag,
        device=cfg.device,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=cfg.max_episode_steps,
        residual_scale=cfg.residual_scale,
        reward_shaping=reward_shaping,
    )

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    model = ResidualActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    obs_rms = RunningMeanStd.create((obs_dim,))

    metrics_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "rl_finetune_config.json"
    with config_path.open("w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    wandb_run = maybe_init_wandb(cfg, output_dir)

    global_step = 0
    update_idx = 0
    best_eval_success = -1.0
    recent_returns: deque[float] = deque(maxlen=20)
    recent_successes: deque[float] = deque(maxlen=20)
    recent_lengths: deque[float] = deque(maxlen=20)
    train_start = time.time()

    obs, _ = env.reset(seed=cfg.seed)
    obs_rms.update(obs)

    try:
        while global_step < cfg.total_timesteps:
            update_idx += 1

            obs_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
            obs_norm_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
            act_buf = np.zeros((cfg.rollout_steps, act_dim), dtype=np.float32)
            logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
            rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
            done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
            val_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
            step_infos: list[dict[str, Any]] = []

            rollout_len = 0
            for t in range(cfg.rollout_steps):
                if global_step >= cfg.total_timesteps:
                    break

                obs_buf[t] = obs
                obs_norm = obs_rms.normalize(obs)
                obs_norm_buf[t] = obs_norm
                obs_t = torch.as_tensor(obs_norm[None, :], dtype=torch.float32, device=device)
                with torch.inference_mode():
                    action_t, logp_t, value_t = model.act(obs_t, deterministic=False)
                action = action_t.cpu().numpy()[0].astype(np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = bool(terminated or truncated)

                act_buf[t] = action
                logp_buf[t] = float(logp_t.cpu().numpy()[0])
                val_buf[t] = float(value_t.cpu().numpy()[0])
                rew_buf[t] = float(reward)
                done_buf[t] = float(done)
                step_infos.append(step_info)

                rollout_len += 1
                global_step += 1
                obs = next_obs
                obs_rms.update(obs)

                if done:
                    recent_returns.append(float(step_info.get("episode_return_shaped", 0.0)))
                    recent_successes.append(float(step_info.get("episode_success", False)))
                    recent_lengths.append(float(step_info.get("episode_steps", 0)))
                    obs, _ = env.reset(seed=cfg.seed + global_step + update_idx)
                    obs_rms.update(obs)

            if rollout_len == 0:
                break

            obs_norm = obs_rms.normalize(obs)
            obs_t = torch.as_tensor(obs_norm[None, :], dtype=torch.float32, device=device)
            with torch.inference_mode():
                _, _, next_value_t = model.act(obs_t, deterministic=True)
            next_value = float(next_value_t.cpu().numpy()[0])

            advantages, returns = compute_gae(
                rewards=rew_buf[:rollout_len],
                values=val_buf[:rollout_len],
                dones=done_buf[:rollout_len],
                next_value=next_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            advantages_raw = advantages.copy()
            adv_mean = float(np.mean(advantages))
            adv_std = float(np.std(advantages) + 1e-8)
            advantages = ((advantages - adv_mean) / adv_std).astype(np.float32)

            b_obs_norm = torch.as_tensor(obs_norm_buf[:rollout_len], dtype=torch.float32, device=device)
            b_actions = torch.as_tensor(act_buf[:rollout_len], dtype=torch.float32, device=device)
            b_logp_old = torch.as_tensor(logp_buf[:rollout_len], dtype=torch.float32, device=device)
            b_adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
            b_returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
            b_values_old = torch.as_tensor(val_buf[:rollout_len], dtype=torch.float32, device=device)

            batch_size = rollout_len
            mb_size = max(1, min(cfg.mini_batch_size, batch_size))
            inds = np.arange(batch_size)
            last_loss: dict[str, float] = {}

            for _epoch in range(cfg.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, mb_size):
                    mb_inds = inds[start : start + mb_size]
                    mb_obs = b_obs_norm[mb_inds]
                    mb_actions = b_actions[mb_inds]
                    mb_logp_old = b_logp_old[mb_inds]
                    mb_adv = b_adv[mb_inds]
                    mb_returns = b_returns[mb_inds]
                    mb_values_old = b_values_old[mb_inds]

                    new_logp, entropy, value = model.evaluate_actions(mb_obs, mb_actions)
                    logratio = new_logp - mb_logp_old
                    ratio = torch.exp(logratio)

                    pg_loss_1 = -mb_adv * ratio
                    pg_loss_2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                    v_loss_unclipped = torch.square(value - mb_returns)
                    v_clipped = mb_values_old + torch.clamp(
                        value - mb_values_old, -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss_clipped = torch.square(v_clipped - mb_returns)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_mean = entropy.mean()
                    loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy_mean

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                    approx_kl = ((ratio - 1.0) - logratio).mean().detach()
                    clipfrac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_coef).float()).detach()
                    last_loss = {
                        "ppo/loss_total": float(loss.detach().cpu()),
                        "ppo/policy_loss": float(pg_loss.detach().cpu()),
                        "ppo/value_loss": float(v_loss.detach().cpu()),
                        "ppo/entropy": float(entropy_mean.detach().cpu()),
                        "ppo/approx_kl": float(approx_kl.cpu()),
                        "ppo/clipfrac": float(clipfrac.cpu()),
                    }

            reward_comp_means = {
                f"reward/{name}": collect_info_metric(step_infos, "rl_reward_components", name)
                for name in [
                    "base",
                    "success",
                    "height_delta",
                    "hold_progress",
                    "residual_l2",
                    "shaped",
                ]
            }
            reward_comp_means = {k: v for k, v in reward_comp_means.items() if v is not None}

            residual_metric_names = [
                "base_action_mean_abs",
                "base_action_max_abs",
                "residual_mean_abs",
                "residual_max_abs",
                "residual_l2",
                "adapted_action_mean_abs",
                "adapted_action_max_abs",
                "residual_to_base_ratio",
            ]
            residual_metrics = {
                f"vla_residual/{name}": collect_info_metric(step_infos, "rl_residual_stats", name)
                for name in residual_metric_names
            }
            residual_metrics = {k: v for k, v in residual_metrics.items() if v is not None}

            returns_np = returns.astype(np.float32)
            values_np = val_buf[:rollout_len].astype(np.float32)
            metrics: dict[str, Any] = {
                "train/update": update_idx,
                "train/global_step": global_step,
                "train/rollout_len": rollout_len,
                "train/fps": float(global_step / max(1e-6, (time.time() - train_start))),
                "rollout/reward_mean": float(np.mean(rew_buf[:rollout_len])),
                "rollout/reward_std": float(np.std(rew_buf[:rollout_len])),
                "rollout/done_rate": float(np.mean(done_buf[:rollout_len])),
                "rollout/recent_episode_return_mean": float(np.mean(recent_returns)) if recent_returns else None,
                "rollout/recent_episode_success_rate": float(np.mean(recent_successes)) if recent_successes else None,
                "rollout/recent_episode_len_mean": float(np.mean(recent_lengths)) if recent_lengths else None,
                "ppo/advantages_mean_raw": float(np.mean(advantages_raw)),
                "ppo/advantages_std_raw": float(np.std(advantages_raw)),
                "ppo/returns_mean": float(np.mean(returns_np)),
                "ppo/returns_std": float(np.std(returns_np)),
                "ppo/values_mean": float(np.mean(values_np)),
                "ppo/explained_variance": explained_variance(values_np, returns_np),
                "optim/lr": float(optimizer.param_groups[0]["lr"]),
                "task/n_action_steps": cfg.n_action_steps,
                "task/max_episode_steps": cfg.max_episode_steps,
                **last_loss,
                **reward_comp_means,
                **residual_metrics,
            }
            metrics = {k: v for k, v in metrics.items() if v is not None}

            if cfg.eval_episodes > 0 and (update_idx % max(1, cfg.eval_every_updates) == 0):
                eval_metrics = evaluate_deterministic(
                    env=env,
                    model=model,
                    obs_rms=obs_rms,
                    episodes=cfg.eval_episodes,
                    seed_start=cfg.seed + 100000 + update_idx * 100,
                    device=device,
                )
                metrics.update(eval_metrics)
                eval_success = eval_metrics.get("eval/success_rate", -1.0)
                if eval_success > best_eval_success:
                    best_eval_success = eval_success
                    save_residual_checkpoint(
                        output_dir / "checkpoints" / "residual_best.pt",
                        model=model,
                        obs_keys=env.obs_keys,
                        action_specs=env.action_specs,
                        obs_rms=obs_rms,
                        residual_scale=cfg.residual_scale,
                        hidden_dims=hidden_dims,
                        metadata={
                            "base_model_path": cfg.base_model_path,
                            "env_name": cfg.env_name,
                            "embodiment_tag": cfg.embodiment_tag.value,
                            "n_action_steps": cfg.n_action_steps,
                            "max_episode_steps": cfg.max_episode_steps,
                            "best_eval_success_rate": float(best_eval_success),
                            "update": update_idx,
                        },
                    )

            if update_idx % max(1, cfg.save_every_updates) == 0 or global_step >= cfg.total_timesteps:
                save_residual_checkpoint(
                    output_dir / "checkpoints" / "residual_latest.pt",
                    model=model,
                    obs_keys=env.obs_keys,
                    action_specs=env.action_specs,
                    obs_rms=obs_rms,
                    residual_scale=cfg.residual_scale,
                    hidden_dims=hidden_dims,
                    metadata={
                        "base_model_path": cfg.base_model_path,
                        "env_name": cfg.env_name,
                        "embodiment_tag": cfg.embodiment_tag.value,
                        "n_action_steps": cfg.n_action_steps,
                        "max_episode_steps": cfg.max_episode_steps,
                        "update": update_idx,
                        "global_step": global_step,
                    },
                )

            with metrics_path.open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics))
            wandb_log(wandb_run, metrics)

        save_residual_checkpoint(
            output_dir / "checkpoints" / "residual_final.pt",
            model=model,
            obs_keys=env.obs_keys,
            action_specs=env.action_specs,
            obs_rms=obs_rms,
            residual_scale=cfg.residual_scale,
            hidden_dims=hidden_dims,
            metadata={
                "base_model_path": cfg.base_model_path,
                "env_name": cfg.env_name,
                "embodiment_tag": cfg.embodiment_tag.value,
                "n_action_steps": cfg.n_action_steps,
                "max_episode_steps": cfg.max_episode_steps,
                "global_step": global_step,
            },
        )
        print("RL residual fine-tuning completed.")
        print(f"  Latest checkpoint: {output_dir / 'checkpoints' / 'residual_latest.pt'}")
        print(f"  Final checkpoint: {output_dir / 'checkpoints' / 'residual_final.pt'}")
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    cfg = tyro.cli(RLFineTuneConfig, description=__doc__)
    main(cfg)
