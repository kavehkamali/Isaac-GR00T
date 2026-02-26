from dataclasses import dataclass, field

from gr00t.data.embodiment_tags import EmbodimentTag


@dataclass
class RLFineTuneConfig:
    """Configuration for PPO-based RL residual fine-tuning on top of GR00T."""

    base_model_path: str
    """Path to the pretrained GR00T checkpoint (HF repo id or local dir)."""

    env_name: str
    """Gym env id to train on (e.g. robocasa_panda_omron/CountertopMugPickup_PandaOmron_Env)."""

    embodiment_tag: EmbodimentTag
    """Embodiment tag for the GR00T model (e.g. ROBOCASA_PANDA_OMRON)."""

    output_dir: str = "./outputs/rl_finetune"
    """Directory to save PPO residual checkpoints and training metrics."""

    device: str = "cuda"
    """Torch device for GR00T inference and PPO network."""

    seed: int = 123
    """Random seed for env resets and PPO training."""

    n_action_steps: int = 8
    """Action chunk horizon; must match the eval setting you plan to use."""

    max_episode_steps: int = 600
    """Maximum wrapped env steps per episode during RL training."""

    residual_scale: float = 0.25
    """Scale of PPO residual relative to half-range of action bounds."""

    total_timesteps: int = 50_000
    """Total PPO interaction steps (outer wrapped env steps)."""

    rollout_steps: int = 512
    """On-policy rollout length per PPO update."""

    mini_batch_size: int = 128
    """Minibatch size for PPO gradient updates."""

    update_epochs: int = 10
    """Number of PPO epochs per rollout."""

    learning_rate: float = 3e-4
    """PPO optimizer learning rate."""

    gamma: float = 0.99
    """Discount factor."""

    gae_lambda: float = 0.95
    """GAE lambda."""

    clip_coef: float = 0.2
    """PPO clipping epsilon."""

    value_coef: float = 0.5
    """Value loss coefficient."""

    entropy_coef: float = 0.0
    """Entropy bonus coefficient."""

    max_grad_norm: float = 0.5
    """Gradient clipping norm."""

    hidden_dim: int = 512
    """MLP hidden size for the residual actor-critic."""

    hidden_layers: int = 2
    """Number of hidden layers for the residual actor-critic."""

    eval_episodes: int = 5
    """Deterministic PPO evaluation episodes per checkpoint save; set 0 to disable."""

    eval_every_updates: int = 5
    """Run internal eval every N PPO updates."""

    save_every_updates: int = 5
    """Save 'latest' checkpoint every N PPO updates."""

    success_bonus: float = 10.0
    height_coef: float = 25.0
    hold_progress_coef: float = 2.0
    action_l2_coef: float = 0.02

    use_wandb: bool = False
    """Enable Weights & Biases logging for RL + VLA residual metrics."""

    wandb_project: str = "gr00t-rl-residual"
    """wandb project name."""

    wandb_entity: str | None = None
    """Optional wandb entity (team/user). If None, use local wandb default."""

    wandb_group: str | None = None
    """Optional wandb group to cluster runs."""

    wandb_run_name: str | None = None
    """Optional custom wandb run name."""

    wandb_tags: list[str] = field(default_factory=lambda: ["rl", "ppo", "gr00t", "robocasa"])
    """wandb tags for filtering runs."""
