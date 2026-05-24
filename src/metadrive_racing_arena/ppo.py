from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
from tqdm import tqdm

from .config import ProjectConfig
from .envs import (
    agent_obs,
    close_env,
    flatten_observation,
    infer_space_dimensions,
    is_done,
    make_env,
    reset_env,
    scale_action,
    step_env,
)
from .utils import append_jsonl, ensure_dir, set_global_seeds, utc_timestamp, write_json


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, int] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(last_dim, hidden), nn.Tanh()])
            last_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(last_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        return self.actor_mean(features), self.log_std.expand_as(self.actor_mean(features)), self.critic(features).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean, log_std, _ = self(obs)
        return Normal(mean, log_std.exp())

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, log_std, value = self(obs)
        dist = Normal(mean, log_std.exp())
        action = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        mean, log_std, value = self(obs)
        dist = Normal(mean, log_std.exp())
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.size = size
        self.device = device
        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, act_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)


class PPOTrainer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.device = resolve_device(config.ppo.device)
        set_global_seeds(config.env.seed)

        self.env = make_env(config.env, seed=config.env.seed)
        obs, _ = reset_env(self.env, seed=config.env.seed)
        self.obs_dim, self.act_dim, self.action_low, self.action_high = infer_space_dimensions(self.env, obs)

        self.model = ActorCritic(self.obs_dim, self.act_dim, config.ppo.hidden_sizes).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.ppo.learning_rate)
        self.global_step = 0
        self.run_id = f"{config.training.policy_id}_{utc_timestamp()}"
        self.checkpoint_dir = ensure_dir(Path(config.training.checkpoint_dir) / self.run_id)
        self.log_path = ensure_dir(config.training.log_dir) / f"{self.run_id}.jsonl"

    def train(self) -> Path:
        cfg = self.config.ppo
        buffer = RolloutBuffer(cfg.rollout_steps, self.obs_dim, self.act_dim, self.device)
        obs, _ = reset_env(self.env, seed=self.config.env.seed)
        episode_return = 0.0
        episode_length = 0
        episode_count = 0
        last_checkpoint = self.checkpoint_dir / "latest.pt"

        progress = tqdm(total=cfg.total_steps, initial=self.global_step, desc="PPO training")
        while self.global_step < cfg.total_steps:
            for step in range(cfg.rollout_steps):
                obs_vec = torch.as_tensor(
                    flatten_observation(agent_obs(obs)), dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    raw_action, log_prob, _, value = self.model.act(obs_vec.unsqueeze(0))
                env_action = scale_action(raw_action.squeeze(0).cpu().numpy(), self.action_low, self.action_high)

                next_obs, reward, done, info = step_env(self.env, env_action)
                reward_scalar = float(np.asarray(reward).mean())
                done_scalar = is_done(done)

                buffer.obs[step] = obs_vec
                buffer.actions[step] = raw_action.squeeze(0)
                buffer.log_probs[step] = log_prob.squeeze(0)
                buffer.rewards[step] = reward_scalar
                buffer.dones[step] = float(done_scalar)
                buffer.values[step] = value.squeeze(0)

                self.global_step += 1
                episode_return += reward_scalar
                episode_length += 1
                progress.update(1)

                if done_scalar:
                    metrics = self._episode_metrics(
                        episode_count=episode_count,
                        episode_return=episode_return,
                        episode_length=episode_length,
                        info=info,
                    )
                    append_jsonl(self.log_path, metrics)
                    obs, _ = reset_env(self.env)
                    episode_return = 0.0
                    episode_length = 0
                    episode_count += 1
                else:
                    obs = next_obs

                if self.global_step >= cfg.total_steps:
                    break

            self._compute_gae(buffer, obs)
            self._update(buffer)

            if (
                self.global_step % self.config.training.save_interval_steps < cfg.rollout_steps
                or self.global_step >= cfg.total_steps
            ):
                last_checkpoint = self.save_checkpoint("latest.pt")
                self.save_checkpoint(f"step_{self.global_step}.pt")

        progress.close()
        close_env(self.env)
        return last_checkpoint

    def _compute_gae(self, buffer: RolloutBuffer, last_obs: Any) -> None:
        with torch.no_grad():
            last_obs_vec = torch.as_tensor(
                flatten_observation(agent_obs(last_obs)), dtype=torch.float32, device=self.device
            )
            _, _, last_value = self.model(last_obs_vec.unsqueeze(0))
            last_gae = 0.0
            for t in reversed(range(buffer.size)):
                if t == buffer.size - 1:
                    next_non_terminal = 1.0 - buffer.dones[t]
                    next_value = last_value.squeeze(0)
                else:
                    next_non_terminal = 1.0 - buffer.dones[t + 1]
                    next_value = buffer.values[t + 1]
                delta = (
                    buffer.rewards[t]
                    + self.config.ppo.gamma * next_value * next_non_terminal
                    - buffer.values[t]
                )
                last_gae = delta + self.config.ppo.gamma * self.config.ppo.gae_lambda * next_non_terminal * last_gae
                buffer.advantages[t] = last_gae
            buffer.returns = buffer.advantages + buffer.values

    def _update(self, buffer: RolloutBuffer) -> None:
        cfg = self.config.ppo
        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        indices = torch.arange(buffer.size, device=self.device)

        for _ in range(cfg.update_epochs):
            perm = indices[torch.randperm(buffer.size, device=self.device)]
            for start in range(0, buffer.size, cfg.minibatch_size):
                batch_idx = perm[start : start + cfg.minibatch_size]
                new_log_prob, entropy, value = self.model.evaluate_actions(
                    buffer.obs[batch_idx], buffer.actions[batch_idx]
                )
                ratio = (new_log_prob - buffer.log_probs[batch_idx]).exp()
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                policy_loss = -torch.min(ratio * advantages[batch_idx], clipped * advantages[batch_idx]).mean()
                value_loss = 0.5 * (buffer.returns[batch_idx] - value).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

    def _episode_metrics(self, episode_count: int, episode_return: float, episode_length: int, info: Any):
        info = info or {}
        if isinstance(info, dict) and "__common__" in info:
            info = info["__common__"]
        return {
            "run_id": self.run_id,
            "policy_id": self.config.training.policy_id,
            "global_step": self.global_step,
            "episode": episode_count,
            "reward": episode_return,
            "episode_length": episode_length,
            "arrive_dest": bool(_info_get(info, "arrive_dest", False)),
            "crash": bool(_info_get(info, "crash", False) or _info_get(info, "crash_vehicle", False)),
        }

    def save_checkpoint(self, filename: str) -> Path:
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "config": self.config.to_dict(),
                "policy_id": self.config.training.policy_id,
                "global_step": self.global_step,
            },
            path,
        )
        write_json(
            self.checkpoint_dir / "metadata.json",
            {
                "run_id": self.run_id,
                "policy_id": self.config.training.policy_id,
                "global_step": self.global_step,
                "checkpoint": str(path),
                "config": self.config.to_dict(),
            },
        )
        return path


def _info_get(info: Any, key: str, default: Any = None) -> Any:
    if isinstance(info, dict):
        return info.get(key, default)
    return getattr(info, key, default)


def train_from_config(config: ProjectConfig) -> Path:
    return PPOTrainer(config).train()

