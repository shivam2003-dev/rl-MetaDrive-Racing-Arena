from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .envs import flatten_observation, scale_action
from .ppo import ActorCritic, resolve_device


class RacingPolicy(Protocol):
    policy_id: str

    def act(self, obs, deterministic: bool = True) -> np.ndarray:
        ...


@dataclass
class PPOPolicy:
    policy_id: str
    model: ActorCritic
    action_low: np.ndarray
    action_high: np.ndarray
    device: torch.device

    @classmethod
    def load(cls, checkpoint_path: str | Path, device: str = "auto") -> "PPOPolicy":
        torch_device = resolve_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=torch_device)
        model = ActorCritic(
            int(checkpoint["obs_dim"]),
            int(checkpoint["act_dim"]),
            tuple(checkpoint.get("config", {}).get("ppo", {}).get("hidden_sizes", (256, 256))),
        ).to(torch_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(
            policy_id=str(checkpoint.get("policy_id") or Path(checkpoint_path).parent.name),
            model=model,
            action_low=np.asarray(checkpoint["action_low"], dtype=np.float32),
            action_high=np.asarray(checkpoint["action_high"], dtype=np.float32),
            device=torch_device,
        )

    def act(self, obs, deterministic: bool = True) -> np.ndarray:
        obs_vec = torch.as_tensor(flatten_observation(obs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _, _ = self.model.act(obs_vec.unsqueeze(0), deterministic=deterministic)
        return scale_action(action.squeeze(0).cpu().numpy(), self.action_low, self.action_high)


@dataclass
class RandomPolicy:
    policy_id: str
    action_low: np.ndarray
    action_high: np.ndarray

    def act(self, obs, deterministic: bool = True) -> np.ndarray:
        del obs, deterministic
        return np.random.uniform(self.action_low, self.action_high).astype(np.float32)

