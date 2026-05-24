from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    scenario: str = "single_agent"
    map: str = "SSSS"
    traffic_density: float = 0.1
    traffic_mode: str = "respawn"
    num_agents: int = 2
    horizon: int = 1000
    seed: int = 7
    use_render: bool = False
    metadrive: dict[str, Any] = field(default_factory=dict)


@dataclass
class PPOConfig:
    total_steps: int = 100_000
    rollout_steps: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, int] = (256, 256)
    deterministic_eval: bool = True
    device: str = "auto"


@dataclass
class TrainingConfig:
    policy_id: str = "ppo_driver"
    checkpoint_dir: str = "data/checkpoints"
    log_dir: str = "data/logs"
    save_interval_steps: int = 25_000
    eval_interval_steps: int = 25_000
    eval_episodes: int = 5


@dataclass
class EvaluationConfig:
    episodes: int = 10
    deterministic: bool = True
    render: bool = False
    results_dir: str = "data/results"


@dataclass
class TournamentConfig:
    policies_dir: str = "data/checkpoints"
    leaderboard_path: str = "data/leaderboards/leaderboard.json"
    match_results_path: str = "data/results/matches.jsonl"
    rating_history_path: str = "data/leaderboards/rating_history.jsonl"
    races_per_pair: int = 3
    max_pairs: int | None = None
    k_factor: float = 32.0
    draw_margin: float = 0.05
    seed: int = 7
    tracks: list[str] = field(default_factory=lambda: ["SSSS", "X", "O"])


@dataclass
class ProjectConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _coerce_dataclass(cls: type, data: dict[str, Any]):
    fields = {field_name for field_name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in data.items() if key in fields})


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ProjectConfig:
    base = ProjectConfig().to_dict()
    if path:
        with Path(path).open("r", encoding="utf-8") as fh:
            file_cfg = yaml.safe_load(fh) or {}
        base = _deep_update(base, file_cfg)
    if overrides:
        base = _deep_update(base, overrides)

    if isinstance(base["ppo"].get("hidden_sizes"), list):
        base["ppo"]["hidden_sizes"] = tuple(base["ppo"]["hidden_sizes"])

    return ProjectConfig(
        env=_coerce_dataclass(EnvConfig, base["env"]),
        ppo=_coerce_dataclass(PPOConfig, base["ppo"]),
        training=_coerce_dataclass(TrainingConfig, base["training"]),
        evaluation=_coerce_dataclass(EvaluationConfig, base["evaluation"]),
        tournament=_coerce_dataclass(TournamentConfig, base["tournament"]),
    )

