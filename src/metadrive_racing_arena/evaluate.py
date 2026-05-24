from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import ProjectConfig
from .policy import PPOPolicy
from .race import _run_single_policy
from .utils import ensure_dir, utc_timestamp, write_json


def evaluate_checkpoint(config: ProjectConfig, checkpoint: str | Path) -> dict:
    policy = PPOPolicy.load(checkpoint, device=config.ppo.device)
    env_cfg = replace(config.env, use_render=config.evaluation.render)
    results = []
    for episode in range(config.evaluation.episodes):
        stats = _run_single_policy(
            policy,
            env_cfg,
            seed=config.env.seed + episode,
            deterministic=config.evaluation.deterministic,
        )
        results.append(stats.to_dict())

    reward_avg = sum(row["reward"] for row in results) / len(results)
    payload = {
        "policy_id": policy.policy_id,
        "checkpoint": str(checkpoint),
        "episodes": len(results),
        "avg_reward": reward_avg,
        "finish_rate": sum(row["arrived"] for row in results) / len(results),
        "crash_rate": sum(row["crashed"] for row in results) / len(results),
        "episodes_detail": results,
    }
    output_path = ensure_dir(config.evaluation.results_dir) / f"eval_{policy.policy_id}_{utc_timestamp()}.json"
    write_json(output_path, payload)
    payload["output_path"] = str(output_path)
    return payload

