from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import EnvConfig
from .envs import (
    agent_obs,
    alive_agent_ids,
    close_env,
    infer_space_dimensions,
    is_done,
    make_env,
    reset_env,
    step_env,
)
from .policy import PPOPolicy, RacingPolicy
from .utils import as_float, set_global_seeds


@dataclass
class AgentRaceStats:
    policy_id: str
    reward: float = 0.0
    steps: int = 0
    arrived: bool = False
    crashed: bool = False
    distance: float = 0.0
    lap_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "reward": round(self.reward, 6),
            "steps": self.steps,
            "arrived": self.arrived,
            "crashed": self.crashed,
            "distance": round(self.distance, 6),
            "lap_time": round(self.lap_time, 6),
        }


@dataclass
class RaceResult:
    race_id: str
    policy_a: str
    policy_b: str
    outcome: str
    track: str
    seed: int
    stats: dict[str, AgentRaceStats]

    def to_dict(self) -> dict:
        return {
            "race_id": self.race_id,
            "policy_a": self.policy_a,
            "policy_b": self.policy_b,
            "outcome": self.outcome,
            "track": self.track,
            "seed": self.seed,
            "stats": {agent_id: stats.to_dict() for agent_id, stats in self.stats.items()},
        }


def load_policy(path: str, device: str = "auto") -> PPOPolicy:
    return PPOPolicy.load(path, device=device)


def run_head_to_head(
    policy_a: RacingPolicy,
    policy_b: RacingPolicy,
    env_config: EnvConfig,
    race_id: str,
    seed: int,
    deterministic: bool = True,
    draw_margin: float = 0.05,
) -> RaceResult:
    set_global_seeds(seed)
    env = make_env(env_config, seed=seed)
    obs, _ = reset_env(env, seed=seed)
    agent_ids = alive_agent_ids(obs)

    if len(agent_ids) < 2:
        result = _run_sequential_comparison(
            policy_a, policy_b, env_config, race_id, seed, deterministic, draw_margin
        )
        close_env(env)
        return result

    assignments = {agent_ids[0]: policy_a, agent_ids[1]: policy_b}
    stats = {
        agent_ids[0]: AgentRaceStats(policy_id=policy_a.policy_id),
        agent_ids[1]: AgentRaceStats(policy_id=policy_b.policy_id),
    }

    while True:
        actions = {}
        for agent_id, policy in assignments.items():
            if isinstance(obs, dict) and agent_id not in obs:
                continue
            actions[agent_id] = policy.act(agent_obs(obs, agent_id), deterministic=deterministic)

        obs, rewards, done, infos = step_env(env, actions)
        _update_stats(stats, rewards, infos)
        if is_done(done):
            break

    close_env(env)
    outcome = determine_outcome(stats[agent_ids[0]], stats[agent_ids[1]], draw_margin=draw_margin)
    return RaceResult(
        race_id=race_id,
        policy_a=policy_a.policy_id,
        policy_b=policy_b.policy_id,
        outcome=outcome,
        track=env_config.map,
        seed=seed,
        stats=stats,
    )


def _run_sequential_comparison(
    policy_a: RacingPolicy,
    policy_b: RacingPolicy,
    env_config: EnvConfig,
    race_id: str,
    seed: int,
    deterministic: bool,
    draw_margin: float,
) -> RaceResult:
    stats = {
        "agent_a": _run_single_policy(policy_a, env_config, seed, deterministic),
        "agent_b": _run_single_policy(policy_b, env_config, seed, deterministic),
    }
    outcome = determine_outcome(stats["agent_a"], stats["agent_b"], draw_margin=draw_margin)
    return RaceResult(
        race_id=race_id,
        policy_a=policy_a.policy_id,
        policy_b=policy_b.policy_id,
        outcome=outcome,
        track=env_config.map,
        seed=seed,
        stats=stats,
    )


def _run_single_policy(
    policy: RacingPolicy, env_config: EnvConfig, seed: int, deterministic: bool
) -> AgentRaceStats:
    env = make_env(env_config, seed=seed)
    obs, _ = reset_env(env, seed=seed)
    infer_space_dimensions(env, obs)
    stats = AgentRaceStats(policy_id=policy.policy_id)
    while True:
        action = policy.act(agent_obs(obs), deterministic=deterministic)
        obs, reward, done, info = step_env(env, action)
        _update_single_stats(stats, reward, info)
        if is_done(done):
            break
    close_env(env)
    return stats


def _update_stats(stats: dict[str, AgentRaceStats], rewards: Any, infos: Any) -> None:
    for agent_id, stat in stats.items():
        reward = rewards.get(agent_id, 0.0) if isinstance(rewards, dict) else rewards
        info = infos.get(agent_id, {}) if isinstance(infos, dict) else infos
        _update_single_stats(stat, reward, info)


def _update_single_stats(stat: AgentRaceStats, reward: Any, info: Any) -> None:
    stat.reward += float(np.asarray(reward).mean())
    stat.steps += 1
    if not isinstance(info, dict):
        info = {}
    stat.arrived = stat.arrived or bool(info.get("arrive_dest", False) or info.get("success", False))
    stat.crashed = stat.crashed or bool(
        info.get("crash", False) or info.get("crash_vehicle", False) or info.get("out_of_road", False)
    )
    stat.distance = max(
        stat.distance,
        as_float(info.get("route_completion"), 0.0),
        as_float(info.get("progress"), 0.0),
        as_float(info.get("distance"), 0.0),
    )
    if stat.arrived and stat.lap_time == 0.0:
        stat.lap_time = as_float(info.get("episode_length"), float(stat.steps))


def determine_outcome(a: AgentRaceStats, b: AgentRaceStats, draw_margin: float = 0.05) -> str:
    score_a = _performance_score(a)
    score_b = _performance_score(b)
    if abs(score_a - score_b) <= draw_margin:
        return "draw"
    return "a_win" if score_a > score_b else "b_win"


def _performance_score(stats: AgentRaceStats) -> float:
    finish_bonus = 1000.0 if stats.arrived else 0.0
    crash_penalty = 250.0 if stats.crashed else 0.0
    time_penalty = stats.steps * 0.01
    return finish_bonus + stats.distance * 100.0 + stats.reward - crash_penalty - time_penalty
