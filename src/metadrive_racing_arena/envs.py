from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .config import EnvConfig


class MetaDriveImportError(RuntimeError):
    pass


def _import_env_class(scenario: str):
    candidates: dict[str, list[tuple[str, str]]] = {
        "single_agent": [
            ("metadrive", "MetaDriveEnv"),
            ("metadrive.envs.metadrive_env", "MetaDriveEnv"),
        ],
        "multi_agent": [
            ("metadrive", "MultiAgentMetaDrive"),
            ("metadrive.envs.marl_envs.multi_agent_metadrive", "MultiAgentMetaDrive"),
        ],
        "intersection": [
            ("metadrive", "MultiAgentIntersectionEnv"),
            ("metadrive.envs.marl_envs.multi_agent_intersection", "MultiAgentIntersectionEnv"),
        ],
        "roundabout": [
            ("metadrive", "MultiAgentRoundaboutEnv"),
            ("metadrive.envs.marl_envs.multi_agent_roundabout", "MultiAgentRoundaboutEnv"),
        ],
        "tollgate": [
            ("metadrive", "MultiAgentTollgateEnv"),
            ("metadrive.envs.marl_envs.multi_agent_tollgate", "MultiAgentTollgateEnv"),
        ],
        "bottleneck": [
            ("metadrive", "MultiAgentBottleneckEnv"),
            ("metadrive.envs.marl_envs.multi_agent_bottleneck", "MultiAgentBottleneckEnv"),
        ],
        "parking": [
            ("metadrive", "MultiAgentParkingLotEnv"),
            ("metadrive.envs.marl_envs.multi_agent_parking_lot", "MultiAgentParkingLotEnv"),
        ],
    }
    errors: list[str] = []
    for module_name, class_name in candidates.get(scenario, candidates["single_agent"]):
        try:
            module = __import__(module_name, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as exc:  # MetaDrive may fail import if optional render deps are missing.
            errors.append(f"{module_name}.{class_name}: {exc}")
    raise MetaDriveImportError(
        "Could not import a MetaDrive environment for scenario "
        f"{scenario!r}. Install MetaDrive and check its version. Tried: {errors}"
    )


def build_metadrive_config(config: EnvConfig, seed: int | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "map": config.map,
        "traffic_density": config.traffic_density,
        "traffic_mode": config.traffic_mode,
        "horizon": config.horizon,
        "num_agents": config.num_agents,
        "use_render": config.use_render,
        "start_seed": config.seed if seed is None else seed,
        "log_level": 50,
    }
    cfg.update(config.metadrive)
    if config.scenario == "single_agent":
        cfg.pop("num_agents", None)
    return cfg


def make_env(config: EnvConfig, seed: int | None = None):
    env_cls = _import_env_class(config.scenario)
    return env_cls(build_metadrive_config(config, seed=seed))


def reset_env(env, seed: int | None = None):
    try:
        result = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = _merge_done(terminated, truncated)
        return obs, reward, done, info
    obs, reward, done, info = result
    return obs, reward, done, info


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _merge_done(terminated, truncated):
    if isinstance(terminated, dict):
        keys = set(terminated) | set(truncated)
        merged = {key: bool(terminated.get(key, False) or truncated.get(key, False)) for key in keys}
        merged["__all__"] = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        return merged
    return bool(terminated or truncated)


def is_done(done: Any) -> bool:
    if isinstance(done, dict):
        return bool(done.get("__all__", all(done.values()) if done else False))
    return bool(done)


def alive_agent_ids(obs: Any) -> list[str]:
    if isinstance(obs, dict):
        return [str(key) for key in obs.keys()]
    return ["agent0"]


def agent_obs(obs: Any, agent_id: str | None = None):
    if isinstance(obs, dict):
        key = agent_id if agent_id in obs else next(iter(obs.keys()))
        return obs[key]
    return obs


def flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [flatten_observation(obs[key]) for key in sorted(obs)]
        return np.concatenate(parts).astype(np.float32, copy=False)
    arr = np.asarray(obs, dtype=np.float32)
    if arr.dtype == object:
        raise TypeError(f"Unsupported object observation type: {type(obs)!r}")
    return arr.reshape(-1)


def infer_space_dimensions(env, sample_obs: Any | None = None) -> tuple[int, int, np.ndarray, np.ndarray]:
    if sample_obs is None:
        sample_obs, _ = reset_env(env)
    obs_dim = int(flatten_observation(agent_obs(sample_obs)).shape[0])

    action_space = getattr(env, "action_space", None)
    if isinstance(action_space, dict):
        action_space = next(iter(action_space.values()))
    low = np.asarray(getattr(action_space, "low", [-1.0, -1.0]), dtype=np.float32).reshape(-1)
    high = np.asarray(getattr(action_space, "high", [1.0, 1.0]), dtype=np.float32).reshape(-1)
    act_dim = int(low.shape[0])
    return obs_dim, act_dim, low, high


def scale_action(raw_action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(low.shape)
    squashed = np.tanh(raw_action)
    action = low + (squashed + 1.0) * 0.5 * (high - low)
    return np.clip(action, low, high).astype(np.float32)


def env_config_to_dict(config: EnvConfig) -> dict[str, Any]:
    return asdict(config)

