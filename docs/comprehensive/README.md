# MetaDrive Racing Arena Comprehensive Guide

This folder documents the full reinforcement learning platform around MetaDrive Racing Arena. It is intended as a systems and research companion to the implementation in `src/metadrive_racing_arena/`.

## Contents

- [01 Simulator and Environment](./01-simulator-and-environment.md)
- [02 PPO and Learning Theory](./02-ppo-and-learning-theory.md)
- [03 Distributed Infrastructure](./03-distributed-infrastructure.md)
- [04 Evaluation Competition and Leaderboards](./04-evaluation-competition-and-leaderboards.md)
- [05 Advanced Methods and Failure Modes](./05-advanced-methods-and-failure-modes.md)

## Scope

The material covers:

- MetaDrive simulator design and racing environment assumptions
- PPO training pipeline and mathematical derivations
- multi-agent reinforcement learning and self-play
- reward shaping, autonomous driving, and trajectory optimization
- policy evaluation and ELO systems
- Ray RLlib rollout workers, replay buffers, distributed training, and GPU scaling
- evaluation arenas, matchmaking, policy registry, and leaderboard backend
- curriculum learning, adversarial racing, imitation learning, and transformer policies
- infrastructure setup, Kubernetes cluster design, hyperparameter tuning, and operational failure modes

## Reader Map

- Start with `01` for simulator assumptions and data flow.
- Read `02` for the optimization objective, PPO math, and reward design.
- Use `03` when planning production-scale rollout and training infrastructure.
- Use `04` for tournaments, ELO, evaluation metrics, and ranking services.
- Use `05` for research extensions, edge cases, and reliability analysis.

