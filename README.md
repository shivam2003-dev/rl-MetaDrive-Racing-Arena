# MetaDrive Racing Arena

MetaDrive Racing Arena is a Python reinforcement learning project for training PPO driving agents, racing saved policies head-to-head, running automated tournaments, and ranking policies with an ELO leaderboard.

## Features

- PyTorch PPO trainer for MetaDrive continuous-control agents.
- Single-agent PPO training and evaluation.
- Multi-agent race simulation when a MetaDrive multi-agent environment is available.
- Sequential same-seed comparison fallback for single-agent environments.
- Automated round-robin tournaments with configurable tracks and repeated matches.
- ELO leaderboard with wins, losses, draws, reward, finish rate, crash rate, and race history.
- JSON/JSONL/CSV outputs for checkpoints, logs, match results, rating history, and leaderboard exports.
- YAML configs for reproducible seeds, tracks, traffic, PPO hyperparameters, and tournament rules.

## Assumptions

The MetaDrive API has changed across releases. This project isolates MetaDrive-specific code in `src/metadrive_racing_arena/envs.py` and dynamically imports common classes:

- `MetaDriveEnv` for single-agent training.
- `MultiAgentMetaDrive` for two-policy races.
- Optional multi-agent scenarios such as intersection, roundabout, tollgate, bottleneck, and parking.

The PPO implementation is intentionally local PyTorch code instead of a large RL framework, so the project is easy to inspect and extend. It expects continuous Box actions, which matches standard MetaDrive steering/throttle control.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If your platform needs extra rendering dependencies for MetaDrive, install those according to the MetaDrive documentation.

## Train a Policy

```bash
python -m metadrive_racing_arena.cli --config configs/default.yaml train \
  --policy-id driver_a \
  --steps 100000 \
  --track SSSS \
  --seed 7
```

Checkpoints are written under:

```text
data/checkpoints/<policy_id>_<timestamp>/
```

Training logs are JSONL files in `data/logs/` with reward, episode length, crash, and finish signals.

## Evaluate a Policy

```bash
python -m metadrive_racing_arena.cli --config configs/default.yaml evaluate \
  --checkpoint data/checkpoints/driver_a_YYYYMMDDTHHMMSSZ/latest.pt \
  --episodes 10
```

To render evaluation:

```bash
python -m metadrive_racing_arena.cli --config configs/default.yaml evaluate \
  --checkpoint data/checkpoints/driver_a_YYYYMMDDTHHMMSSZ/latest.pt \
  --episodes 3 \
  --render
```

## Run a Head-to-Head Race

```bash
python -m metadrive_racing_arena.cli --config configs/tournament.yaml race \
  --policy-a data/checkpoints/driver_a_YYYYMMDDTHHMMSSZ/latest.pt \
  --policy-b data/checkpoints/driver_b_YYYYMMDDTHHMMSSZ/latest.pt \
  --track X \
  --seed 21
```

Race results are saved as JSON under `data/results/`.

## Run a Tournament

Use all discovered checkpoints:

```bash
python -m metadrive_racing_arena.cli --config configs/tournament.yaml tournament
```

Use explicit checkpoints:

```bash
python -m metadrive_racing_arena.cli --config configs/tournament.yaml tournament \
  --policies data/checkpoints/driver_a/latest.pt data/checkpoints/driver_b/latest.pt \
  --races-per-pair 5
```

Tournament outputs:

- `data/results/matches.jsonl`
- `data/leaderboards/rating_history.jsonl`
- `data/leaderboards/leaderboard.json`

## View and Export the Leaderboard

```bash
python -m metadrive_racing_arena.cli --config configs/tournament.yaml leaderboard
```

```bash
python -m metadrive_racing_arena.cli --config configs/tournament.yaml leaderboard \
  --export-csv data/leaderboards/leaderboard.csv \
  --export-json data/leaderboards/leaderboard_export.json
```

The leaderboard is sorted by ELO and includes current ELO, wins, losses, draws, win rate, finish rate, crash rate, average reward, and average lap time.

## Comprehensive Documentation

Detailed system and research documentation lives in [docs/comprehensive/README.md](/Users/shivamkumar/Desktop/temp/metadrive/docs/comprehensive/README.md). It covers simulator design, PPO derivations, multi-agent RL, self-play, reward shaping, trajectory optimization, distributed RL infrastructure, Ray RLlib rollout workers, Kubernetes cluster design, GPU scaling, evaluation arenas, matchmaking, policy registry design, leaderboard backend architecture, curriculum learning, adversarial racing, imitation learning, transformer policies, hyperparameter tuning, evaluation metrics, and failure modes.

## Project Layout

```text
configs/
  default.yaml              PPO training and evaluation defaults
  tournament.yaml           Race and tournament defaults
scripts/
  train.py                  CLI wrapper
  evaluate.py               CLI wrapper
  race.py                   CLI wrapper
  tournament.py             CLI wrapper
  leaderboard.py            CLI wrapper
src/metadrive_racing_arena/
  config.py                 YAML config dataclasses
  envs.py                   MetaDrive environment setup and API compatibility
  ppo.py                    PyTorch PPO trainer
  policy.py                 Saved PPO policy loader
  evaluate.py               Evaluation pipeline
  race.py                   Head-to-head race simulation
  tournament.py             Tournament manager
  elo.py                    ELO rating system
  leaderboard.py            Leaderboard export and formatting
  utils.py                  Seeds, JSON, and filesystem helpers
tests/
  test_elo.py
  test_leaderboard.py
```

## Extending

To add a new MetaDrive scenario, add an import candidate in `envs.py` and set `env.scenario` in YAML. To change competition rules, adjust `determine_outcome()` in `race.py`. To add metrics, extend `AgentRaceStats` and the leaderboard aggregation.
