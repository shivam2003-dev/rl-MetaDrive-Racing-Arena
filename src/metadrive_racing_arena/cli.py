from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .evaluate import evaluate_checkpoint
from .leaderboard import Leaderboard, format_leaderboard
from .policy import PPOPolicy
from .race import run_head_to_head
from .tournament import TournamentManager
from .ppo import train_from_config
from .utils import utc_timestamp, write_json


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="metadrive-arena")
    parser.add_argument("--config", default="configs/default.yaml")
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", parents=[config_parent])
    train.add_argument("--policy-id")
    train.add_argument("--steps", type=int)
    train.add_argument("--track")
    train.add_argument("--seed", type=int)

    evaluate = sub.add_parser("evaluate", parents=[config_parent])
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--episodes", type=int)
    evaluate.add_argument("--render", action="store_true")

    race = sub.add_parser("race", parents=[config_parent])
    race.add_argument("--policy-a", required=True)
    race.add_argument("--policy-b", required=True)
    race.add_argument("--track")
    race.add_argument("--seed", type=int)
    race.add_argument("--output", default=None)

    tournament = sub.add_parser("tournament", parents=[config_parent])
    tournament.add_argument("--policies", nargs="*")
    tournament.add_argument("--races-per-pair", type=int)
    tournament.add_argument("--max-pairs", type=int)

    board = sub.add_parser("leaderboard", parents=[config_parent])
    board.add_argument("--export-csv")
    board.add_argument("--export-json")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "train":
        if args.policy_id:
            cfg.training.policy_id = args.policy_id
        if args.steps:
            cfg.ppo.total_steps = args.steps
        if args.track:
            cfg.env.map = args.track
        if args.seed is not None:
            cfg.env.seed = args.seed
        checkpoint = train_from_config(cfg)
        print(checkpoint)

    elif args.command == "evaluate":
        if args.episodes:
            cfg.evaluation.episodes = args.episodes
        if args.render:
            cfg.evaluation.render = True
        result = evaluate_checkpoint(cfg, args.checkpoint)
        print(result["output_path"])

    elif args.command == "race":
        if args.track:
            cfg.env.map = args.track
        if args.seed is not None:
            cfg.env.seed = args.seed
        env_cfg = replace(cfg.env, scenario="multi_agent")
        policy_a = PPOPolicy.load(args.policy_a, device=cfg.ppo.device)
        policy_b = PPOPolicy.load(args.policy_b, device=cfg.ppo.device)
        result = run_head_to_head(
            policy_a,
            policy_b,
            env_cfg,
            race_id=f"manual_{utc_timestamp()}",
            seed=cfg.env.seed,
            draw_margin=cfg.tournament.draw_margin,
        )
        output = args.output or f"data/results/{result.race_id}.json"
        write_json(output, result.to_dict())
        print(output)

    elif args.command == "tournament":
        if args.races_per_pair:
            cfg.tournament.races_per_pair = args.races_per_pair
        if args.max_pairs:
            cfg.tournament.max_pairs = args.max_pairs
        manager = TournamentManager(cfg)
        results = manager.run(args.policies)
        print(f"completed {len(results)} races")

    elif args.command == "leaderboard":
        leaderboard = Leaderboard(cfg.tournament.leaderboard_path)
        if args.export_csv:
            leaderboard.export_csv(args.export_csv)
        if args.export_json:
            leaderboard.export_json(args.export_json)
        print(format_leaderboard(leaderboard.sorted_rows()))


if __name__ == "__main__":
    main()
