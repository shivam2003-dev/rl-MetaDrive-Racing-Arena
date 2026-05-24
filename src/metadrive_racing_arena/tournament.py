from __future__ import annotations

import itertools
import random
from dataclasses import replace
from pathlib import Path

from .config import ProjectConfig
from .elo import outcome_to_score, update_ratings
from .leaderboard import Leaderboard
from .policy import PPOPolicy
from .race import RaceResult, run_head_to_head
from .utils import append_jsonl, utc_timestamp


def discover_checkpoints(policies_dir: str | Path) -> list[Path]:
    root = Path(policies_dir)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.pt") if path.name == "latest.pt" or path.name.startswith("step_"))


class TournamentManager:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.tournament = config.tournament
        self.leaderboard = Leaderboard(self.tournament.leaderboard_path)
        random.seed(self.tournament.seed)

    def run(self, checkpoint_paths: list[str | Path] | None = None) -> list[RaceResult]:
        paths = [Path(path) for path in checkpoint_paths] if checkpoint_paths else discover_checkpoints(self.tournament.policies_dir)
        policies = [PPOPolicy.load(path, device=self.config.ppo.device) for path in paths]
        if len(policies) < 2:
            raise ValueError("Tournament requires at least two policy checkpoints.")

        pairs = list(itertools.combinations(policies, 2))
        random.shuffle(pairs)
        if self.tournament.max_pairs is not None:
            pairs = pairs[: self.tournament.max_pairs]

        results: list[RaceResult] = []
        match_index = 0
        for policy_a, policy_b in pairs:
            for repeat in range(self.tournament.races_per_pair):
                track = self.tournament.tracks[(match_index + repeat) % len(self.tournament.tracks)]
                env_cfg = replace(self.config.env, scenario="multi_agent", map=track)
                seed = self.tournament.seed + match_index * 1000 + repeat
                race_id = f"race_{utc_timestamp()}_{match_index}_{repeat}"
                result = run_head_to_head(
                    policy_a,
                    policy_b,
                    env_cfg,
                    race_id=race_id,
                    seed=seed,
                    draw_margin=self.tournament.draw_margin,
                )
                self._record_result(result)
                results.append(result)
            match_index += 1
        self.leaderboard.save()
        return results

    def _record_result(self, result: RaceResult) -> None:
        rec_a = self.leaderboard.ensure(result.policy_a)
        rec_b = self.leaderboard.ensure(result.policy_b)
        score_a = outcome_to_score(result.outcome)
        rec_a.rating, rec_b.rating = update_ratings(
            rec_a.rating, rec_b.rating, score_a, k_factor=self.tournament.k_factor
        )

        if result.outcome == "a_win":
            rec_a.wins += 1
            rec_b.losses += 1
        elif result.outcome == "b_win":
            rec_b.wins += 1
            rec_a.losses += 1
        else:
            rec_a.draws += 1
            rec_b.draws += 1

        for stat in result.stats.values():
            record = self.leaderboard.ensure(stat.policy_id)
            record.races += 1
            record.reward_sum += stat.reward
            record.lap_time_sum += stat.lap_time
            record.crash_count += int(stat.crashed)
            record.finish_count += int(stat.arrived)

        append_jsonl(self.tournament.match_results_path, result.to_dict())
        append_jsonl(
            self.tournament.rating_history_path,
            {
                "race_id": result.race_id,
                "policy_a": result.policy_a,
                "policy_b": result.policy_b,
                "outcome": result.outcome,
                "rating_a": rec_a.rating,
                "rating_b": rec_b.rating,
            },
        )
