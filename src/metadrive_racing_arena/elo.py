from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EloRecord:
    policy_id: str
    rating: float = 1000.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    races: int = 0
    reward_sum: float = 0.0
    lap_time_sum: float = 0.0
    crash_count: int = 0
    finish_count: int = 0

    def to_dict(self) -> dict:
        avg_reward = self.reward_sum / self.races if self.races else 0.0
        avg_lap_time = self.lap_time_sum / self.finish_count if self.finish_count else 0.0
        return {
            "policy_id": self.policy_id,
            "elo": round(self.rating, 2),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "races": self.races,
            "win_rate": round(self.wins / self.races, 4) if self.races else 0.0,
            "finish_rate": round(self.finish_count / self.races, 4) if self.races else 0.0,
            "crash_rate": round(self.crash_count / self.races, 4) if self.races else 0.0,
            "avg_reward": round(avg_reward, 4),
            "avg_lap_time": round(avg_lap_time, 4),
        }


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_ratings(
    rating_a: float, rating_b: float, score_a: float, k_factor: float = 32.0
) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    expected_b = expected_score(rating_b, rating_a)
    score_b = 1.0 - score_a
    return (
        rating_a + k_factor * (score_a - expected_a),
        rating_b + k_factor * (score_b - expected_b),
    )


def outcome_to_score(outcome: str) -> float:
    if outcome == "a_win":
        return 1.0
    if outcome == "b_win":
        return 0.0
    return 0.5

