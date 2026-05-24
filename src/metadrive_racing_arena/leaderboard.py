from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .elo import EloRecord
from .utils import ensure_dir, read_json, write_json


class Leaderboard:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, EloRecord] = {}
        self.load()

    def load(self) -> None:
        raw = read_json(self.path, default={}) or {}
        for policy_id, row in raw.items():
            self.records[policy_id] = EloRecord(
                policy_id=policy_id,
                rating=float(row.get("elo", row.get("rating", 1000.0))),
                wins=int(row.get("wins", 0)),
                losses=int(row.get("losses", 0)),
                draws=int(row.get("draws", 0)),
                races=int(row.get("races", 0)),
                reward_sum=float(row.get("reward_sum", row.get("avg_reward", 0.0) * max(row.get("races", 0), 1))),
                lap_time_sum=float(row.get("lap_time_sum", 0.0)),
                crash_count=int(row.get("crash_count", int(row.get("crash_rate", 0.0) * max(row.get("races", 0), 1)))),
                finish_count=int(row.get("finish_count", int(row.get("finish_rate", 0.0) * max(row.get("races", 0), 1)))),
            )

    def ensure(self, policy_id: str) -> EloRecord:
        if policy_id not in self.records:
            self.records[policy_id] = EloRecord(policy_id=policy_id)
        return self.records[policy_id]

    def sorted_rows(self) -> list[dict]:
        return [record.to_dict() for record in sorted(self.records.values(), key=lambda r: r.rating, reverse=True)]

    def save(self) -> None:
        payload = {
            record.policy_id: {
                **record.to_dict(),
                "rating": record.rating,
                "reward_sum": record.reward_sum,
                "lap_time_sum": record.lap_time_sum,
                "crash_count": record.crash_count,
                "finish_count": record.finish_count,
            }
            for record in self.records.values()
        }
        write_json(self.path, payload)

    def export_json(self, path: str | Path) -> None:
        write_json(path, self.sorted_rows())

    def export_csv(self, path: str | Path) -> None:
        rows = self.sorted_rows()
        ensure_dir(Path(path).parent)
        with Path(path).open("w", newline="", encoding="utf-8") as fh:
            if not rows:
                fh.write("")
                return
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def format_leaderboard(rows: Iterable[dict]) -> str:
    rows = list(rows)
    if not rows:
        return "No policies ranked yet."
    columns = ["rank", "policy_id", "elo", "wins", "losses", "draws", "win_rate", "finish_rate", "crash_rate"]
    table_rows = []
    for rank, row in enumerate(rows, start=1):
        table_rows.append({**row, "rank": rank})
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in table_rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    sep = "  ".join("-" * widths[column] for column in columns)
    body = [
        "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in table_rows
    ]
    return "\n".join([header, sep, *body])

