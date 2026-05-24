from metadrive_racing_arena.leaderboard import Leaderboard, format_leaderboard


def test_leaderboard_sorts_by_elo(tmp_path):
    board = Leaderboard(tmp_path / "leaderboard.json")
    board.ensure("slow").rating = 900.0
    board.ensure("fast").rating = 1200.0
    rows = board.sorted_rows()
    assert [row["policy_id"] for row in rows] == ["fast", "slow"]


def test_leaderboard_exports_csv(tmp_path):
    board = Leaderboard(tmp_path / "leaderboard.json")
    board.ensure("agent").wins = 1
    output = tmp_path / "leaderboard.csv"
    board.export_csv(output)
    assert "policy_id" in output.read_text(encoding="utf-8")
    assert "agent" in output.read_text(encoding="utf-8")


def test_format_empty_leaderboard():
    assert format_leaderboard([]) == "No policies ranked yet."

