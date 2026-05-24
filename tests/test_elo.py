from metadrive_racing_arena.elo import expected_score, outcome_to_score, update_ratings


def test_expected_score_equal_ratings():
    assert expected_score(1000.0, 1000.0) == 0.5


def test_rating_winner_gains_points():
    new_a, new_b = update_ratings(1000.0, 1000.0, score_a=1.0, k_factor=32.0)
    assert new_a == 1016.0
    assert new_b == 984.0


def test_outcome_mapping():
    assert outcome_to_score("a_win") == 1.0
    assert outcome_to_score("b_win") == 0.0
    assert outcome_to_score("draw") == 0.5

