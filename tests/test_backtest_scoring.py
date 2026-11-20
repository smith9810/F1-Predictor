import numpy as np
import pandas as pd

from f1 import backtest as bt


def test_rps_zero_for_a_perfect_forecast_and_positive_otherwise():
    D = np.eye(4)
    assert np.allclose(bt.rps_rows(D, [1, 2, 3, 4]), 0)
    assert (bt.rps_rows(np.full((4, 4), 0.25), [1, 2, 3, 4]) > 0).all()


def test_score_race_uses_winner_probability():
    s = pd.DataFrame({"driver": list("abc"), "status": ["FIN", "FIN", "DNF"], "pos": [1.0, 2.0, np.nan],
                      "actual": [1, 2, 3]})
    probs = pd.DataFrame({"p_win": [0.5, 0.3, 0.2], "p_podium": [1.0, 1.0, 1.0], "p_points": [1.0, 1.0, 1.0]})
    sc = bt.score_race(np.full((3, 3), 1 / 3), probs, s)
    assert np.isclose(sc["win_ll"], -np.log(0.5))
    assert np.isclose(sc["win_brier"], 0.25 + 0.09 + 0.04)


def test_table_forecast_normalises():
    M = np.ones((26, 26)) + np.diag(np.full(26, 10.0))
    D, p = bt.table_forecast(M, [1, 2, 3, 4, 5], 5)
    assert np.allclose(D.sum(axis=1), 1)
    assert np.isclose(p["p_win"].sum(), 1) and np.isclose(p["p_podium"].sum(), 3)
    assert p["p_win"].iloc[0] == p["p_win"].max()
