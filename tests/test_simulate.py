import numpy as np

from f1 import simulate as sim


def test_probabilities_add_up():
    pace = np.linspace(1, -1, 20)
    pos, dnf = sim.simulate(pace, np.full(20, 0.1), 2.0, 0.5, None, 4000, 1)
    s = sim.summary(pos, dnf)
    assert np.isclose(s["p_win"].sum(), 1.0)
    assert np.isclose(s["p_podium"].sum(), 3.0)
    assert np.isclose(s["exp_pos"].sum(), sum(range(1, 21)))
    assert s["p_win"].iloc[0] > s["p_win"].iloc[-1]
    assert abs(s["p_dnf"].mean() - 0.1) < 0.01


def test_dominant_driver_wins_unless_he_retires():
    pace = np.array([20.0, 0, 0, 0])
    pos, dnf = sim.simulate(pace, np.array([0.25, 0, 0, 0]), 2.0, 0.0, np.array([1, 2, 3, 4]), 20000, 2)
    s = sim.summary(pos, dnf)
    assert abs(s["p_win"].iloc[0] - 0.75) < 0.02
    assert (pos[dnf[:, 0], 0] == 4).all()          # a retirement takes the last place


def test_grid_matters_when_gamma_positive():
    pace = np.zeros(10)
    pos, dnf = sim.simulate(pace, np.zeros(10), 2.0, 1.0, np.arange(1, 11), 20000, 3)
    s = sim.summary(pos, dnf)
    assert s["p_win"].iloc[0] > 3 * s["p_win"].iloc[9]


def test_position_distribution_rows_sum_to_one():
    pos, dnf = sim.simulate(np.linspace(1, -1, 6), np.zeros(6), 2.0, 0.3, None, 2000, 4)
    D = sim.position_dist(pos)
    assert np.allclose(D.sum(axis=1), 1) and np.allclose(D.sum(axis=0), 1)


def test_calibration_sharpens_favourites_and_keeps_totals():
    import pandas as pd
    from f1 import calibrate
    p = np.array([0.3, 0.2, 0.1] + [0.4 / 17] * 17)
    q = calibrate.sharpen_win(p, 1.6)
    assert np.isclose(q.sum(), 1) and q[0] > p[0] and q[-1] < p[-1]
    pod = calibrate.apply_platt(np.linspace(0.6, 0.02, 20), (0.4, 1.4), 3)
    assert np.isclose(pod.sum(), 3) and (pod <= 1).all()
