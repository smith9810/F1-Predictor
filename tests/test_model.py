import numpy as np
import pandas as pd

from f1 import model


def test_pl_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(3, 6))
    mask = np.ones((3, 6), bool)
    mask[1, 4:] = False
    ll, g = model.pl_loglik(u, mask)
    eps = 1e-6
    for e, k in [(0, 0), (0, 5), (1, 2), (2, 3)]:
        up = u.copy()
        up[e, k] += eps
        num = (model.pl_loglik(up, mask)[0][e] - ll[e]) / eps
        assert abs(num - g[e, k]) < 1e-4


def test_pl_probability_of_two_driver_order():
    u = np.array([[1.0, 0.0]])
    ll, _ = model.pl_loglik(u, np.ones((1, 2), bool))
    assert np.isclose(np.exp(ll[0]), np.exp(1) / (np.exp(1) + 1))


def synthetic_tables(n_races=40, seed=0):
    """Two seasons, 4 teams x 2 drivers, known strengths: driver skill + car pace."""
    rng = np.random.default_rng(seed)
    drivers = [f"d{i}" for i in range(8)]
    teams = {f"d{i}": f"t{i // 2}" for i in range(8)}
    skill = {d: s for d, s in zip(drivers, np.linspace(0.4, -0.4, 8))}
    car = {"t0": 1.0, "t1": 0.4, "t2": -0.2, "t3": -1.0}
    races, results, quali = [], [], []
    for r in range(n_races):
        date = pd.Timestamp("2029-03-01") + pd.Timedelta(days=14 * r)
        season = date.year
        rid = 1000 + r
        races.append({"race_id": rid, "season": season, "round": r + 1, "date": date, "time_utc": "13:00",
                      "gp": "x", "name": "x", "circuit": "c", "circuit_type": "RACE", "laps": 50,
                      "qualifying_date": str((date - pd.Timedelta(days=1)).date()), "qualifying_time": "14:00",
                      "sprint": False, "folder": "x"})
        strength = np.array([skill[d] + car[teams[d]] for d in drivers])
        q = 3 * strength + rng.gumbel(size=8)
        qorder = np.argsort(-q)
        grid = np.empty(8)
        grid[qorder] = np.arange(1, 9)
        perf = strength + 0.8 * -np.log(grid) + rng.gumbel(size=8)
        order = np.argsort(-perf)
        for k, i in enumerate(qorder):
            quali.append({"race_id": rid, "season": season, "round": r + 1, "driver": drivers[i],
                          "constructor": teams[drivers[i]], "pos": float(k + 1), "best_ms": 80000.0 + k})
        for k, i in enumerate(order):
            results.append({"race_id": rid, "season": season, "round": r + 1, "driver": drivers[i],
                            "constructor": teams[drivers[i]], "engine": "e", "number": i, "pos": float(k + 1),
                            "status": "FIN", "grid": grid[i], "laps": 50, "points": 0.0, "reason": None,
                            "shared": False})
    empty = pd.DataFrame(columns=["race_id", "season", "round", "driver", "constructor", "pos", "status", "grid"])
    return {"races": pd.DataFrame(races), "results": pd.DataFrame(results), "quali": pd.DataFrame(quali),
            "sprint": empty, "standings": pd.DataFrame(columns=["race_id", "season", "round", "driver", "pos", "points"]),
            "practice": pd.DataFrame(columns=["race_id", "season", "round", "session", "driver", "constructor", "pos", "best_ms"])}, skill, car, teams


def test_fit_recovers_the_strength_order():
    t, skill, car, teams = synthetic_tables(60)
    ev = model.build_events(t, t["races"]["date"].max() + pd.Timedelta(days=1), {"half_life_days": 5000})
    R = model.fit(ev, {"half_life_days": 5000, "link": 0.0})
    season = int(t["races"]["season"].max())
    est = R.pace(list(skill), [teams[d] for d in skill], season)
    true = np.array([skill[d] + car[teams[d]] for d in skill])
    assert np.corrcoef(est, true)[0, 1] > 0.95
    assert 1.5 < R.kappa[1] < 6                     # qualifying is more orderly than the race (true 3)
    assert 0.3 < R.gamma[0] < 1.5                   # grid effect (true 0.8)


def test_no_look_ahead_sessions_on_or_after_as_of_are_excluded():
    t, *_ = synthetic_tables(10)
    cut = t["races"]["date"].iloc[5]
    ev = model.build_events(t, cut)
    # 5 races before the cut, plus the qualifying the day before race 6 (before the cut)
    assert (ev.typ == 0).sum() == 5 and (ev.typ == 1).sum() == 6
    t2 = {k: v.copy() for k, v in t.items()}
    late = t2["results"]["race_id"] >= t2["races"]["race_id"].iloc[5]
    t2["results"].loc[late, "pos"] = t2["results"].loc[late, "pos"].to_numpy()[::-1]   # rewrite the future
    a = model.fit(ev)
    b = model.fit(model.build_events(t2, cut))
    assert np.allclose(a.d, b.d) and np.allclose(a.c, b.c)


def test_new_season_car_uses_last_season_shrunk():
    t, *_ = synthetic_tables(30)
    ev = model.build_events(t, t["races"]["date"].max() + pd.Timedelta(days=1))
    R = model.fit(ev)
    last = int(t["races"]["season"].max())
    nxt = R.car_pace("t0", last + 1)
    assert 0 < nxt < R.car_pace("t0", last)
    assert R.car_pace("unknown-team", last) == 0.0
