import numpy as np
import pandas as pd

from f1 import title


def _season(points_after, n_rounds, wins=None, season=2030):
    races = pd.DataFrame({"race_id": np.arange(n_rounds) + 1, "season": season, "round": np.arange(1, n_rounds + 1),
                          "sprint": False})
    st, res = [], []
    for i, (a, b) in enumerate(points_after):
        st += [{"race_id": i + 1, "driver": "A", "points": a}, {"race_id": i + 1, "driver": "B", "points": b}]
        w = (wins or {}).get(i + 1, "A")
        res.append({"race_id": i + 1, "driver": w, "pos": 1.0, "status": "FIN"})
    return {"races": races, "standings": pd.DataFrame(st), "results": pd.DataFrame(res)}


def test_clinch_when_lead_exceeds_points_left():
    # 2030 rules: 25 a race. After round 2 of 5 the lead is 80 > 75 left.
    t = _season([(25, 18), (98, 18)], 5)
    c = title.clinch_rounds(t, first=2030).iloc[0]
    assert c["clinch_round"] == 2 and c["races_after"] == 3 and c["champion"] == "A"


def test_level_at_best_counts_only_if_countback_is_safe():
    # lead 75 with 75 left: B could tie. A has 2 wins, B 0 + 3 possible = 3 -> not decided.
    t = _season([(25, 0), (75, 0)], 5)
    assert np.isnan(title.clinch_rounds(t, first=2030).iloc[0]["clinch_round"])


def test_temperature_estimate_recovers_known_scale():
    rng = np.random.default_rng(0)
    U = rng.normal(size=(400, 10))
    order = np.argsort(-(0.5 * U + rng.gumbel(size=U.shape)), axis=1)     # true tau = 0.5
    Us = np.take_along_axis(U, order, axis=1)
    tau = title.best_tau(Us, np.ones_like(Us, bool))
    assert 0.4 < tau < 0.6
