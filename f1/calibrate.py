"""Recalibration of win and podium probabilities, fitted walk-forward on earlier out-of-sample forecasts.

Why: the backtest shows the raw model is under-confident about favourites (drivers given 30-40% to win
won 55% of the time) and over-confident about outsiders. One exponent fixes most of it:
    p_win' = p_win ** a / sum(p_win ** a)       (a > 1 sharpens), chosen to maximise the winner's log-likelihood
    podium / points: logistic recalibration logit(p') = alpha + beta * logit(p), then scaled so that the
    field's podium probabilities add up to 3 (points: 10), capped at 1.
Each season uses parameters fitted on the previous `years` seasons only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sharpen_win(p, a: float) -> np.ndarray:
    lp = a * np.log(np.clip(np.asarray(p, float), EPS, 1.0))
    lp -= lp.max()
    e = np.exp(lp)
    return e / e.sum()


def fit_win_power(fc: pd.DataFrame) -> float:
    """fc: one row per driver per race with p_win and won (0/1)."""
    groups = [(g["p_win"].to_numpy(float), g["won"].to_numpy(int)) for _, g in fc.groupby("race_id") if g["won"].sum() == 1]

    def nll(a):
        return -sum(np.log(sharpen_win(p, a)[w.argmax()] + 1e-12) for p, w in groups)

    return float(minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded").x)


def fit_platt(p, y) -> tuple[float, float]:
    x, y = _logit(np.asarray(p, float)), np.asarray(y, float)

    def nll(th):
        q = 1 / (1 + np.exp(-(th[0] + th[1] * x)))
        q = np.clip(q, EPS, 1 - EPS)
        return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))

    r = minimize(nll, np.array([0.0, 1.0]), method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def apply_platt(p, ab, total: float) -> np.ndarray:
    q = 1 / (1 + np.exp(-(ab[0] + ab[1] * _logit(np.asarray(p, float)))))
    for _ in range(20):                       # rescale to the field total, keeping each at most 1
        q = np.minimum(q * total / q.sum(), 1.0)
        if abs(q.sum() - total) < 1e-9:
            break
    return q


def fit_all(fc: pd.DataFrame) -> dict:
    """Parameters for one stage, from a frame with p_win, p_podium, p_points and outcomes."""
    return {"win_power": fit_win_power(fc), "podium_platt": fit_platt(fc["p_podium"], fc["podium"]),
            "points_platt": fit_platt(fc["p_points"], fc["points_finish"]), "rows": int(len(fc)),
            "seasons": f"{int(fc['season'].min())}-{int(fc['season'].max())}"}


def outcomes(fc: pd.DataFrame) -> pd.DataFrame:
    f = fc.copy()
    fin = f["status"] == "FIN"
    f["won"] = (fin & (f["pos"] == 1)).astype(int)
    f["podium"] = (fin & (f["pos"] <= 3)).astype(int)
    f["points_finish"] = (fin & (f["pos"] <= 10)).astype(int)
    return f


def apply(race_fc: pd.DataFrame, par: dict) -> pd.DataFrame:
    """Calibrated copies of p_win, p_podium, p_points for one race (keeps the raw values as *_raw)."""
    out = race_fc.copy()
    for c in ("p_win", "p_podium", "p_points"):
        out[f"{c}_raw"] = out[c]
    n = len(out)
    out["p_win"] = sharpen_win(out["p_win_raw"], par["win_power"])
    out["p_podium"] = apply_platt(out["p_podium_raw"], par["podium_platt"], min(3, n))
    out["p_points"] = apply_platt(out["p_points_raw"], par["points_platt"], min(10, n))
    return out


def walk_forward(fc: pd.DataFrame, first_season: int, years: int = 6) -> pd.DataFrame:
    """Calibrated forecasts for seasons >= first_season, each from the previous `years` seasons."""
    f = outcomes(fc)
    out = []
    for (season, stage), g in f[f["season"] >= first_season].groupby(["season", "stage"]):
        past = f[(f["stage"] == stage) & (f["season"] < season) & (f["season"] >= season - years)]
        par = fit_all(past)
        for _, r in g.groupby("race_id"):
            out.append(apply(r, par))
    return pd.concat(out, ignore_index=True)
