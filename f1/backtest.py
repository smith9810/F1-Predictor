"""Walk-forward backtest: each race is forecast twice, using only sessions before the forecast time.

  pre   before qualifying (sessions before the qualifying date); the grid is simulated
  post  after qualifying (sessions before race day, so this weekend's qualifying and sprint count);
        the real starting grid is used

Baselines, also walk-forward:
  uniform  every starter equally likely
  grid     P(finishing place | starting slot) from the previous 10 seasons (post-qualifying baseline)
  form     the grid table applied to a grid guessed from average qualifying place in the last 5 races
           (pre-qualifying baseline)

Scores per race (lower is better): win_ll = -log P(winner), win_brier, podium_brier, points_brier and
pos_rps = ranked probability score of each driver's finishing-place distribution (uses the whole order).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from f1 import dnf as dnf_mod, model, simulate as sim

STARTERS = ("FIN", "DNF", "NC", "DSQ", "EX")


def starters(t: dict, race_id: int) -> pd.DataFrame:
    r = t["results"]
    g = r[(r["race_id"] == race_id) & r["status"].isin(STARTERS)].copy()
    g["fin"] = (g["status"] == "FIN").astype(int)
    g = g.sort_values(["fin", "pos", "laps"], ascending=[False, True, False], na_position="last")
    g["actual"] = np.arange(1, len(g) + 1)          # classified order, then retirements by laps
    return g.reset_index(drop=True)


def grid_table(t: dict, season: int, years: int = 10, max_slot: int = 26) -> np.ndarray:
    """P(finish place k | grid slot g), rows g = 1..max_slot, columns k = 1..max_slot (Laplace-smoothed)."""
    r = t["results"]
    r = r[(r["season"] < season) & (r["season"] >= season - years) & r["status"].isin(STARTERS)]
    M = np.ones((max_slot, max_slot)) * 0.5
    for rid, g in r.groupby("race_id"):
        s = starters({"results": g}, rid)
        gs = np.clip(s["grid"].fillna(len(s)).astype(int).to_numpy(), 1, max_slot)
        fs = np.clip(s["actual"].to_numpy(), 1, max_slot)
        np.add.at(M, (gs - 1, fs - 1), 1.0)
    return M


def table_forecast(M: np.ndarray, grid, N: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-driver place distributions and win/podium/points probabilities from a grid table."""
    g = np.clip(np.asarray(grid, float), 1, M.shape[0]).astype(int)
    D = M[g - 1, :N]
    D = D / D.sum(axis=1, keepdims=True)
    probs = {}
    for name, k in (("p_win", 1), ("p_podium", 3), ("p_points", 10)):
        v = D[:, :k].sum(axis=1)
        probs[name] = v / v.sum() * min(k, N)
    return D, pd.DataFrame(probs)


def rps_rows(D: np.ndarray, actual) -> np.ndarray:
    N = D.shape[1]
    F = np.cumsum(D, axis=1)[:, :-1]
    O = (np.asarray(actual)[:, None] <= np.arange(1, N)[None, :]).astype(float)
    return ((F - O) ** 2).sum(axis=1) / (N - 1)


def score_race(D: np.ndarray, probs: pd.DataFrame, s: pd.DataFrame) -> dict:
    a = s["actual"].to_numpy()
    won = (s["status"] == "FIN") & (s["pos"] == 1)
    pod = (s["status"] == "FIN") & (s["pos"] <= 3)
    pts = (s["status"] == "FIN") & (s["pos"] <= 10)
    pw = np.clip(probs["p_win"].to_numpy(), 1e-4, 1)
    return {"win_ll": float(-np.log(pw[won.to_numpy()]).sum()) if won.any() else np.nan,
            "win_brier": float(((probs["p_win"].to_numpy() - won) ** 2).sum()),
            "podium_brier": float(((probs["p_podium"].to_numpy() - pod) ** 2).mean()),
            "points_brier": float(((probs["p_points"].to_numpy() - pts) ** 2).mean()),
            "pos_rps": float(rps_rows(D, a).mean())}


def form_grid(t: dict, race_id: int, drivers, n_races: int = 5) -> np.ndarray:
    q = t["quali"]
    races = t["races"].sort_values(["season", "round"])
    order = races["race_id"].tolist()
    i = order.index(race_id)
    prev = order[max(0, i - n_races):i]
    qq = q[q["race_id"].isin(prev) & q["pos"].notna()]
    avg = qq.groupby("driver")["pos"].mean()
    v = np.array([avg.get(d, 99.0) for d in drivers])
    return pd.Series(v).rank(method="first").to_numpy()


class DNFModels:
    """Yearly-refitted retirement models (pre- and post-qualifying)."""

    def __init__(self, t: dict, years: int = 10):
        self.f = dnf_mod.features(t)
        self.years, self.cache = years, {}

    def models(self, season: int):
        if season not in self.cache:
            tr = self.f[(self.f["season"] < season) & (self.f["season"] >= season - self.years)]
            self.cache[season] = (dnf_mod.fit(tr, False), dnf_mod.fit(tr, True))
        return self.cache[season]

    def probs(self, race_id: int, drivers, season: int, post: bool, grid=None) -> np.ndarray:
        rows = self.f[self.f["race_id"] == race_id].set_index("driver").reindex(list(drivers)).reset_index()
        rows[dnf_mod.COLS_PRE] = rows[dnf_mod.COLS_PRE].fillna(rows[dnf_mod.COLS_PRE].median())
        if grid is not None:
            rows["grid"] = np.asarray(grid, float)
        pre, pst = self.models(season)
        return dnf_mod.predict(pst if post else pre, rows, post)


def forecast(t: dict, race: pd.Series, entries: pd.DataFrame, stage: str, params: dict, dnfm: DNFModels,
             n_sims: int = 5000, seed: int = 0, init=None):
    """Returns (ratings, per-driver summary, place distributions)."""
    if stage == "pre":
        q = pd.to_datetime(race.get("qualifying_date"), errors="coerce")
        as_of = q if pd.notna(q) else pd.Timestamp(race["date"]) - pd.Timedelta(days=1)
    else:
        as_of = pd.Timestamp(race["date"])
    ev = model.build_events(t, as_of, params)
    R = model.fit(ev, params, init)
    pace = R.pace(entries["driver"], entries["constructor"], int(race["season"]))
    grid = entries["grid"].to_numpy(float) if stage == "post" else None
    p_dnf = dnfm.probs(int(race["race_id"]), entries["driver"], int(race["season"]), stage == "post", grid)
    pos, dn = sim.simulate(pace, p_dnf, R.kappa[1], R.gamma[0], grid, n_sims, seed)
    out = sim.summary(pos, dn)
    out.insert(0, "driver", entries["driver"].to_numpy())
    out.insert(1, "constructor", entries["constructor"].to_numpy())
    out["pace"] = pace
    return R, out, sim.position_dist(pos)


def run(t: dict, first_season: int, last_season: int, params: dict | None = None, stages=("pre", "post"),
        n_sims: int = 5000, dnfm: DNFModels | None = None, keep_forecasts: bool = False):
    params = {**model.DEFAULT, **(params or {})}
    dnfm = dnfm or DNFModels(t)
    races = t["races"]
    done = set(t["results"].loc[t["results"]["status"] == "FIN", "race_id"])
    ev_races = races[(races["season"] >= first_season) & (races["season"] <= last_season)
                     & races["race_id"].isin(done)].sort_values(["season", "round"])
    rows, fcs = [], []
    tables = {}
    prev = {"pre": None, "post": None}
    for _, race in ev_races.iterrows():
        s = starters(t, int(race["race_id"]))
        if len(s) < 6:
            continue
        N = len(s)
        season = int(race["season"])
        if season not in tables:
            tables[season] = grid_table(t, season)
        base = {"race_id": int(race["race_id"]), "season": season, "round": int(race["round"]),
                "gp": race["gp"], "n": N}
        # uniform baseline
        U = np.full((N, N), 1.0 / N)
        up = pd.DataFrame({"p_win": np.full(N, 1 / N), "p_podium": np.full(N, 3 / N), "p_points": np.full(N, min(10, N) / N)})
        rows.append({**base, "stage": "pre", "method": "uniform", **score_race(U, up, s)})
        rows.append({**base, "stage": "post", "method": "uniform", **score_race(U, up, s)})
        # grid and form baselines
        D, pr = table_forecast(tables[season], s["grid"].fillna(N).to_numpy(), N)
        rows.append({**base, "stage": "post", "method": "grid", **score_race(D, pr, s)})
        D, pr = table_forecast(tables[season], form_grid(t, int(race["race_id"]), s["driver"]), N)
        rows.append({**base, "stage": "pre", "method": "form", **score_race(D, pr, s)})
        for st in stages:
            try:
                R, out, D = forecast(t, race, s, st, params, dnfm, n_sims, seed=int(race["race_id"]), init=prev[st])
            except ValueError:
                continue
            prev[st] = R
            rows.append({**base, "stage": st, "method": "model", **score_race(D, out, s),
                         "kappa_q": R.kappa[1], "gamma": R.gamma[0]})
            if keep_forecasts:
                o = out.copy()
                o["race_id"], o["season"], o["round"], o["gp"], o["stage"] = base["race_id"], season, base["round"], race["gp"], st
                o["actual"], o["status"], o["pos"], o["grid"] = s["actual"].to_numpy(), s["status"].to_numpy(), s["pos"].to_numpy(), s["grid"].to_numpy()
                fcs.append(o)
    res = pd.DataFrame(rows)
    return (res, pd.concat(fcs, ignore_index=True) if fcs else pd.DataFrame()) if keep_forecasts else res


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    cols = ["win_ll", "win_brier", "podium_brier", "points_brier", "pos_rps"]
    return res.groupby(["stage", "method"])[cols].mean().reset_index().merge(
        res.groupby(["stage", "method"]).size().rename("races").reset_index())
