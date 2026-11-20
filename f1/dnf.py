"""Retirement (DNF) probabilities: logistic regression on reliability histories.

Features for each starter, built only from earlier races (time-decayed counts):
  base     log-odds of the recent DNF rate across the whole field (era: e.g. ~25% in 2005, ~10% in 2024)
  team     log ratio of the team's recent DNF rate to the field rate (shrunk: 10 pseudo-starts)
  driver   same for the driver (20 pseudo-starts) - crashes and bad luck
  circuit  same for the circuit, long memory (30 pseudo-starts) - street circuits, first-lap chaos
  grid     log starting slot (post-qualifying only): the back of the grid gets caught in more incidents
Coefficients are refitted each January on the previous 10 seasons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from f1.model import lineage

DNF_STATUS = ("DNF", "NC")
STARTERS = ("FIN", "DNF", "NC", "DSQ", "EX")


def features(t: dict, half_life_races: float = 40.0, circuit_half_life: float = 200.0) -> pd.DataFrame:
    """One row per starter of every race, features computed from races before it."""
    races = t["races"].sort_values(["season", "round"]).reset_index(drop=True)
    res = t["results"]
    res = res[res["status"].isin(STARTERS)]
    by_race = {rid: g for rid, g in res.groupby("race_id")}
    dec, cdec = 0.5 ** (1 / half_life_races), 0.5 ** (1 / circuit_half_life)
    base_n = base_k = 0.0
    team, drv, circ = {}, {}, {}
    rows = []
    for _, r in races.iterrows():
        g = by_race.get(r["race_id"])
        if g is None or len(g) == 0:
            continue
        p0 = (base_k + 2.0) / (base_n + 10.0)
        cn, ck = circ.get(r["circuit"], (0.0, 0.0))     # (expected dnfs at base rate, actual)
        f_circ = np.log((ck + 30 * p0) / (cn + 30 * p0)) if True else 0.0
        for _, e in g.iterrows():
            tl = lineage(e["constructor"])
            tn, tk = team.get(tl, (0.0, 0.0))
            dn, dk = drv.get(e["driver"], (0.0, 0.0))
            rows.append({"race_id": r["race_id"], "season": r["season"], "round": r["round"], "driver": e["driver"],
                         "constructor": e["constructor"], "grid": e["grid"],
                         "f_base": np.log(p0 / (1 - p0)),
                         "f_team": np.log((tk + 10 * p0) / (tn + 10) / p0),
                         "f_driver": np.log((dk + 20 * p0) / (dn + 20) / p0),
                         "f_circuit": f_circ,
                         "dnf": int(e["status"] in DNF_STATUS)})
        # update after the race
        k = g["status"].isin(DNF_STATUS).to_numpy()
        base_n, base_k = base_n * dec ** 1 + len(g), base_k * dec + k.sum()
        for tl in set(team):
            n_, k_ = team[tl]
            team[tl] = (n_ * dec, k_ * dec)
        for dv in list(drv):
            n_, k_ = drv[dv]
            drv[dv] = (n_ * dec, k_ * dec)
        for (_, e), kk in zip(g.iterrows(), k):
            tl = lineage(e["constructor"])
            n_, k_ = team.get(tl, (0.0, 0.0))
            team[tl] = (n_ + 1, k_ + kk)
            n_, k_ = drv.get(e["driver"], (0.0, 0.0))
            drv[e["driver"]] = (n_ + 1, k_ + kk)
        for c in list(circ):
            n_, k_ = circ[c]
            circ[c] = (n_ * cdec, k_ * cdec)
        n_, k_ = circ.get(r["circuit"], (0.0, 0.0))
        circ[r["circuit"]] = (n_ + len(g) * p0, k_ + k.sum())
    return pd.DataFrame(rows)


def state_for(t: dict, upto_race_id: int, entries: pd.DataFrame, circuit: str,
              half_life_races: float = 40.0, circuit_half_life: float = 200.0) -> pd.DataFrame:
    """Features for a race that has not happened: append a fake race with the entry list and read its rows."""
    t2 = {k: v for k, v in t.items()}
    races = t["races"]
    fake_id = int(races["race_id"].max()) + 100000
    last = races[races["race_id"] == upto_race_id].iloc[0]
    fake = pd.DataFrame([{**last.to_dict(), "race_id": fake_id, "round": int(last["round"]) + 0.5,
                          "circuit": circuit}])
    t2["races"] = pd.concat([races[races["race_id"].isin(t["results"]["race_id"])], fake], ignore_index=True)
    e = entries.copy()
    e["race_id"], e["status"] = fake_id, "FIN"
    e["season"], e["round"] = last["season"], fake.iloc[0]["round"]
    if "grid" not in e:
        e["grid"] = np.nan
    t2["results"] = pd.concat([t["results"], e], ignore_index=True)
    f = features(t2, half_life_races, circuit_half_life)
    return f[f["race_id"] == fake_id].reset_index(drop=True)


COLS_PRE = ["f_base", "f_team", "f_driver", "f_circuit"]
COLS_POST = COLS_PRE + ["f_grid"]


def design(f: pd.DataFrame, post: bool) -> np.ndarray:
    X = f[COLS_PRE].to_numpy(float)
    if post:
        g = f["grid"].to_numpy(float)
        g = np.where(np.isfinite(g) & (g > 0), g, np.nanmedian(np.where(g > 0, g, np.nan)) if np.isfinite(g).any() else 10.0)
        X = np.column_stack([X, np.log(g)])
    return X


def fit(f: pd.DataFrame, post: bool) -> LogisticRegression:
    m = LogisticRegression(C=10.0, max_iter=1000)
    m.fit(design(f, post), f["dnf"].to_numpy(int))
    return m


def predict(m: LogisticRegression, f: pd.DataFrame, post: bool) -> np.ndarray:
    return m.predict_proba(design(f, post))[:, 1]
