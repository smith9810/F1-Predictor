"""Driver + car strength model fitted to finishing orders (Plackett-Luce, a.k.a. rank-ordered logit).

For driver i in team-season k at session e:
    u = kappa[type] * (d[i] + c[k]) + gamma[type] * (-log grid)        (grid term in races/sprints only)
The probability of an observed order is the product, position by position, of exp(u) of the driver who
took that position over the sum of exp(u) of everyone not yet placed. Equivalent: performance = u plus
standard Gumbel noise, highest first.

  d       driver skill, ridge prior towards 0 (= average driver). v2: one value per driver per season, linked
          to the previous season, so a driver can improve (young drivers fastest) or decline
  c       car pace for one team in one season; prior links it to the same team's previous season
          (weakly in seasons with new technical rules: 2009, 2014, 2017, 2022, 2026); team renames are
          followed through LINEAGE (Toro Rosso -> AlphaTauri -> RB -> Racing Bulls, etc.)
  kappa   how deterministic a session type is: race = 1 (reference), qualifying and sprint fitted
  gamma   value of track position: a better starting slot raises the race strength
Sessions: race finishing order (classified finishers only - retirements are modelled separately in
`dnf.py`), qualifying order, sprint order. Older sessions count less: weight halves every `half_life`
days. Fitted by penalised maximum likelihood (L-BFGS, analytic gradient).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

NEW_RULES = {2009, 2014, 2017, 2022, 2026}
LINEAGE = {
    "benetton": "enstone", "renault": "enstone", "lotus-f1": "enstone", "alpine": "enstone",
    "jordan": "silverstone", "midland": "silverstone", "spyker": "silverstone", "force-india": "silverstone",
    "racing-point": "silverstone", "aston-martin": "silverstone",
    "minardi": "faenza", "toro-rosso": "faenza", "alphatauri": "faenza", "rb": "faenza", "racing-bulls": "faenza",
    "sauber": "hinwil", "bmw-sauber": "hinwil", "alfa-romeo": "hinwil", "kick-sauber": "hinwil", "audi": "hinwil",
    "bar": "brackley", "honda": "brackley", "brawn": "brackley", "mercedes": "brackley",
    "jaguar": "milton-keynes", "stewart": "milton-keynes", "red-bull": "milton-keynes",
    "lotus-racing": "caterham", "caterham": "caterham",
    "virgin": "banbury", "marussia": "banbury", "manor": "banbury",
}
TYPES = ("race", "quali", "sprint")

DEFAULT = {"half_life_days": 365.0, "ridge_driver": 0.3, "ridge_car": 0.3, "link": 1.0, "link_new_rules": 0.3,
           "window_seasons": 4, "use_quali": True, "use_sprint": True,
           # driver_link None = one skill per driver for all seasons (v1). A number = one skill per driver per
           # season, linked to the previous season with this strength (weaker in a driver's first 3 seasons).
           "driver_link": None, "driver_link_young": None}


def lineage(constructor: str) -> str:
    return LINEAGE.get(constructor, constructor)


def car_key(constructor: str, season: int) -> str:
    return f"{lineage(constructor)}:{int(season)}"


@dataclass
class Events:
    """Sessions padded into an E x M matrix, columns in finishing order."""
    drv: np.ndarray          # E x M driver index (-1 = padding)
    car: np.ndarray          # E x M car index
    g: np.ndarray            # E x M -log(grid) (0 where unknown or not a race)
    has_grid: np.ndarray     # E x M bool
    typ: np.ndarray          # E session type index (0 race, 1 quali, 2 sprint)
    w: np.ndarray            # E weights
    mask: np.ndarray         # E x M bool
    drivers: list
    cars: list
    car_season: dict = field(default_factory=dict)
    young: set = field(default_factory=set)


@dataclass
class Ratings:
    drivers: list
    d: np.ndarray
    cars: list
    c: np.ndarray
    kappa: np.ndarray        # race, quali, sprint
    gamma: np.ndarray        # race, quali (unused, 0), sprint
    params: dict
    n_events: int = 0

    def driver_skill(self, driver: str, season: int | None = None) -> float:
        idx = {x: j for j, x in enumerate(self.drivers)}
        if self.params.get("driver_link") is None or season is None:
            i = idx.get(driver)
            if i is None and season is not None:            # per-season fit asked for a plain name
                i = idx.get(f"{driver}:{season}")
            return float(self.d[i]) if i is not None else 0.0
        k = f"{driver}:{int(season)}"
        if k in idx:
            return float(self.d[idx[k]])
        prev = f"{driver}:{int(season) - 1}"
        if prev in idx:                                  # random walk: last season's value carries over
            return float(self.d[idx[prev]])
        return 0.0

    def car_pace(self, constructor: str, season: int) -> float:
        cars = {x: j for j, x in enumerate(self.cars)}
        k = car_key(constructor, season)
        if k in cars:
            return float(self.c[cars[k]])
        prev = car_key(constructor, season - 1)
        if prev in cars:                            # new season, no session yet: prior mean
            link = self.params["link_new_rules"] if season in NEW_RULES else self.params["link"]
            return float(self.c[cars[prev]] * link / (link + self.params["ridge_car"]))
        return 0.0

    def pace(self, drivers, constructors, season) -> np.ndarray:
        return np.array([self.driver_skill(a, season) + self.car_pace(b, season) for a, b in zip(drivers, constructors)])

    def table(self, entries: pd.DataFrame, season: int) -> pd.DataFrame:
        t = entries[["driver", "constructor"]].copy()
        t["driver_skill"] = [self.driver_skill(x, season) for x in t["driver"]]
        t["car_pace"] = [self.car_pace(x, season) for x in t["constructor"]]
        t["pace"] = t["driver_skill"] + t["car_pace"]
        return t.sort_values("pace", ascending=False).reset_index(drop=True)


def _session_rows(tbl: pd.DataFrame, typ: str, races: pd.DataFrame, as_of, weight_from) -> list:
    """One entry per session: (date, type, [driver], [constructor], [grid]) in finishing order."""
    out = []
    if tbl is None or len(tbl) == 0:
        return out
    date_col = races.set_index("race_id")["date"]
    qdate = races.set_index("race_id")["qualifying_date"]
    sub = tbl[tbl["race_id"].isin(date_col.index)]
    if typ in ("race", "sprint"):
        sub = sub[sub["status"] == "FIN"]
    else:
        sub = sub[sub["pos"].notna()]
    for rid, g in sub.groupby("race_id", sort=False):
        g = g.sort_values("pos")
        when = pd.Timestamp(date_col[rid])
        if typ == "quali":
            q = pd.to_datetime(qdate.get(rid), errors="coerce")
            when = q if pd.notna(q) else when - pd.Timedelta(days=1)
        elif typ == "sprint":
            when = when - pd.Timedelta(days=1)
        if when >= as_of:
            continue
        grid = g["grid"].to_numpy(float) if "grid" in g else np.full(len(g), np.nan)
        out.append((when, typ, g["driver"].tolist(), g["constructor"].tolist(), g["season"].iloc[0], grid))
    return out


def build_events(t: dict, as_of, params: dict | None = None) -> Events:
    """All sessions strictly before `as_of` within the last `window_seasons` seasons."""
    p = {**DEFAULT, **(params or {})}
    as_of = pd.Timestamp(as_of)
    races = t["races"]
    first = as_of.year - p["window_seasons"]
    races = races[(races["season"] >= first) & (races["date"] < as_of + pd.Timedelta(days=1))]
    sess = _session_rows(t["results"], "race", races, as_of, None)
    if p["use_quali"]:
        sess += _session_rows(t["quali"], "quali", races, as_of, None)
    if p["use_sprint"]:
        sess += _session_rows(t["sprint"], "sprint", races, as_of, None)
    if not sess:
        raise ValueError("no sessions before " + str(as_of))
    per_season = p.get("driver_link") is not None
    dkey = (lambda x, season: f"{x}:{int(season)}") if per_season else (lambda x, season: x)
    drivers = sorted({dkey(x, s[4]) for s in sess for x in s[2]})
    cars = sorted({car_key(c, s[4]) for s in sess for c in s[3]})
    di, ci = {x: i for i, x in enumerate(drivers)}, {x: i for i, x in enumerate(cars)}
    young = set()
    if per_season:
        res = t["results"]
        debut = res[res["status"].isin(("FIN", "DNF", "NC", "DSQ", "EX"))].groupby("driver")["season"].min()
        young = {k for k in drivers if int(k.rsplit(":", 1)[1]) - debut.get(k.rsplit(":", 1)[0], -99) <= 2}
    M = max(len(s[2]) for s in sess)
    E = len(sess)
    drv = np.full((E, M), -1, int)
    car = np.full((E, M), -1, int)
    g = np.zeros((E, M))
    hg = np.zeros((E, M), bool)
    typ = np.zeros(E, int)
    w = np.zeros(E)
    for e, (when, ty, ds, cs, season, grid) in enumerate(sess):
        n = len(ds)
        drv[e, :n] = [di[dkey(x, season)] for x in ds]
        car[e, :n] = [ci[car_key(c, season)] for c in cs]
        typ[e] = TYPES.index(ty)
        w[e] = 0.5 ** ((as_of - when).days / p["half_life_days"])
        if ty != "quali":
            ok = np.isfinite(grid) & (grid > 0)
            g[e, :n] = np.where(ok, -np.log(np.where(ok, grid, 1.0)), 0.0)
            hg[e, :n] = ok
    mask = drv >= 0
    car_season = {k: int(k.split(":")[1]) for k in cars}
    return Events(drv, car, g, hg, typ, w, mask, drivers, cars, car_season, young)


def _links(ev: Events) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = {k: i for i, k in enumerate(ev.cars)}
    a, b, new = [], [], []
    for k, i in idx.items():
        lin, s = k.split(":")
        prev = f"{lin}:{int(s) - 1}"
        if prev in idx:
            a.append(i)
            b.append(idx[prev])
            new.append(int(s) in NEW_RULES)
    return np.array(a, int), np.array(b, int), np.array(new, bool)


def pl_loglik(u: np.ndarray, mask: np.ndarray):
    """Plackett-Luce log-likelihood per session and d/du. u is E x M in finishing order."""
    m = np.where(mask, u, -np.inf).max(axis=1, keepdims=True)
    v = np.where(mask, np.exp(np.where(mask, u, 0.0) - m), 0.0)
    S = np.cumsum(v[:, ::-1], axis=1)[:, ::-1]                  # sum over drivers not yet placed
    S = np.where(mask, S, 1.0)
    ll = np.where(mask, u - m - np.log(S), 0.0).sum(axis=1)
    inv = np.cumsum(np.where(mask, 1.0 / S, 0.0), axis=1)
    grad = np.where(mask, 1.0 - v * inv, 0.0)
    return ll, grad


def _driver_links(ev: Events, p: dict):
    if p.get("driver_link") is None:
        return np.array([], int), np.array([], int), np.array([])
    idx = {k: i for i, k in enumerate(ev.drivers)}
    a, b, lam = [], [], []
    young_link = p.get("driver_link_young") or p["driver_link"]
    for k, i in idx.items():
        name, s = k.rsplit(":", 1)
        prev = f"{name}:{int(s) - 1}"
        if prev in idx:
            a.append(i)
            b.append(idx[prev])
            lam.append(young_link if k in ev.young else p["driver_link"])
    return np.array(a, int), np.array(b, int), np.array(lam, float)


def fit(ev: Events, params: dict | None = None, init: Ratings | None = None) -> Ratings:
    p = {**DEFAULT, **(params or {})}
    D, C = len(ev.drivers), len(ev.cars)
    la, lb, lnew = _links(ev)
    lam_link = np.where(lnew, p["link_new_rules"], p["link"])
    da, db, lam_d = _driver_links(ev, p)
    # random walk over seasons: the ridge anchors only each driver's first season in the window
    rd_mask = np.ones(D)
    if len(da):
        rd_mask[da] = 0.0
    # theta = [d (D), c (C), log kappa_quali, log kappa_sprint, gamma_race, gamma_sprint]
    x0 = np.zeros(D + C + 4)
    x0[D + C:] = [np.log(2.0), 0.0, 0.3, 0.3]
    if init is not None:
        dmap = {x: v for x, v in zip(init.drivers, init.d)}
        cmap = {x: v for x, v in zip(init.cars, init.c)}
        x0[:D] = [dmap.get(x, 0.0) for x in ev.drivers]
        x0[D:D + C] = [cmap.get(x, 0.0) for x in ev.cars]
        x0[D + C:] = [np.log(init.kappa[1]), np.log(init.kappa[2]), init.gamma[0], init.gamma[2]]
    drv = np.where(ev.mask, ev.drv, 0)
    car = np.where(ev.mask, ev.car, 0)
    is_q, is_s, is_r = ev.typ == 1, ev.typ == 2, ev.typ == 0

    def obj(x):
        d, c = x[:D], x[D:D + C]
        lkq, lks, gr, gs = x[D + C:]
        kap = np.where(is_q, np.exp(lkq), np.where(is_s, np.exp(lks), 1.0))[:, None]
        gam = np.where(is_r, gr, np.where(is_s, gs, 0.0))[:, None]
        base = d[drv] + c[car]
        u = kap * base + gam * ev.g
        ll, gu = pl_loglik(u, ev.mask)
        gu = gu * ev.w[:, None]
        f = -(ev.w * ll).sum()
        gd = -np.bincount(drv[ev.mask], (gu * kap)[ev.mask], D)
        gc = -np.bincount(car[ev.mask], (gu * kap)[ev.mask], C)
        gk = -(gu * base).sum(axis=1)
        g_lkq = (gk[is_q]).sum() * np.exp(lkq)
        g_lks = (gk[is_s]).sum() * np.exp(lks)
        g_gr = -(gu * ev.g).sum(axis=1)[is_r].sum()
        g_gs = -(gu * ev.g).sum(axis=1)[is_s].sum()
        # priors
        f += 0.5 * p["ridge_driver"] * (rd_mask * d ** 2).sum() + 0.5 * p["ridge_car"] * (c ** 2).sum()
        gd += p["ridge_driver"] * rd_mask * d
        gc += p["ridge_car"] * c
        if len(da):
            dd = d[da] - d[db]
            f += 0.5 * (lam_d * dd ** 2).sum()
            np.add.at(gd, da, lam_d * dd)
            np.add.at(gd, db, -lam_d * dd)
        if len(la):
            diff = c[la] - c[lb]
            f += 0.5 * (lam_link * diff ** 2).sum()
            np.add.at(gc, la, lam_link * diff)
            np.add.at(gc, lb, -lam_link * diff)
        f += 0.5 * ((lkq - np.log(2.0)) ** 2 + lks ** 2) + 0.5 * 0.1 * (gr ** 2 + gs ** 2)
        g_lkq += lkq - np.log(2.0)
        g_lks += lks
        g_gr += 0.1 * gr
        g_gs += 0.1 * gs
        return f, np.concatenate([gd, gc, [g_lkq, g_lks, g_gr, g_gs]])

    r = minimize(obj, x0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    x = r.x
    return Ratings(ev.drivers, x[:D], ev.cars, x[D:D + C],
                   np.array([1.0, np.exp(x[D + C]), np.exp(x[D + C + 1])]),
                   np.array([x[D + C + 2], 0.0, x[D + C + 3]]), p, n_events=len(ev.w))
