"""When was the drivers' title decided, and are races after that less predictable?

Clinch rule: after a race, the leader's points minus the runner-up's exceed the most points anyone could
still score (remaining races and sprints at that season's points system). Seasons from 1991 only: before
that, dropped scores make the arithmetic unreliable. A tie at best counts as decided only if the
leader would still win on countback (more wins than the runner-up could reach).

Tests on walk-forward forecasts (`backtest.run(..., keep_forecasts=True)`):
  1. score gap  - model pos_rps / win log loss in post-clinch races minus late-season pre-clinch races
                  (races in the final third of a season), with a bootstrap interval over races
  2. temperature - the factor tau that best rescales the model's race strengths (tau < 1 = results more
                  random than the model expects), fitted separately to post-clinch and comparison races
  3. champion   - the new champion's finishing place minus the model's expected place, and DNF rate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def race_max(season: int, last_race: bool = False) -> float:
    if season <= 2009:
        return 10.0
    if season <= 2018:
        return 50.0 if (season == 2014 and last_race) else 25.0
    if season <= 2024:
        return 26.0                                   # fastest-lap point (top 10 only)
    return 25.0


def sprint_max(season: int) -> float:
    return 0.0 if season < 2021 else 3.0 if season == 2021 else 8.0


def clinch_rounds(t: dict, first: int = 1991) -> pd.DataFrame:
    """One row per season: final round, round at which the title was mathematically decided (or NaN),
    champion, and races left after the clinch."""
    races = t["races"].sort_values(["season", "round"])
    st = t["standings"]
    out = []
    for y, rs in races[races["season"] >= first].groupby("season"):
        rs = rs.reset_index(drop=True)
        n = len(rs)
        maxes = np.array([race_max(y, i == n - 1) + (sprint_max(y) if s else 0.0)
                          for i, s in enumerate(rs["sprint"].astype(bool))])
        remaining = maxes[::-1].cumsum()[::-1]       # remaining[i] = points available from race i on
        clinch, champ = np.nan, None
        for i, r in rs.iterrows():
            s = st[st["race_id"] == r["race_id"]].sort_values("points", ascending=False)
            if len(s) < 2:
                break                                 # season still running: no standings yet
            left = remaining[i + 1] if i + 1 < n else 0.0
            gap = s["points"].iloc[0] - s["points"].iloc[1]
            tie_won = False
            if abs(gap - left) < 1e-9:                # level at best: countback on wins decides
                done = rs.loc[:i, "race_id"]
                w = t["results"][t["results"]["race_id"].isin(done) & (t["results"]["pos"] == 1)
                                 & (t["results"]["status"] == "FIN")]["driver"].value_counts()
                tie_won = w.get(s["driver"].iloc[0], 0) > w.get(s["driver"].iloc[1], 0) + (n - 1 - i)
            if gap > left or tie_won:
                clinch, champ = int(r["round"]), s["driver"].iloc[0]
                break
        out.append({"season": int(y), "rounds": n, "clinch_round": clinch, "champion": champ,
                    "races_after": (n - clinch) if np.isfinite(clinch) else 0})
    return pd.DataFrame(out)


def label_races(res: pd.DataFrame, clinch: pd.DataFrame) -> pd.DataFrame:
    """Adds post_clinch and late_season flags to per-race rows (needs season, round)."""
    c = clinch.set_index("season")
    r = res.copy()
    cr = r["season"].map(c["clinch_round"])
    n = r["season"].map(c["rounds"])
    r["post_clinch"] = (r["round"] > cr).fillna(False)
    r["late_season"] = r["round"] > (2 / 3) * n
    r["champion"] = r["season"].map(c["champion"])
    return r


def boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    bs = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d), float(lo), float(hi), float(p)


def pl_order_loglik(u: np.ndarray, order_mask: np.ndarray) -> float:
    """Log-likelihood of finishing orders (rows padded with -inf) for given strengths."""
    from f1.model import pl_loglik
    ll, _ = pl_loglik(u, order_mask)
    return float(ll.sum())


def strengths_matrix(fc: pd.DataFrame, races: pd.DataFrame, stage: str = "post"):
    """E x M matrix of the model's race strengths (pace + gamma * -log grid) in finishing order,
    classified finishers only, for the races in `races` (needs race_id and gamma)."""
    f = fc[(fc["stage"] == stage) & (fc["status"] == "FIN")]
    gam = races.set_index("race_id")["gamma"]
    rows = []
    for rid, g in f.groupby("race_id"):
        if rid not in gam.index:
            continue
        g = g.sort_values("pos")
        grid = g["grid"].to_numpy(float)
        grid = np.where(np.isfinite(grid) & (grid > 0), grid, len(g))
        u = g["pace"].to_numpy(float) + (gam[rid] * -np.log(grid) if stage == "post" else 0.0)
        rows.append(u)
    M = max(len(r) for r in rows)
    U = np.full((len(rows), M), 0.0)
    mask = np.zeros((len(rows), M), bool)
    for i, r in enumerate(rows):
        U[i, :len(r)], mask[i, :len(r)] = r, True
    return U, mask


def best_tau(U: np.ndarray, mask: np.ndarray) -> float:
    r = minimize_scalar(lambda tau: -pl_order_loglik(U * tau, mask), bounds=(0.05, 3.0), method="bounded")
    return float(r.x)


def tau_interval(U, mask, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    est = best_tau(U, mask)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(U), len(U))
        bs.append(best_tau(U[i], mask[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi), np.array(bs)


def analyse(res: pd.DataFrame, fc: pd.DataFrame, clinch: pd.DataFrame, stage: str = "post") -> pd.DataFrame:
    """Rows: metric, post-clinch value, comparison value, difference, 95% interval, p, races."""
    r = label_races(res[(res["method"] == "model") & (res["stage"] == stage)], clinch)
    post = r[r["post_clinch"]]
    comp = r[~r["post_clinch"] & r["late_season"]]
    out = []
    for m in ("pos_rps", "win_ll", "podium_brier"):
        d, lo, hi, p = boot_diff(post[m].dropna(), comp[m].dropna())
        out.append({"test": f"model {m}", "post_clinch": post[m].mean(), "comparison": comp[m].mean(),
                    "difference": d, "ci_low": lo, "ci_high": hi, "p": p, "races_post": len(post),
                    "races_comparison": len(comp)})
    # temperature
    Up, mp = strengths_matrix(fc, post, stage)
    Uc, mc = strengths_matrix(fc, comp, stage)
    tp, tpl, tph, bp = tau_interval(Up, mp)
    tc, tcl, tch, bc = tau_interval(Uc, mc, seed=1)
    diff = bp - bc
    out.append({"test": "strength scale tau (1 = model right, <1 = more random)", "post_clinch": tp,
                "comparison": tc, "difference": tp - tc, "ci_low": float(np.percentile(diff, 2.5)),
                "ci_high": float(np.percentile(diff, 97.5)), "p": float(2 * min((diff <= 0).mean(), (diff >= 0).mean())),
                "races_post": len(post), "races_comparison": len(comp)})
    # champion
    f = fc[fc["stage"] == stage].merge(r[["race_id", "post_clinch", "late_season", "champion"]], on="race_id")
    ch = f[f["driver"] == f["champion"]]
    chp = ch[ch["post_clinch"]]
    chc = ch[~ch["post_clinch"] & ch["late_season"]]
    resid_p = chp["actual"] - chp["exp_pos"]
    resid_c = chc["actual"] - chc["exp_pos"]
    d, lo, hi, p = boot_diff(resid_p, resid_c)
    out.append({"test": "champion: actual minus expected place", "post_clinch": resid_p.mean(),
                "comparison": resid_c.mean(), "difference": d, "ci_low": lo, "ci_high": hi, "p": p,
                "races_post": len(chp), "races_comparison": len(chc)})
    dn_p = chp["status"].isin(["DNF", "NC"]).astype(float)
    dn_c = chc["status"].isin(["DNF", "NC"]).astype(float)
    d, lo, hi, p = boot_diff(dn_p, dn_c)
    out.append({"test": "champion: DNF rate", "post_clinch": dn_p.mean(), "comparison": dn_c.mean(),
                "difference": d, "ci_low": lo, "ci_high": hi, "p": p, "races_post": len(chp),
                "races_comparison": len(chc)})
    allp = f[f["post_clinch"]]["status"].isin(["DNF", "NC"]).astype(float)
    allc = f[~f["post_clinch"] & f["late_season"]]["status"].isin(["DNF", "NC"]).astype(float)
    exp_p = f[f["post_clinch"]]["p_dnf"].mean()
    exp_c = f[~f["post_clinch"] & f["late_season"]]["p_dnf"].mean()
    d, lo, hi, p = boot_diff(allp - exp_p, allc - exp_c)
    out.append({"test": "whole field: DNF rate minus model's DNF rate", "post_clinch": allp.mean() - exp_p,
                "comparison": allc.mean() - exp_c, "difference": d, "ci_low": lo, "ci_high": hi, "p": p,
                "races_post": len(post), "races_comparison": len(comp)})
    return pd.DataFrame(out)
