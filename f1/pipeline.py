"""Live forecasting: next race, entry list, forecast, append-only log, scoring.

A forecast is logged once per race and stage ("pre" = before qualifying, "post" = after qualifying,
using qualifying order as the grid). Rows are refused once the race has started, so the log can be trusted
as a record of what the model said in advance. Each forecast also saves the full finishing-place
distribution (logs/dists/) so it can be scored with the ranked probability score afterwards.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from f1 import backtest as bt, calibrate, data, dnf as dnf_mod, model, simulate as sim

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "predictions.csv"
SCORES = ROOT / "logs" / "scores.csv"
DISTS = ROOT / "logs" / "dists"
RUNS = ROOT / "logs" / "runs.csv"
REGISTRY = ROOT / "models" / "registry.json"
MODEL_VERSION = "pl-v1-cal"


def load_registry(path: Path = REGISTRY) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"champion": dict(model.DEFAULT), "history": []}


def next_race(t: dict, now) -> pd.Series | None:
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    r = t["races"].copy()
    r["start_utc"] = data.race_start_utc(r)
    done = set(t["results"]["race_id"])
    up = r[(r["start_utc"] > now) & ~r["race_id"].isin(done)].sort_values("start_utc")
    return None if up.empty else up.iloc[0]


def entries_for(t: dict, race: pd.Series, stage: str) -> pd.DataFrame:
    """post: qualifying classification (grid = qualifying place; penalties not applied).
    pre: latest practice session of the weekend with 18+ drivers, else the last race's starters."""
    rid = int(race["race_id"])
    if stage == "post":
        q = t["quali"][t["quali"]["race_id"] == rid].copy()
        q = q.sort_values("pos", na_position="last")
        q["grid"] = np.arange(1, len(q) + 1, dtype=float)
        return q[["driver", "constructor", "grid"]].reset_index(drop=True)
    pr = t["practice"][t["practice"]["race_id"] == rid]
    for s in (3, 2, 1, 4):
        g = pr[pr["session"] == s]
        if len(g) >= 18:
            e = g[["driver", "constructor"]].drop_duplicates("driver")
            return e.assign(grid=np.nan).reset_index(drop=True)
    res = t["results"]
    last = res[res["season"] == race["season"]]
    if last.empty:
        last = res[res["season"] == res["season"].max()]
    last = last[last["race_id"] == last["race_id"].max()]
    e = last[last["status"].isin(bt.STARTERS)][["driver", "constructor"]]
    return e.assign(grid=np.nan).reset_index(drop=True)


def stage_available(t: dict, race: pd.Series) -> str:
    q = t["quali"][t["quali"]["race_id"] == int(race["race_id"])]
    return "post" if q["pos"].notna().sum() >= 10 else "pre"


def forecast(t: dict, race: pd.Series, stage: str, params: dict, now, n_sims: int = 20000, seed: int = 7):
    entries = entries_for(t, race, stage)
    as_of = pd.Timestamp(now).tz_localize(None) if pd.Timestamp(now).tzinfo else pd.Timestamp(now)
    ev = model.build_events(t, as_of, params)          # every session already run (incl. this qualifying)
    R = model.fit(ev, params)
    season = int(race["season"])
    pace = R.pace(entries["driver"], entries["constructor"], season)
    done = t["results"]["race_id"]
    last_done = int(t["races"][t["races"]["race_id"].isin(done)].sort_values("date")["race_id"].iloc[-1])
    feats = dnf_mod.state_for(t, last_done, entries[["driver", "constructor"]].assign(grid=entries["grid"]),
                              race["circuit"])
    f_all = dnf_mod.features(t)
    tr = f_all[(f_all["season"] < season) & (f_all["season"] >= season - 10)]   # as in the backtest
    m = dnf_mod.fit(tr, stage == "post")
    feats = feats.set_index("driver").reindex(entries["driver"]).reset_index()
    feats[dnf_mod.COLS_PRE] = feats[dnf_mod.COLS_PRE].fillna(feats[dnf_mod.COLS_PRE].median())
    feats["grid"] = entries["grid"].to_numpy(float)
    p_dnf = dnf_mod.predict(m, feats, stage == "post")
    grid = entries["grid"].to_numpy(float) if stage == "post" else None
    pos, dn = sim.simulate(pace, p_dnf, R.kappa[1], R.gamma[0], grid, n_sims, seed)
    out = sim.summary(pos, dn)
    out.insert(0, "driver", entries["driver"].to_numpy())
    out.insert(1, "constructor", entries["constructor"].to_numpy())
    out.insert(2, "grid", entries["grid"].to_numpy(float))
    cal = (load_registry().get("calibration") or {}).get(stage)
    if cal:                                    # champion = simulation + walk-forward recalibration
        out = calibrate.apply(out, cal)
    out["pace"] = pace
    out["driver_skill"] = [R.driver_skill(x, season) for x in entries["driver"]]
    out["car_pace"] = [R.car_pace(c, season) for c in entries["constructor"]]
    out["fair_win_odds"] = np.where(out["p_win"] > 0, 1.0 / out["p_win"].clip(lower=1e-6), np.nan)
    out["fair_podium_odds"] = np.where(out["p_podium"] > 0, 1.0 / out["p_podium"].clip(lower=1e-6), np.nan)
    return out, sim.position_dist(pos), R


def append_log(rows: pd.DataFrame, path: Path = LOG) -> int:
    """Append-only. New columns are added to the file (earlier rows get blanks); old values never change."""
    late = pd.to_datetime(rows["logged_at_utc"], utc=True) >= pd.to_datetime(rows["race_start_utc"], utc=True)
    if late.any():
        raise ValueError("refusing to log a forecast made after the race started")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        extra = [c for c in rows.columns if c not in old.columns]
        if extra:
            pd.concat([old, rows], ignore_index=True)[list(old.columns) + extra].to_csv(path, index=False)
            return len(rows)
        rows = rows.reindex(columns=old.columns)
    rows.to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(rows)


def logged(path: Path = LOG) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["race_id", "stage"])


def predict(t: dict, now=None, stage: str = "auto", n_sims: int = 20000, force: bool = False,
            registry: dict | None = None):
    """Forecast the next race. Returns (status message, forecast table or None)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    race = next_race(t, now)
    if race is None:
        return "no scheduled race", None
    if (race["start_utc"] - now) > pd.Timedelta(days=10):
        return f"next race {race['gp']} {race['season']} is more than 10 days away", None
    st = stage_available(t, race) if stage == "auto" else stage
    reg = registry or load_registry()
    fc, D, R = forecast(t, race, st, reg["champion"], now.tz_localize(None), n_sims)
    meta = {"logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "race_id": int(race["race_id"]),
            "season": int(race["season"]), "round": int(race["round"]), "gp": race["gp"],
            "race_start_utc": race["start_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"), "stage": st,
            "model_version": MODEL_VERSION}
    rows = pd.concat([pd.DataFrame([meta] * len(fc)), fc.reset_index(drop=True)], axis=1)
    lg = logged(LOG)
    already = ((lg["race_id"] == meta["race_id"]) & (lg["stage"] == st)
               & (lg["model_version"] == MODEL_VERSION)).any() if len(lg) else False
    msg = f"{race['gp']} {race['season']} ({st})"
    if already and not force:
        msg += ": already logged; dashboard files refreshed only"
    else:
        append_log(rows, LOG)
        DISTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(D, index=fc["driver"], columns=[f"P{k + 1}" for k in range(D.shape[1])]).to_csv(
            DISTS / f"{meta['race_id']}_{st}_{MODEL_VERSION}.csv")
        msg += f": logged {len(rows)} rows"
    return msg, rows


def score(t: dict, path: Path = LOG, out: Path = SCORES) -> pd.DataFrame:
    """Score every logged forecast whose race now has a result (model and baselines on the same race)."""
    lg = logged(path)
    if lg.empty:
        return pd.DataFrame()
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["race_id", "stage", "model_version"])
    if "model_version" not in lg:
        lg["model_version"] = "pl-v1"
    new = []
    for (rid, st, ver), g in lg.groupby(["race_id", "stage", "model_version"]):
        if ((have["race_id"] == rid) & (have["stage"] == st) & (have.get("model_version") == ver)).any():
            continue
        s = bt.starters(t, int(rid))
        if s.empty or not (s["status"] == "FIN").any():
            continue
        g = g.drop_duplicates("driver", keep="first").set_index("driver")
        s = s[s["driver"].isin(g.index)].reset_index(drop=True)
        s["actual"] = np.arange(1, len(s) + 1)
        f = DISTS / f"{rid}_{st}_{ver}.csv"
        if not f.exists():
            f = DISTS / f"{rid}_{st}.csv"
        N = len(s)
        if f.exists():
            D = pd.read_csv(f, index_col=0).reindex(s["driver"]).to_numpy(float)[:, :N]
            D = D / D.sum(axis=1, keepdims=True)
        else:
            D = np.full((N, N), 1.0 / N)
        probs = g.loc[s["driver"], ["p_win", "p_podium", "p_points"]].reset_index(drop=True)
        race = t["races"][t["races"]["race_id"] == rid].iloc[0]
        base = {"race_id": int(rid), "season": int(race["season"]), "round": int(race["round"]), "gp": race["gp"],
                "stage": st, "model_version": ver,
                "scored_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
        new.append({**base, "method": "model", **bt.score_race(D, probs, s)})
        M = bt.grid_table(t, int(race["season"]))
        if st == "post":
            Dg, pg = bt.table_forecast(M, s["grid"].fillna(N).to_numpy(), N)
            new.append({**base, "method": "grid", **bt.score_race(Dg, pg, s)})
        else:
            Df, pf = bt.table_forecast(M, bt.form_grid(t, int(rid), s["driver"]), N)
            new.append({**base, "method": "form", **bt.score_race(Df, pf, s)})
    if not new:
        return pd.DataFrame()
    df = pd.DataFrame(new)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    return df


def heartbeat(job: str, status: str, detail: str, now=None, path: Path = RUNS) -> None:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    row = pd.DataFrame([{"run_at_utc": now.strftime("%Y-%m-%dT%H:%MZ"), "job": job, "status": status,
                         "detail": str(detail)[:200]}])
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
