"""Dashboard datasets (out/dashboard/*.csv) from the walk-forward study and the live log."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from f1 import backtest as bt, calibrate as cal, title as title_mod

LABELS = {("post", "model_cal"): "Model, after qualifying (calibrated, live version)",
          ("pre", "model_cal"): "Model, before qualifying (calibrated, live version)",
          ("post", "model"): "Model, after qualifying (raw)", ("post", "grid"): "Starting grid only",
          ("pre", "model"): "Model, before qualifying (raw)", ("pre", "form"): "Recent qualifying form only",
          ("pre", "uniform"): "Every driver equal", ("post", "uniform"): "Every driver equal"}


def methods(res: pd.DataFrame, first_test: int) -> pd.DataFrame:
    r = res[res["season"] >= first_test]
    s = bt.summarise(r)
    s["label"] = [LABELS.get((a, b), b) for a, b in zip(s["stage"], s["method"])]
    s = s[~((s["method"] == "uniform") & (s["stage"] == "pre"))]
    # paired gap to the matching baseline, bootstrap over races
    out = []
    for st, base in (("post", "grid"), ("pre", "form")):
        m = r[(r["stage"] == st) & (r["method"] == "model")].set_index("race_id")["pos_rps"]
        b = r[(r["stage"] == st) & (r["method"] == base)].set_index("race_id")["pos_rps"]
        j = m.index.intersection(b.index)
        d = (m[j] - b[j]).to_numpy()
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
        out.append({"stage": st, "baseline": base, "gap_pos_rps": d.mean(), "gap_ci_low": np.percentile(bs, 2.5),
                    "gap_ci_high": np.percentile(bs, 97.5), "races": len(d),
                    "model_better_share": float((d < 0).mean())})
    return s, pd.DataFrame(out)


def seasons(res: pd.DataFrame) -> pd.DataFrame:
    keep = res[res["method"].isin(["model", "grid", "form"])]
    p = keep.pivot_table(index="season", columns=["stage", "method"], values="pos_rps", aggfunc="mean")
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    return p.reset_index()


def calibration(fc: pd.DataFrame, first_test: int, stage: str = "post") -> pd.DataFrame:
    f = fc[(fc["stage"] == stage) & (fc["season"] >= first_test)].copy()
    f["won"] = ((f["status"] == "FIN") & (f["pos"] == 1)).astype(float)
    f["podium"] = ((f["status"] == "FIN") & (f["pos"] <= 3)).astype(float)
    rows = []
    for col, y in (("p_win", "won"), ("p_podium", "podium")):
        edges = [0, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
        f["bin"] = pd.cut(f[col], edges, right=False)
        g = f.groupby("bin", observed=True).agg(predicted=(col, "mean"), observed=(y, "mean"), n=(y, "size"))
        g = g[g["n"] >= 5].reset_index()
        g["market"] = "win" if col == "p_win" else "podium"
        g["bin"] = g["bin"].astype(str)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def calibrated_scores(fc: pd.DataFrame, first_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward recalibrated forecasts and their per-race scores (method 'model_cal')."""
    wf = cal.outcomes(cal.walk_forward(fc, first_test))
    rows = []
    for (rid, st), g in wf.groupby(["race_id", "stage"]):
        w = g["won"].to_numpy()
        rows.append({"race_id": rid, "season": int(g["season"].iloc[0]), "round": int(g["round"].iloc[0]),
                     "gp": g["gp"].iloc[0], "stage": st, "method": "model_cal",
                     "win_ll": float(-np.log(max(g["p_win"].to_numpy()[w.argmax()], 1e-4))) if w.sum() == 1 else np.nan,
                     "win_brier": float(((g["p_win"] - w) ** 2).sum()),
                     "podium_brier": float(((g["p_podium"] - g["podium"]) ** 2).mean()),
                     "points_brier": float(((g["p_points"] - g["points_finish"]) ** 2).mean())})
    return wf, pd.DataFrame(rows)


def calibration_gate(res: pd.DataFrame, cal_rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    rng = np.random.default_rng(0)
    for st in ("post", "pre"):
        raw = res[(res["method"] == "model") & (res["stage"] == st)].set_index("race_id")
        c = cal_rows[cal_rows["stage"] == st].set_index("race_id")
        j = raw.index.intersection(c.index)
        for k in ("win_ll", "win_brier", "podium_brier", "points_brier"):
            d = (c.loc[j, k] - raw.loc[j, k]).dropna().to_numpy()
            bs = np.array([rng.choice(d, len(d)).mean() for _ in range(4000)])
            out.append({"stage": st, "score": k, "raw": raw.loc[j, k].mean(), "calibrated": c.loc[j, k].mean(),
                        "difference": d.mean(), "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
                        "p": 2 * min((bs <= 0).mean(), (bs >= 0).mean()), "races": len(d)})
    return pd.DataFrame(out)


def write_backtest(res: pd.DataFrame, fc: pd.DataFrame, t: dict, out: Path, first_test: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    wf, cal_rows = calibrated_scores(fc, first_test)
    gate = calibration_gate(res, cal_rows)
    gate.round(5).to_csv(out / "calibration_gate.csv", index=False)
    res = pd.concat([res, cal_rows], ignore_index=True)
    m, gaps = methods(res, first_test)
    m.round(5).to_csv(out / "methods.csv", index=False)
    gaps.round(5).to_csv(out / "model_vs_baseline.csv", index=False)
    seasons(res).round(5).to_csv(out / "seasons.csv", index=False)
    c_raw = calibration(fc, first_test).assign(version="raw simulation")
    c_cal = calibration(wf, first_test).assign(version="calibrated (live)")
    pd.concat([c_cal, c_raw], ignore_index=True).round(4).to_csv(out / "calibration.csv", index=False)
    clinch = title_mod.clinch_rounds(t)
    clinch.to_csv(out / "title_clinch.csv", index=False)
    rows = []
    for st in ("post", "pre"):
        a = title_mod.analyse(res, fc, clinch, st)
        a.insert(0, "stage", st)
        rows.append(a)
    te = pd.concat(rows, ignore_index=True)
    te.round(5).to_csv(out / "title_effect.csv", index=False)
    return {"methods": m, "gaps": gaps, "title": te, "calibration_gate": gate}
