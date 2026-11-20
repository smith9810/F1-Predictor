"""Command line: python -m f1 <command>

  update   --f1db PATH   read F1DB's src/data folder into data/*.csv
  predict  [--stage auto|pre|post] [--sims N] [--force]   forecast the next race, log it, write dashboard files
  score                  score logged forecasts whose race has a result
  backtest [--first 1996 --test-from 2013]   walk-forward study + title-decided analysis -> out/dashboard
  report                 rebuild dashboard files and calibration from the saved backtest
  status                 next race, stage available, last log rows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from f1 import backtest as bt, calibrate, data, pipeline, report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "dashboard"


def _tables():
    return data.load(ROOT / "data")


def cmd_update(a):
    t = data.read_f1db(a.f1db)
    issues = [i for i in data.validate(t) if "shared drives" not in i]
    for i in issues:
        print("WARNING:", i, file=sys.stderr)
    data.save(t, ROOT / "data")
    r = t["results"]
    print(f"F1DB read: {len(t['races'])} races, results to {r['season'].max()} round "
          f"{r.loc[r['season'] == r['season'].max(), 'round'].max()}")


def write_live_files(t, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    if rows is not None and len(rows):
        cols = ["gp", "season", "round", "race_start_utc", "stage", "logged_at_utc", "driver", "constructor", "grid",
                "p_win", "p_podium", "p_top6", "p_points", "p_dnf", "exp_pos", "exp_points", "fair_win_odds",
                "fair_podium_odds"]
        nr = rows[cols].sort_values("p_win", ascending=False)
        nr.round(4).to_csv(OUT / "next_race.csv", index=False)
        rt = rows[["driver", "constructor", "driver_skill", "car_pace", "pace"]].sort_values("pace", ascending=False)
        rt.insert(0, "as_of_utc", rows["logged_at_utc"].iloc[0])
        rt.round(4).to_csv(OUT / "ratings.csv", index=False)
    if pipeline.SCORES.exists():
        pd.read_csv(pipeline.SCORES).round(5).to_csv(OUT / "live_scores.csv", index=False)


def cmd_predict(a):
    t = _tables()
    try:
        msg, rows = pipeline.predict(t, stage=a.stage, n_sims=a.sims, force=a.force)
        write_live_files(t, rows)
        pipeline.heartbeat("predict", "ok", msg)
    except Exception as exc:
        pipeline.heartbeat("predict", "error", repr(exc))
        raise
    print(msg)


def cmd_score(a):
    t = _tables()
    new = pipeline.score(t)
    write_live_files(t, None)
    pipeline.heartbeat("score", "ok", f"{len(new)} new score rows")
    print(new.round(4).to_string(index=False) if len(new) else "nothing new to score")


def cmd_backtest(a):
    t = _tables()
    reg = pipeline.load_registry()
    last = int(t["results"]["season"].max())
    res, fc = bt.run(t, a.first, last, reg["champion"], n_sims=a.sims, keep_forecasts=True)
    res.to_csv(ROOT / "out" / "backtest_races.csv", index=False)
    fc.round(5).to_csv(ROOT / "out" / "backtest_forecasts.csv.gz", index=False, compression="gzip")
    _report(t, res, fc, reg, a.test_from, last)


def cmd_report(a):
    """Rebuild the dashboard files and calibration from the saved backtest (no refit)."""
    t = _tables()
    res = pd.read_csv(ROOT / "out" / "backtest_races.csv")
    res = res[res["method"] != "model_cal"]
    fc = pd.read_csv(ROOT / "out" / "backtest_forecasts.csv.gz")
    _report(t, res, fc, pipeline.load_registry(), a.test_from, int(fc["season"].max()))


def _report(t, res, fc, reg, test_from, last):
    out = report.write_backtest(res, fc, t, OUT, test_from)
    f = calibrate.outcomes(fc)
    recent = f[f["season"] >= last - 5]
    reg["calibration"] = {st: calibrate.fit_all(recent[recent["stage"] == st]) for st in ("post", "pre")}
    reg.setdefault("history", []).append({"date": pd.Timestamp.now().strftime("%Y-%m-%d"), "event": "backtest",
                                          "calibration_gate": out["calibration_gate"].round(5).to_dict("records")})
    pipeline.REGISTRY.write_text(json.dumps(reg, indent=1, default=str))
    print(out["methods"].round(4).to_string(index=False))
    print(out["gaps"].round(5).to_string(index=False))
    print(out["title"].round(4).to_string(index=False))
    print(out["calibration_gate"].round(4).to_string(index=False))
    pipeline.heartbeat("backtest", "ok", f"{res['race_id'].nunique()} races")


def cmd_status(a):
    t = _tables()
    now = pd.Timestamp.now(tz="UTC")
    r = pipeline.next_race(t, now)
    if r is None:
        print("no scheduled race")
        return
    print(f"next: {r['gp']} {r['season']} round {r['round']} starts {r['start_utc']} UTC;"
          f" stage available: {pipeline.stage_available(t, r)}")
    lg = pipeline.logged()
    if len(lg):
        print(lg.groupby(["race_id", "stage"])["logged_at_utc"].first().tail(6).to_string())


def main(argv=None):
    p = argparse.ArgumentParser(prog="f1")
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update")
    u.add_argument("--f1db", required=True, help="path to F1DB src/data")
    pr = sub.add_parser("predict")
    pr.add_argument("--stage", default="auto", choices=["auto", "pre", "post"])
    pr.add_argument("--sims", type=int, default=20000)
    pr.add_argument("--force", action="store_true")
    sub.add_parser("score")
    b = sub.add_parser("backtest")
    b.add_argument("--first", type=int, default=1996)
    b.add_argument("--test-from", type=int, default=2013)
    b.add_argument("--sims", type=int, default=4000)
    rp = sub.add_parser("report")
    rp.add_argument("--test-from", type=int, default=2013)
    sub.add_parser("status")
    a = p.parse_args(argv)
    {"update": cmd_update, "predict": cmd_predict, "score": cmd_score, "backtest": cmd_backtest,
     "report": cmd_report, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
