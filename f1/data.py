"""F1DB (github.com/f1db/f1db, CC BY 4.0) -> flat tables, cached as CSV in data/.

Tables
  races      one row per Grand Prix (past and scheduled): season, round, date, start time (UTC), circuit,
             circuit type (RACE / STREET / ROAD), sprint weekend flag
  results    race classification: one row per entry. status = FIN (classified, pos 1..n), DNF, NC (running
             but not classified), DSQ, EX, DNS, DNQ, DNPQ, DNP. grid = starting slot (pit-lane start = last)
  quali      qualifying classification (pos) and best lap in ms
  sprint     sprint classification, same columns as results
  standings  drivers' championship after each race (pos, points)

The race "starters" are FIN, DNF, NC, DSQ and EX entries. DNF and NC count as did-not-finish for the
reliability model; DSQ/EX are dropped from the finishing order (post-race decisions, not pace).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data"
STARTED = ("FIN", "DNF", "NC", "DSQ", "EX")
TABLES = ("races", "results", "quali", "sprint", "standings", "practice")
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _yaml(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_LOADER)


def lap_ms(t) -> float:
    """'1:16.514' or '76.514' -> 76514.0 ms; blanks -> NaN."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return np.nan
    s = str(t).strip()
    if not s or not re.match(r"^[\d:.]+$", s):
        return np.nan
    parts = s.split(":")
    try:
        sec = float(parts[-1]) + (60 * float(parts[-2]) if len(parts) >= 2 else 0) + (3600 * float(parts[-3]) if len(parts) == 3 else 0)
    except ValueError:
        return np.nan
    return sec * 1000.0


def clock(v):
    """YAML 1.1 reads an unquoted 13:00 as the base-60 integer 780; turn either form back into 'HH:MM'."""
    if v is None or v == "":
        return None
    if isinstance(v, int):
        return f"{v // 60:02d}:{v % 60:02d}"
    return str(v)[:5]


def _status(pos) -> tuple[float, str]:
    if isinstance(pos, int):
        return float(pos), "FIN"
    return np.nan, str(pos)


def _grid(g, n_starters: int) -> float:
    if isinstance(g, int):
        return float(g)
    if g == "PL":
        return float(n_starters)          # pit-lane start: behind the whole grid
    return np.nan


def _classification(rows, race_id, season, rnd) -> list[dict]:
    rows = rows or []
    n = sum(1 for r in rows if _status(r.get("position"))[1] in STARTED)
    out = []
    for r in rows:
        pos, st = _status(r.get("position"))
        out.append({"race_id": race_id, "season": season, "round": rnd, "driver": r["driverId"],
                    "constructor": r["constructorId"], "engine": r.get("engineManufacturerId"),
                    "number": r.get("driverNumber"), "pos": pos, "status": st,
                    "grid": _grid(r.get("gridPosition"), n), "laps": r.get("laps"),
                    "points": float(r.get("points") or 0.0), "reason": r.get("reasonRetired"),
                    "shared": bool(r.get("sharedCar", False))})
    return out


def read_f1db(src: str | Path, seasons=None) -> dict[str, pd.DataFrame]:
    """Read F1DB's src/data folder (YAML). seasons: iterable of years, or None for all."""
    src = Path(src)
    circuits = {}
    for f in (src / "circuits").glob("*.yml"):
        c = _yaml(f)
        circuits[c["id"]] = c.get("type")
    races, results, quali, sprint, standings, practice = [], [], [], [], [], []
    years = sorted(int(p.name) for p in (src / "seasons").iterdir() if p.name.isdigit())
    if seasons is not None:
        years = [y for y in years if y in set(seasons)]
    for y in years:
        for folder in sorted((src / "seasons" / str(y) / "races").iterdir()):
            r = _yaml(folder / "race.yml")
            if not r:
                continue
            rid, rnd = int(r["id"]), int(r["round"])
            races.append({"race_id": rid, "season": y, "round": rnd, "date": str(r["date"]),
                          "time_utc": clock(r.get("time")), "gp": r.get("grandPrixId"), "name": r.get("officialName"),
                          "circuit": r.get("circuitId"),
                          "circuit_type": r.get("circuitType") or circuits.get(r.get("circuitId")),
                          "laps": r.get("laps"), "qualifying_date": str(r.get("qualifyingDate") or ""),
                          "qualifying_time": clock(r.get("qualifyingTime")),
                          "sprint": bool(r.get("sprintRaceDate")) or (folder / "sprint-race-results.yml").exists(),
                          "folder": folder.name})
            results += _classification(_yaml(folder / "race-results.yml"), rid, y, rnd)
            sprint += _classification(_yaml(folder / "sprint-race-results.yml"), rid, y, rnd)
            for q in _yaml(folder / "qualifying-results.yml") or []:
                times = [lap_ms(q.get(k)) for k in ("q1", "q2", "q3", "time")]
                pos = q.get("position")
                quali.append({"race_id": rid, "season": y, "round": rnd, "driver": q["driverId"],
                              "constructor": q["constructorId"],
                              "pos": float(pos) if isinstance(pos, int) else np.nan,
                              "best_ms": np.nanmin(times) if np.isfinite(times).any() else np.nan})
            for k in (1, 2, 3, 4):
                for q in _yaml(folder / f"free-practice-{k}-results.yml") or []:
                    pos = q.get("position")
                    practice.append({"race_id": rid, "season": y, "round": rnd, "session": k,
                                     "driver": q["driverId"], "constructor": q["constructorId"],
                                     "pos": float(pos) if isinstance(pos, int) else np.nan,
                                     "best_ms": lap_ms(q.get("time"))})
            for s in _yaml(folder / "driver-standings.yml") or []:
                standings.append({"race_id": rid, "season": y, "round": rnd, "driver": s["driverId"],
                                  "pos": s.get("position"), "points": float(s.get("points") or 0.0)})
    t = {"races": pd.DataFrame(races), "results": pd.DataFrame(results), "quali": pd.DataFrame(quali),
         "sprint": pd.DataFrame(sprint), "standings": pd.DataFrame(standings),
         "practice": pd.DataFrame(practice, columns=["race_id", "season", "round", "session", "driver",
                                                     "constructor", "pos", "best_ms"])}
    t["races"]["date"] = pd.to_datetime(t["races"]["date"])
    return t


def save(tables: dict, folder: Path = CACHE) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        v.to_csv(folder / f"{k}.csv", index=False)


def load(folder: Path = CACHE) -> dict[str, pd.DataFrame]:
    t = {k: pd.read_csv(folder / f"{k}.csv") for k in TABLES}
    t["races"]["date"] = pd.to_datetime(t["races"]["date"])
    return t


def merge_update(old: dict, new: dict, seasons) -> dict:
    """Replace the given seasons in cached tables with freshly read ones."""
    seasons = set(seasons)
    out = {}
    for k in old:
        keep = old[k][~old[k]["season"].isin(seasons)] if len(old[k]) else old[k]
        out[k] = pd.concat([keep, new[k]], ignore_index=True).sort_values(
            [c for c in ("season", "round") if c in keep.columns], kind="stable").reset_index(drop=True)
    out["races"]["date"] = pd.to_datetime(out["races"]["date"])
    return out


def race_start_utc(races: pd.DataFrame) -> pd.Series:
    t = races["time_utc"].fillna("12:00").astype(str).str.slice(0, 5)
    return pd.to_datetime(races["date"].dt.strftime("%Y-%m-%d") + " " + t, utc=True)


def validate(t: dict) -> list[str]:
    issues = []
    r = t["results"]
    fin = r[r["status"] == "FIN"]
    dup = fin.duplicated(["race_id", "pos"]).sum()
    if dup:
        issues.append(f"{dup} duplicated finishing positions (shared drives in the 1950s are expected)")
    done = r.groupby("race_id")["status"].apply(lambda s: (s == "FIN").sum())
    if (done == 0).any():
        issues.append(f"{int((done == 0).sum())} races with no classified finisher")
    return issues
