import numpy as np
import pandas as pd

from f1 import data


def test_clock_handles_yaml_base60_integers():
    assert data.clock(780) == "13:00"
    assert data.clock(1260) == "21:00"
    assert data.clock("04:00") == "04:00"
    assert data.clock(None) is None


def test_lap_ms():
    assert data.lap_ms("1:16.514") == 76514.0
    assert data.lap_ms("58.2") == 58200.0
    assert np.isnan(data.lap_ms(None)) and np.isnan(data.lap_ms("DNF"))


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_read_small_f1db_folder(tmp_path):
    src = tmp_path / "src"
    _write(src / "circuits" / "x.yml", "id: x\ntype: STREET\n")
    race = src / "seasons" / "2030" / "races" / "01-x"
    _write(race / "race.yml", "id: 9001\nround: 1\ndate: 2030-03-10\ntime: 13:00\ngrandPrixId: x\n"
                              "circuitId: x\nlaps: 50\nqualifyingDate: 2030-03-09\nqualifyingTime: 14:00\n")
    _write(race / "race-results.yml",
           "- position: 1\n  driverId: a\n  constructorId: t1\n  gridPosition: 2\n  laps: 50\n  points: 25\n"
           "- position: 2\n  driverId: b\n  constructorId: t2\n  gridPosition: 1\n  laps: 50\n  points: 18\n"
           "- position: DNF\n  driverId: c\n  constructorId: t2\n  gridPosition: PL\n  laps: 3\n  points: 0\n"
           "- position: DNS\n  driverId: d\n  constructorId: t1\n  gridPosition:\n  laps: 0\n  points: 0\n")
    _write(race / "qualifying-results.yml",
           "- position: 1\n  driverId: b\n  constructorId: t2\n  q1: \"1:20.000\"\n  q2: \"1:19.500\"\n  q3:\n"
           "- position: 2\n  driverId: a\n  constructorId: t1\n  q1: \"1:20.100\"\n  q2:\n  q3:\n")
    t = data.read_f1db(src)
    r = t["races"].iloc[0]
    assert r["time_utc"] == "13:00" and r["circuit_type"] == "STREET"
    res = t["results"].set_index("driver")
    assert res.loc["a", "status"] == "FIN" and res.loc["c", "status"] == "DNF" and res.loc["d", "status"] == "DNS"
    assert res.loc["c", "grid"] == 3.0          # pit-lane start = behind all 3 starters
    assert t["quali"].set_index("driver").loc["b", "best_ms"] == 79500.0
    start = data.race_start_utc(t["races"]).iloc[0]
    assert start == pd.Timestamp("2030-03-10 13:00", tz="UTC")
