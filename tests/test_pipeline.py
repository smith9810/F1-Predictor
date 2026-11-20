import pandas as pd
import pytest

from f1 import pipeline


def _rows(n, extra=False):
    d = {"logged_at_utc": ["2026-11-20T09:00:00Z"] * n, "race_start_utc": ["2026-11-22T04:00:00Z"] * n,
         "race_id": [1170] * n, "stage": ["pre"] * n, "driver": [f"d{i}" for i in range(n)], "p_win": [1 / n] * n}
    if extra:
        d["p_win_raw"] = [0.5] * n
    return pd.DataFrame(d)


def test_append_log_refuses_forecasts_after_the_start(tmp_path):
    r = _rows(2)
    r["logged_at_utc"] = "2026-11-22T04:00:00Z"
    with pytest.raises(ValueError):
        pipeline.append_log(r, tmp_path / "log.csv")


def test_append_log_keeps_a_valid_csv_when_columns_are_added(tmp_path):
    p = tmp_path / "log.csv"
    pipeline.append_log(_rows(2), p)
    pipeline.append_log(_rows(3, extra=True), p)
    pipeline.append_log(_rows(1), p)
    df = pd.read_csv(p)
    assert len(df) == 6 and df["p_win_raw"].isna().sum() == 3
    assert list(df["p_win"].round(3)) == [0.5, 0.5, 0.333, 0.333, 0.333, 1.0]
