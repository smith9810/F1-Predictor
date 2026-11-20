"""Reproduce the settings search on 2005-2012 (validation seasons only). About 10 minutes.
Writes models/tuning_2005_2012.json. Run from the repo root: python scripts/tune.py"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f1 import backtest as bt, data  # noqa: E402

T = data.load(ROOT / "data")
dm = bt.DNFModels(T)
out = []
for hl, rd, lk in itertools.product([180, 365, 730], [0.3, 1.0, 3.0], [1.0, 3.0, 10.0]):
    p = {"half_life_days": hl, "ridge_driver": rd, "link": lk}
    res = bt.run(T, 2005, 2012, p, n_sims=2000, dnfm=dm)
    s = res[res.method == "model"].groupby("stage")[["pos_rps", "win_ll", "podium_brier"]].mean()
    out.append({**p, **{f"{st}_{c}": float(s.loc[st, c]) for st in s.index for c in s.columns}})
    print(out[-1], flush=True)
(ROOT / "models" / "tuning_2005_2012.json").write_text(json.dumps(out, indent=1))
