"""Monte Carlo race forecasts from strengths, starting grid and retirement probabilities.

Post-qualifying (grid known):  race strength = pace + gamma * (-log grid); finishers ordered by
strength + Gumbel noise (exactly Plackett-Luce); each driver retires with his DNF probability and
retirements fill the places after the finishers.
Pre-qualifying: qualifying is simulated first (kappa_quali * pace + Gumbel), its order becomes the grid,
then the race is simulated as above. Grid penalties are not known before qualifying and are ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POINTS_2026 = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], float)


def _gumbel(rng, shape):
    return -np.log(-np.log(rng.uniform(1e-12, 1.0, size=shape)))


def simulate(pace, p_dnf, kappa_q: float, gamma: float, grid=None, n_sims: int = 20000, seed: int = 0) -> np.ndarray:
    """Returns an n_sims x N matrix of finishing positions (1 = win). DNFs take the last places."""
    rng = np.random.default_rng(seed)
    pace = np.asarray(pace, float)
    N = len(pace)
    if grid is None:
        q = kappa_q * pace[None, :] + _gumbel(rng, (n_sims, N))
        gpos = np.empty((n_sims, N))
        gpos[np.arange(n_sims)[:, None], np.argsort(-q, axis=1)] = np.arange(1, N + 1)
    else:
        g = np.asarray(grid, float)
        g = np.where(np.isfinite(g) & (g > 0), g, N)
        gpos = np.broadcast_to(g, (n_sims, N))
    perf = pace[None, :] + gamma * (-np.log(gpos)) + _gumbel(rng, (n_sims, N))
    dnf = rng.uniform(size=(n_sims, N)) < np.asarray(p_dnf, float)[None, :]
    perf = np.where(dnf, -1e6 + rng.uniform(size=(n_sims, N)), perf)    # retirements last, random order
    order = np.argsort(-perf, axis=1)
    pos = np.empty((n_sims, N), int)
    pos[np.arange(n_sims)[:, None], order] = np.arange(1, N + 1)
    return pos, dnf


def summary(pos: np.ndarray, dnf: np.ndarray, points=POINTS_2026) -> pd.DataFrame:
    N = pos.shape[1]
    pts = np.zeros(N + 1)
    pts[1:len(points) + 1] = points[:N]
    return pd.DataFrame({
        "p_win": (pos == 1).mean(0), "p_podium": (pos <= 3).mean(0), "p_top6": (pos <= 6).mean(0),
        "p_points": ((pos <= 10) & ~dnf).mean(0), "p_dnf": dnf.mean(0), "exp_pos": pos.mean(0),
        "exp_points": np.where(dnf, 0.0, pts[np.minimum(pos, N)]).mean(0),
    })


def position_dist(pos: np.ndarray) -> np.ndarray:
    """N x N matrix: row = driver, column k = P(finishing position k+1)."""
    N = pos.shape[1]
    out = np.zeros((N, N))
    for k in range(N):
        out[:, k] = (pos == k + 1).mean(0)
    return out
