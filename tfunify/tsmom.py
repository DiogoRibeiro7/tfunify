from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .core import span_to_nu, ewma_volatility_from_returns, vol_normalised_returns, pct_returns_from_prices

FloatArray = NDArray[np.floating]


@dataclass
class TSMOMConfig:
    sigma_target_annual: float = 0.15
    a: int = 260
    span_sigma: int = 33
    L: int = 10
    M: int = 10


class TSMOM:
    """TSMOM with L, M blocks; vol targeting on annualised sigma."""

    def __init__(self, cfg: TSMOMConfig):
        self.cfg = cfg

    def run_from_prices(self, prices: FloatArray):
        r = pct_returns_from_prices(prices)
        return self.run_from_returns(r)

    def run_from_returns(self, r: FloatArray):
        cfg = self.cfg
        nu_sigma = span_to_nu(cfg.span_sigma)
        sigma_daily = ewma_volatility_from_returns(r, nu_sigma)
        z = vol_normalised_returns(r, sigma_daily)
        sigma_annual = sigma_daily * math.sqrt(cfg.a)

        n = r.size
        w = np.zeros(n)
        s_grid = np.zeros(n)
        L, M = cfg.L, cfg.M
        norm = math.sqrt(M * L)

        grid_idx = np.arange(0, n, L, dtype=int)
        for idx in grid_idx:
            block_ends = idx - np.arange(M) * L
            valid = block_ends[block_ends >= (L - 1)]
            if valid.size < M:
                continue
            signs = []
            for be in valid:
                start = be - (L - 1)
                c = np.mean(z[start : be + 1])
                signs.append(np.sign(c) if np.isfinite(c) and c != 0.0 else 0.0)
            s_val = (np.sum(signs) / M) * norm
            s_grid[idx] = s_val
            if idx > 0 and sigma_annual[idx - 1] > 0.0:
                w[idx] = cfg.sigma_target_annual / sigma_annual[idx - 1] * s_val
        for t in range(1, n):
            if w[t] == 0.0:
                w[t] = w[t - 1]
        f = np.zeros_like(r)
        f[1:] = w[:-1] * r[1:]
        return f, w, s_grid, sigma_daily
