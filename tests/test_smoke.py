import numpy as np
from tfunify.european import EuropeanTF, EuropeanTFConfig


def test_smoke_european_runs():
    np.random.seed(0)
    n = 600
    a = 260
    r = 0.02 / a + 0.01 * np.random.randn(n)
    prices = 100 * np.cumprod(1 + np.r_[0.0, r[1:]])

    cfg = EuropeanTFConfig(
        sigma_target_annual=0.15, a=a, span_sigma=33, mode="longshort", span_long=250, span_short=20
    )
    sysm = EuropeanTF(cfg)
    f, w, s, sigma = sysm.run_from_prices(prices)

    assert f.shape == prices.shape
    assert np.isfinite(f[10:]).all()
