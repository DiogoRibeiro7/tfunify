import numpy as np
from tfunify.core import span_to_nu
from tfunify.european import EuropeanTF, EuropeanTFConfig

def test_span_to_nu_bounds():
    assert 0.0 < span_to_nu(2) < 1.0

def test_european_basic_signal_nonzero():
    np.random.seed(42)
    n = 800
    a = 260
    r = 0.03/a + 0.01*np.random.randn(n)
    prices = 100*np.cumprod(1 + np.r_[0.0, r[1:]])

    cfg = EuropeanTFConfig(sigma_target_annual=0.15, a=a, span_sigma=33, mode="single", span_long=100)
    sysm = EuropeanTF(cfg)
    f, w, s, sigma = sysm.run_from_prices(prices)
    assert np.nanstd(s) > 0.0
    assert np.isfinite(w[10:]).all()
