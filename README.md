# tfunify

[![CI](https://github.com/diogoribeiro7/tfunify/workflows/CI/badge.svg)](https://github.com/diogoribeiro7/tfunify/actions) [![PyPI version](https://badge.fury.io/py/tfunify.svg)](https://badge.fury.io/py/tfunify) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Unified trend-following systems implemented in pure NumPy with comprehensive validation and testing.

## Systems

- **European TF**: Variance-preserving EWMA on volatility-normalised returns with volatility targeting
- **American TF**: Fast/slow price filters with ATR buffers and trailing stops
- **TSMOM**: Time Series Momentum using block-averaged signs of cumulative vol-normalised returns

## Installation

```bash
# Standard installation
pip install tfunify

# With Yahoo Finance data downloading
pip install tfunify[yahoo]

# Development installation with Poetry
poetry add tfunify
poetry add tfunify[yahoo]  # with extras
```

## Quick Start

### European TF System

```python
import numpy as np
from tfunify.european import EuropeanTF, EuropeanTFConfig

# Generate sample data or load your own
np.random.seed(0)
n = 1500
returns = 0.05/260 + 0.01 * np.random.randn(n)  
prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])

# Configure system
cfg = EuropeanTFConfig(
    sigma_target_annual=0.15,  # 15% vol target
    a=260,                     # trading days per year
    span_sigma=33,             # volatility estimation span
    mode="longshort",          # "single" or "longshort"
    span_long=250,             # long MA span
    span_short=20              # short MA span
)

# Run system
system = EuropeanTF(cfg)
pnl, weights, signal, volatility = system.run_from_prices(prices)

# Analyze results
annual_return = np.mean(pnl) * cfg.a
annual_vol = np.std(pnl, ddof=0) * np.sqrt(cfg.a)
sharpe_ratio = annual_return / annual_vol
print(f"Annual Return: {annual_return:.2%}")
print(f"Annual Volatility: {annual_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
```

### American TF System

```python
from tfunify.american import AmericanTF, AmericanTFConfig

# Configure breakout system with ATR
cfg = AmericanTFConfig(
    span_long=250,      # slow MA span
    span_short=20,      # fast MA span  
    atr_period=33,      # ATR calculation period
    q=5.0,             # entry threshold (ATR multiples)
    p=5.0,             # stop loss (ATR multiples)
    r_multiple=0.01    # risk per unit
)

system = AmericanTF(cfg)
pnl, units = system.run(close_prices, high_prices, low_prices)
```

### TSMOM System

```python
from tfunify.tsmom import TSMOM, TSMOMConfig

# Configure time series momentum
cfg = TSMOMConfig(
    sigma_target_annual=0.15,
    span_sigma=33,
    L=10,    # block length in days
    M=10     # number of blocks to average
)

system = TSMOM(cfg)
pnl, weights, signal_grid, volatility = system.run_from_prices(prices)
```

## Command Line Interface

The CLI provides easy access to all systems:

```bash
# Download data from Yahoo Finance
tfu download SPY --out spy_data.csv --period 2y

# Run European TF system
tfu european --csv spy_data.csv --target 0.15 --longshort \
    --span-long 250 --span-short 20

# Run American TF system  
tfu american --csv spy_data.csv --q 2.0 --p 3.0 \
    --atr-period 20 --r-multiple 0.01

# Run TSMOM system
tfu tsmom --csv spy_data.csv --target 0.12 \
    --L 5 --M 12 --span-sigma 33
```

All commands output summary statistics and save detailed results to `.npz` files.

## Data Format

CSV files must contain at minimum a `close` column. Optional `high` and `low` columns will be used if available, otherwise they default to `close` values.

```csv
date,open,high,low,close,volume
2020-01-02,323.8,325.0,322.1,324.1,28000000
2020-01-03,325.2,327.1,324.8,326.9,31000000
...
```

## System Details

### European TF

- Applies exponentially weighted moving averages to volatility-normalized returns
- Supports single trend filter or long-short configuration
- Includes volatility targeting for consistent risk exposure
- Variance-preserving EWMA ensures proper signal scaling

### American TF

- Uses fast/slow moving average crossovers with ATR-based buffers
- Implements trailing stops for risk management
- Position sizing based on ATR and risk multiples
- Handles entry/exit logic with state machine

### TSMOM

- Divides time series into blocks of length L
- Calculates momentum signal from M historical blocks
- Uses sign of average returns within each block
- Applies volatility targeting to final positions

## Performance Considerations

All systems are implemented in pure NumPy for performance:

- Vectorized operations where possible
- Minimal Python loops (only where state is required)
- Memory-efficient array operations
- Input validation with clear error messages

## Development

```bash
# Clone and setup
git clone https://github.com/diogoribeiro7/tfunify.git
cd tfunify
poetry install --with dev

# Run tests
poetry run pytest --cov=tfunify

# Type checking
poetry run mypy src

# Linting  
poetry run ruff check .
poetry run ruff format .

# Install pre-commit hooks
poetry run pre-commit install
```

## Theory Background

This implementation is based on the unified framework presented in:

**Sepp, A. & Lucic, V. (2025). "The Science and Practice of Trend-following Systems." 17th June, 2025.**

The paper provides theoretical foundations for classifying trend-following systems into three categories:

### European TF

Based on variance-preserving exponentially weighted moving averages applied to volatility-normalized returns. The paper derives an exact relationship between the system's P&L and the autocorrelation function of the underlying return process, showing that TF systems are profitable when returns exhibit positive long-term autocorrelation, even with short-term mean reversion. The volatility targeting ensures consistent risk-adjusted exposure across market regimes.

### American TF

Implements classic breakout methodology with Average True Range (ATR) for adaptive thresholds. This system uses fast/slow moving average crossovers with ATR-based buffers for entry signals and trailing stops for systematic risk management. The approach allows trends to develop while providing downside protection.

### TSMOM (Time Series Momentum)

Follows the time series momentum framework using block-averaged momentum signals. The system divides the return series into blocks, calculates the sign of cumulative returns within each block, and averages across multiple historical blocks. Volatility normalization ensures robust trend detection across different market conditions.

The theoretical framework shows that TF systems benefit from the square of the drift in the return process and provides performance attribution to trend, mean reversion, and long-term drift components. The paper also demonstrates the defensive profile of TF systems and their diversification benefits when combined with long-only portfolios.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions welcome! Please ensure:

- Tests pass (`pytest`)
- Type checking passes (`mypy`)
- Code is formatted (`ruff format`)
- Pre-commit hooks pass

## Citation

If you use this library in academic work, please cite both the software and the foundational paper:

```bibtex
@software{tfunify2025,
  author = {Diogo Ribeiro},
  title = {tfunify: Unified Trend-Following Systems},
  year = {2025},
  url = {https://github.com/diogoribeiro7/tfunify}
}

@article{sepp2025trend,
  author = {Sepp, A. and Lucic, V.},
  title = {The Science and Practice of Trend-following Systems},
  year = {2025},
  month = {June},
  day = {17},
  abstract = {We present a unified approach to the design of trend-following (TF) systems and their classification into European, American, and Time Series Momentum systems...},
  keywords = {Trend-following Strategies, Managed Futures, Fractional Processes, Autocorrelation, Portfolio Diversification}
}
```
