# tfunify Examples

This directory contains comprehensive examples demonstrating how to use tfunify for trend-following system development and analysis.

## Quick Start

### Installation

For all examples:

```bash
pip install tfunify matplotlib pandas scipy

# Or with Yahoo Finance support:
pip install tfunify[yahoo] matplotlib pandas scipy
```

For individual examples, see the requirements listed at the top of each file.

### Running Examples

```bash
# Run from the project root directory
python examples/basic_usage.py
python examples/real_data_analysis.py
python examples/performance_comparison.py
python examples/parameter_optimization.py
python examples/portfolio_integration.py
```

## Examples Overview

### 1\. <basic_usage.py>

**Getting Started with tfunify**

Demonstrates basic usage of all three trend-following systems with synthetic data.

**Features:**

- European TF with volatility targeting
- American TF with ATR-based breakouts
- TSMOM with block-averaged momentum
- Performance comparison and visualization
- Clear explanations for beginners

**Key Learning Points:**

- System configuration and setup
- Running backtests
- Interpreting results
- Basic performance metrics

**Output:**

- Console performance summary
- `basic_usage_comparison.png` - Comparison chart

--------------------------------------------------------------------------------

### 2\. <real_data_analysis.py>

**Real Market Data Analysis**

Shows how to use tfunify with real market data from Yahoo Finance.

**Features:**

- Automatic data downloading
- Data quality analysis
- Multi-system backtesting
- Market regime analysis
- Comprehensive performance reporting

**Requirements:**

```bash
pip install tfunify[yahoo] matplotlib pandas
```

**Key Learning Points:**

- Working with real market data
- Data preprocessing and validation
- Performance across market regimes
- Rolling performance analysis

**Output:**

- Downloaded data in `examples/data/`
- `real_data_analysis.png` - Performance charts
- Regime-based performance breakdown

--------------------------------------------------------------------------------

### 3\. <performance_comparison.py>

**Comprehensive System Comparison**

Conducts thorough comparison of all three systems across multiple market scenarios.

**Features:**

- 6 different market scenarios (trending, mean-reverting, volatile, etc.)
- Statistical significance testing
- Monte Carlo robustness analysis
- Risk-adjusted performance metrics
- Comprehensive visualization

**Key Learning Points:**

- System behavior in different market conditions
- Statistical validation of results
- Risk-adjusted performance comparison
- Robustness testing methodology

**Output:**

- `performance_comparison.png` - Multi-panel analysis
- Statistical test results
- Monte Carlo confidence intervals

--------------------------------------------------------------------------------

### 4\. <parameter_optimization.py>

**Parameter Tuning and Optimization**

Demonstrates various parameter optimization techniques for trend-following systems.

**Features:**

- Grid search optimization
- Random search optimization
- Bayesian/differential evolution optimization
- Walk-forward analysis
- Out-of-sample validation
- Parameter stability analysis

**Requirements:**

```bash
pip install scipy  # For advanced optimization
```

**Key Learning Points:**

- Different optimization methodologies
- Avoiding overfitting
- Walk-forward validation
- Parameter sensitivity analysis

**Output:**

- `parameter_optimization.png` - Optimization results
- Best parameter sets for each system
- Out-of-sample performance validation

--------------------------------------------------------------------------------

### 5\. <portfolio_integration.py>

**Portfolio Management Integration**

Shows how to integrate trend-following systems into broader portfolio strategies.

**Features:**

- Traditional portfolio construction (60/40)
- Risk parity approaches
- Trend-following overlay strategies
- Ensemble methods
- Portfolio optimization
- Multi-asset analysis

**Key Learning Points:**

- Portfolio construction techniques
- Diversification benefits of trend-following
- Risk management at portfolio level
- Ensemble strategy development

**Output:**

- `portfolio_integration.png` - Portfolio analysis
- Risk-return comparison
- Optimal allocation recommendations

## Example Data Structure

When running examples, the following structure is created:

```
examples/
├── README.md                      # This file
├── basic_usage.py                 # Basic usage example
├── real_data_analysis.py          # Real data analysis
├── performance_comparison.py      # System comparison
├── parameter_optimization.py      # Parameter tuning
├── portfolio_integration.py       # Portfolio strategies
├── data/                          # Downloaded market data
│   ├── SPY_5y.csv                # Example: S&P 500 data
│   └── ...                       # Other downloaded datasets
├── basic_usage_comparison.png     # Generated charts
├── real_data_analysis.png
├── performance_comparison.png
├── parameter_optimization.png
└── portfolio_integration.png
```

## Key Concepts Demonstrated

### System Configuration

Each example shows how to properly configure the three systems:

```python
# European TF
config = EuropeanTFConfig(
    sigma_target_annual=0.15,    # Risk targeting
    mode="longshort",            # Filter type
    span_long=250,               # Slow filter
    span_short=20                # Fast filter
)

# American TF
config = AmericanTFConfig(
    q=2.5,                       # Entry threshold
    p=4.0,                       # Stop loss
    r_multiple=0.01              # Position sizing
)

# TSMOM
config = TSMOMConfig(
    L=10,                        # Block length
    M=10,                        # Number of blocks
    sigma_target_annual=0.12     # Risk targeting
)
```

### Performance Analysis

All examples demonstrate comprehensive performance analysis:

- **Return Metrics**: Annual return, total return, compound growth
- **Risk Metrics**: Volatility, maximum drawdown, VaR
- **Risk-Adjusted**: Sharpe ratio, Calmar ratio, Sortino ratio
- **Trading Stats**: Win rate, profit factor, average win/loss

### Best Practices

The examples illustrate important best practices:

1. **Input Validation**: All systems validate inputs and provide clear error messages
2. **Robust Backtesting**: Handle missing data, corporate actions, and edge cases
3. **Out-of-Sample Testing**: Always validate on unseen data
4. **Statistical Significance**: Test whether results are statistically meaningful
5. **Risk Management**: Focus on risk-adjusted returns, not just absolute returns

## Common Use Cases

### Quantitative Research

- Validate academic research on trend-following
- Develop new system variations
- Conduct performance attribution analysis

### Portfolio Management

- Add diversifying strategies to traditional portfolios
- Implement systematic overlay strategies
- Risk parity and factor-based investing

### Risk Management

- Reduce portfolio volatility through diversification
- Implement systematic stop-loss mechanisms
- Monitor and control drawdowns

### Strategy Development

- Prototype new trend-following approaches
- Optimize existing strategies
- Ensemble multiple systems for robustness

## Tips for Success

### 1\. Start Simple

Begin with `basic_usage.py` to understand the core concepts before moving to advanced examples.

### 2\. Use Real Data

The `real_data_analysis.py` example shows how real market data behaves differently from synthetic data.

### 3\. Validate Everything

Always use out-of-sample testing as shown in the optimization example.

### 4\. Consider Regime Changes

Market conditions change - the performance comparison example shows how systems behave differently across regimes.

### 5\. Focus on Risk-Adjusted Returns

Absolute returns can be misleading - always consider risk-adjusted metrics like Sharpe ratio.

## Troubleshooting

### Common Issues

**Import Errors:**

```bash
# Install optional dependencies
pip install tfunify[yahoo] matplotlib pandas scipy
```

**Data Download Issues:**

- Check internet connection
- Try different ticker symbols
- Use shorter time periods if data is limited

**Performance Issues:**

- Reduce dataset size for faster testing
- Use fewer optimization trials
- Consider parallel processing for large parameter grids

**Plot Display Issues:**

- Ensure matplotlib backend is properly configured
- Save plots to files if display doesn't work
- Check if running in headless environment

### Getting Help

1. Check the main [README.md](../README.md) for basic setup
2. Review the [CHANGELOG.md](../CHANGELOG.md) for recent changes
3. Run examples with smaller datasets to isolate issues
4. Ensure all dependencies are installed correctly

## Contributing Examples

We welcome contributions of additional examples! Good candidates include:

- **Sector Rotation**: Using trend-following for sector allocation
- **Cryptocurrency**: Applying systems to digital assets
- **Intraday**: High-frequency trend-following applications
- **Options**: Using trend signals for options strategies
- **International**: Multi-market and currency considerations

When contributing examples:

1. Follow existing code style and documentation patterns
2. Include comprehensive comments and explanations
3. Provide both synthetic and real data examples where applicable
4. Include performance analysis and visualization
5. Test across different market conditions

--------------------------------------------------------------------------------

_These examples demonstrate the power and flexibility of trend-following systems. Start with the basic example and gradually work through more advanced concepts as you build expertise._
