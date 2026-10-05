# Examples

Five scripts in the [`examples/`](https://github.com/DiogoRibeiro7/tfunify/tree/develop/examples) directory of the repository. They run from any directory, and the test suite runs each of them.

```bash
git clone https://github.com/DiogoRibeiro7/tfunify.git
cd tfunify
pip install -e ".[plot]"
python examples/basic_usage.py --plot
```

| Script | What it shows |
|---|---|
| `basic_usage.py` | The three systems on one simulated market: how each is configured, run and summarised. `--plot` saves a chart of the price and of the cumulative returns. |
| `performance_comparison.py` | Sharpe ratios of the three systems in a random walk, a trending, a mean-reverting and a drifting market, averaged over simulated paths, with the closed form for the European system next to them. |
| `parameter_optimization.py` | A filter span chosen by a grid search in sample, the same span out of sample, and the Sharpe ratio the closed form assigns to each span. |
| `portfolio_integration.py` | A trend-following allocation on three simulated assets, held on top of a 60/40 portfolio (a "sleeve"), and how it did in the worst quarters of that portfolio. |
| `real_data_analysis.py` | The three systems on a Yahoo Finance symbol (`--ticker SPY`, needs `tfunify[yahoo]`) or on your own file (`--csv prices.csv`). |

All but the last use simulated data, so their output says how the systems behave on a process whose properties are known. It says nothing about what they would earn on a market.

The last one prints the return of the instrument times the weight, as for a futures position. The price of a share or of a fund such as SPY is not in excess of the cost of financing it: a long position earns less than shown, and a short one more, by about the short-term interest rate times the weight.

## Choosing a span

`parameter_optimization.py` is worth a look before running a grid search of your own:

```text
European system, single filter; AR(1) returns with phi = 0.05, Sharpe ratio 0.5; 15 years in each sample

  span  in sample  out of sample  closed form
     5       1.00           0.71         0.63
    10       0.88           0.56         0.51
    21       0.61           0.43         0.40
    63       0.16           0.38         0.31
   125       0.07           0.45         0.29
   250       0.14           0.55         0.30
```

The best span in sample has a Sharpe ratio of 1.00 there and 0.71 on fresh data from the same process; the closed form, which knows the process, says 0.63. A Sharpe ratio measured on 15 years has a standard error of about 0.26, so the in-sample figure of the chosen span is 1.4 standard errors above what the process supports. That is what picking the largest of six noisy figures does.
