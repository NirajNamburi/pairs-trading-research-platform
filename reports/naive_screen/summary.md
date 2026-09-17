# Pairs trading backtest summary

Formation (screening) period: **2019-01-01 to 2021-12-31**. Out-of-sample trading period: **2022-01-03 to 2025-12-31** (1003 trading days).

## Headline findings

- Universe: 50 tickers across utilities_top50.
- Unit-root pretest: off (naive screen).
- Pairs tested: 1225. Cointegrated at p < 0.05: **94** (about 61 would be expected by chance). Selected for backtesting: 68 (positive hedge ratio required). Alphabetical orientation only.
- Pairs profitable after costs out of sample: **37 of 68** (68 traded at least once).
- Equal-weight portfolio across all 68 pairs: Sharpe **-0.09**, annualised return -0.47%, max drawdown -11.05%, 2167 trades, win rate 63.6%, average hold 14.8 days.
- Best pair by Sharpe: **CWEN/NEE** (Sharpe 0.68, return 22.98%, 34 trades).
- Worst pair by Sharpe: CNP/UGI (Sharpe -1.12, return -41.63%).

## Assumptions

- Signal at close, execution at the next day's open. Rolling z-score window 30 days; entry |z| > 2.0, exit |z| < 0.5.
- Slippage 10 bps and commission 5 bps of notional, each leg, on entry and exit.
- $1 gross notional per pair per trade, fixed hedge ratio from the formation regression, risk-free rate 0, 252 trading days per year.
- Open positions on the last day are force-closed at the final close.

## Top 10 pairs by out-of-sample Sharpe

| ticker_a | ticker_b | sector | pvalue | hedge_ratio | half_life | sharpe | total_return | max_drawdown | n_trades | win_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| CWEN | NEE | utilities_top50 | 0.0001 | 0.44 | 11.7 | 0.68 | 22.98% | -11.30% | 34 | 71% |
| CMS | WTRG | utilities_top50 | 0.0142 | 0.78 | 19.2 | 0.59 | 15.22% | -11.48% | 32 | 78% |
| EVRG | OTTR | utilities_top50 | 0.0318 | 0.49 | 27.9 | 0.49 | 16.64% | -9.08% | 35 | 74% |
| AEP | CWEN | utilities_top50 | 0.0238 | 0.23 | 23.0 | 0.48 | 26.11% | -20.51% | 33 | 70% |
| EXC | NFG | utilities_top50 | 0.0498 | 0.42 | 31.3 | 0.48 | 15.41% | -12.48% | 28 | 64% |
| D | NEE | utilities_top50 | 0.0210 | 0.17 | 12.8 | 0.44 | 20.72% | -24.81% | 30 | 67% |
| AWK | CWEN | utilities_top50 | 0.0196 | 4.02 | 17.2 | 0.44 | 16.12% | -10.45% | 34 | 71% |
| DTE | PEG | utilities_top50 | 0.0027 | 1.94 | 12.4 | 0.38 | 9.34% | -11.50% | 31 | 65% |
| D | EVRG | utilities_top50 | 0.0427 | 0.44 | 17.4 | 0.38 | 12.07% | -13.88% | 29 | 72% |
| AEP | SWX | utilities_top50 | 0.0401 | 0.18 | 27.6 | 0.31 | 14.44% | -15.39% | 36 | 75% |
