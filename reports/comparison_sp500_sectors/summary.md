# Pairs trading backtest summary

Formation (screening) period: **2019-01-01 to 2021-12-31**. Out-of-sample trading period: **2022-01-03 to 2025-12-31** (1003 trading days).

## Headline findings

- Universe: 89 tickers across energy, financials, utilities.
- Pairs tested: 1410. Cointegrated at p < 0.05: **74** (about 70 would be expected by chance). Selected for backtesting: 71 (positive hedge ratio required).
- Pairs profitable after costs out of sample: **35 of 71** (71 traded at least once).
- Equal-weight portfolio across all 71 pairs: Sharpe **0.00**, annualised return 0.00%, max drawdown -4.34%, 2141 trades, win rate 63.2%, average hold 15.4 days.
- Best pair by Sharpe: **AXP/FITB** (Sharpe 0.85, return 33.01%, 32 trades).
- Worst pair by Sharpe: AJG/AMP (Sharpe -1.06, return -42.00%).

## Assumptions

- Signal at close, execution at the next day's open. Rolling z-score window 30 days; entry |z| > 2.0, exit |z| < 0.5.
- Slippage 5 bps and commission 5 bps of notional, each leg, on entry and exit.
- $1 gross notional per pair per trade, fixed hedge ratio from the formation regression, risk-free rate 0, 252 trading days per year.
- Open positions on the last day are force-closed at the final close.

## Top 10 pairs by out-of-sample Sharpe

| ticker_a | ticker_b | sector | pvalue | hedge_ratio | half_life | sharpe | total_return | max_drawdown | n_trades | win_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| AXP | FITB | financials | 0.0413 | 3.55 | 23.3 | 0.85 | 33.01% | -7.95% | 32 | 69% |
| AXP | COF | financials | 0.0042 | 0.79 | 15.3 | 0.68 | 23.63% | -7.99% | 31 | 71% |
| BAC | NTRS | financials | 0.0038 | 0.48 | 14.0 | 0.59 | 19.82% | -6.59% | 33 | 73% |
| DTE | PEG | utilities | 0.0027 | 1.94 | 12.4 | 0.51 | 12.45% | -10.87% | 31 | 68% |
| CB | SCHW | financials | 0.0482 | 1.30 | 30.9 | 0.50 | 21.93% | -9.86% | 39 | 67% |
| D | NEE | utilities | 0.0210 | 0.17 | 12.8 | 0.50 | 23.72% | -23.64% | 30 | 67% |
| AXP | PNC | financials | 0.0498 | 0.85 | 18.2 | 0.50 | 17.71% | -8.61% | 30 | 63% |
| KEY | TFC | financials | 0.0140 | 0.43 | 16.4 | 0.48 | 14.86% | -11.49% | 28 | 57% |
| D | EVRG | utilities | 0.0427 | 0.44 | 17.4 | 0.48 | 14.97% | -13.10% | 29 | 72% |
| D | ES | utilities | 0.0070 | 0.45 | 12.2 | 0.41 | 10.32% | -16.86% | 31 | 77% |
