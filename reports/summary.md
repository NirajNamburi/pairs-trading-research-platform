# Pairs trading backtest summary

Formation (screening) period: **2019-01-01 to 2021-12-31**. Out-of-sample trading period: **2022-01-03 to 2024-12-31** (753 trading days).

## Headline findings

- Universe: 89 tickers across energy, financials, utilities.
- Pairs tested: 1410. Cointegrated at p < 0.05: **74** (about 70 would be expected by chance). Selected for backtesting: 71 (positive hedge ratio required).
- Pairs profitable after costs out of sample: **33 of 71** (71 traded at least once).
- Equal-weight portfolio across all 71 pairs: Sharpe **-0.06**, annualised return -0.16%, max drawdown -4.34%, 1605 trades, win rate 62.7%, average hold 15.5 days.
- Best pair by Sharpe: **AXP/FITB** (Sharpe 1.14, return 30.96%, 25 trades).
- Worst pair by Sharpe: KMI/SLB (Sharpe -1.38, return -39.71%).

## Assumptions

- Signal at close, execution at the next day's open. Rolling z-score window 30 days; entry |z| > 2.0, exit |z| < 0.5.
- Slippage 5 bps and commission 5 bps of notional, each leg, on entry and exit.
- $1 gross notional per pair per trade, fixed hedge ratio from the formation regression, risk-free rate 0, 252 trading days per year.
- Open positions on the last day are force-closed at the final close.

## Top 10 pairs by out-of-sample Sharpe

| ticker_a | ticker_b | sector | pvalue | hedge_ratio | half_life | sharpe | total_return | max_drawdown | n_trades | win_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| AXP | FITB | financials | 0.0413 | 3.55 | 23.3 | 1.14 | 30.96% | -7.55% | 25 | 72% |
| AXP | COF | financials | 0.0042 | 0.79 | 15.3 | 0.71 | 19.09% | -7.99% | 21 | 67% |
| MA | PGR | financials | 0.0088 | 3.84 | 16.9 | 0.69 | 23.67% | -12.52% | 24 | 75% |
| BAC | NTRS | financials | 0.0038 | 0.48 | 14.0 | 0.69 | 18.23% | -6.59% | 26 | 73% |
| AXP | PNC | financials | 0.0498 | 0.85 | 18.2 | 0.65 | 15.30% | -8.05% | 22 | 64% |
| NTRS | SCHW | financials | 0.0478 | 0.84 | 22.6 | 0.59 | 18.87% | -12.54% | 22 | 68% |
| AEP | NEE | utilities | 0.0395 | 0.16 | 24.7 | 0.56 | 19.92% | -17.68% | 25 | 72% |
| AXP | NTRS | financials | 0.0064 | 1.85 | 21.9 | 0.56 | 15.00% | -9.88% | 21 | 67% |
| AEP | CNP | utilities | 0.0201 | 0.37 | 23.9 | 0.55 | 20.33% | -15.88% | 26 | 65% |
| NTRS | ZION | financials | 0.0380 | 1.39 | 27.0 | 0.51 | 17.69% | -11.86% | 22 | 73% |
