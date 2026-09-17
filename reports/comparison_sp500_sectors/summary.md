# Pairs trading backtest summary

Formation (screening) period: **2019-01-01 to 2021-12-31**. Out-of-sample trading period: **2022-01-03 to 2025-12-31** (1003 trading days).

## Headline findings

- Universe: 89 tickers across energy, financials, utilities.
- Unit-root pretest: 89 tickers tested; 6 dropped as stationary on their own (levels ADF p <= 0.05): D, AEP, SRE, ATO, ETR, CMS.
- Pairs tested: 1263. Cointegrated at p < 0.05: **63** (about 63 would be expected by chance). Selected for backtesting: 62 (positive hedge ratio required). Both orientations tested per pair.
- Pairs profitable after costs out of sample: **28 of 62** (62 traded at least once).
- Equal-weight portfolio across all 62 pairs: Sharpe **-0.04**, annualised return -0.10%, max drawdown -4.32%, 1827 trades, win rate 62.2%, average hold 15.5 days.
- Best pair by Sharpe: **NTRS/COF** (Sharpe 1.08, return 44.71%, 32 trades).
- Worst pair by Sharpe: C/USB (Sharpe -1.03, return -36.17%).

## Assumptions

- Signal at close, execution at the next day's open. Rolling z-score window 30 days; entry |z| > 2.0, exit |z| < 0.5.
- Slippage 5 bps and commission 5 bps of notional, each leg, on entry and exit.
- $1 gross notional per pair per trade, fixed hedge ratio from the formation regression, risk-free rate 0, 252 trading days per year.
- Open positions on the last day are force-closed at the final close.

## Top 10 pairs by out-of-sample Sharpe

| ticker_a | ticker_b | sector | pvalue | hedge_ratio | half_life | sharpe | total_return | max_drawdown | n_trades | win_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| NTRS | COF | financials | 0.0015 | 0.39 | 20.3 | 1.08 | 44.71% | -7.79% | 32 | 78% |
| AXP | FITB | financials | 0.0413 | 3.55 | 23.3 | 0.85 | 33.01% | -7.95% | 32 | 69% |
| NTRS | JPM | financials | 0.0375 | 0.53 | 21.5 | 0.84 | 32.36% | -11.33% | 30 | 77% |
| AXP | COF | financials | 0.0042 | 0.79 | 15.3 | 0.68 | 23.63% | -7.99% | 31 | 71% |
| NTRS | BAC | financials | 0.0021 | 1.97 | 13.0 | 0.67 | 22.71% | -6.57% | 32 | 75% |
| NTRS | AXP | financials | 0.0052 | 0.48 | 20.7 | 0.58 | 20.60% | -9.23% | 30 | 70% |
| TFC | KEY | financials | 0.0082 | 2.24 | 15.5 | 0.53 | 16.55% | -11.07% | 31 | 65% |
| AON | AMP | financials | 0.0460 | 0.67 | 23.3 | 0.53 | 21.94% | -16.80% | 29 | 66% |
| DTE | PEG | utilities | 0.0027 | 1.94 | 12.4 | 0.51 | 12.45% | -10.87% | 31 | 68% |
| CB | SCHW | financials | 0.0482 | 1.30 | 30.9 | 0.50 | 21.93% | -9.86% | 39 | 67% |
