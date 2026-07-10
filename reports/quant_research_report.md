# Cross-Asset Graph Diffusion Signals for Equity Return Prediction

*A walk-forward equity research study across semiconductor, Japanese media/gaming, and transport/logistics equities.*

## 1. Abstract

This project tests whether short-horizon information diffusion across economically connected
equities -- driven by supply-chain, thematic, and investor-flow linkages -- contains exploitable
predictive information for next-day residual equity returns. A rolling correlation graph is built
strictly from past returns, and lagged neighbour return / residual-shock features derived from
that graph are fed into interpretable linear models (Ridge, ElasticNet) inside a walk-forward
backtest with realistic transaction costs. The headline out-of-sample net Sharpe ratio for the
Ridge strategy over the test period is **-1.81**,
versus **0.92** for buy-and-hold SPY.
This is a weak/negative result and is reported honestly -- see Sections 9 and the failure-case discussion.

## 2. Hypothesis

Public companies in the same sector, supply chain, or thematic bucket share exposure to common
news shocks (a foundry capacity update, a console sales report, a fuel-price shock), but the market
does not always price that shared exposure into every related name simultaneously. Some names react
immediately; economically linked names may react with a short delay as the market connects the dots,
as arbitrageurs and cross-asset desks slowly transmit the information, or as index/ETF flow effects
spill over. If this delay exists and is even weakly persistent, a stock's own **lagged neighbour
returns** (where "neighbour" is defined by a rolling return-correlation graph, not a static sector
label) should carry information about that stock's **near-future residual return**, after removing
market- and sector-level beta exposure.

This hypothesis is deliberately narrow in scope (semiconductors, Japanese media/gaming/IP, and
transport/logistics -- three groups with plausible supply-chain, thematic, and investor-flow
linkages respectively) rather than a universal claim about all equities.

## 3. Data and Universe

* **Source:** `yfinance` (Yahoo Finance) adjusted daily OHLCV, 2015-01-01 to latest available.
* **Universe:** 29 configured tickers across three sectors (semiconductors/tech,
  Japanese media/gaming/IP, transport/logistics) plus benchmarks SPY, QQQ, SMH, IYT.
  See `config/universe.yaml` for the exact list.
* **Cleaning:** tickers with fewer than 500 trading days of history are dropped
  (a data-sufficiency filter, not a survivorship-bias correction -- this project does **not** claim
  to be survivorship-bias-free; see Section 9). Trading calendars are aligned across the surviving
  universe; short (<=3 day) gaps are forward-filled, longer gaps are dropped.
* **Returns:** simple daily returns from adjusted close. Residual returns are computed via a rolling
  beta regression (window=60 days, method="sector")
  re-estimated **using only data through t-1** at every date, so the beta applied at date t never saw
  date t's own return.

## 4. Feature Construction

All features are computed as of the close of a "feature date" `f` and are only ever paired with a
target return realised strictly after `f` (see `src/graph_diffusion_signal/features.py` module
docstring for the exact timing convention). The rolling correlation graph (window=
60 days, top-k=8, min |corr|=
0.15) is built from returns up to `f-1` only, keeps each node's strongest
positive-correlation neighbours, and is row-normalised. Feature families:

* Own lagged returns (1d, 3d, 5d, 20d)
* Rolling realised volatility (20d) and volume surprise (z-score vs
  20d rolling mean/std)
* Neighbour return diffusion score and neighbour residual-shock diffusion score (graph-weighted
  sums of neighbours' same-day returns/residuals)
* Graph weighted-degree centrality
* Sector benchmark return and broad market return
* Momentum (20d) and short-term reversal (5d)
* Volatility-normalised versions of the return and diffusion features

## 5. Walk-Forward Backtest Design

* Initial training window: 3 years, expanding thereafter.
* Refit frequency: Q (monthly="M", quarterly="Q").
* First eligible test date: 2019-01-01.
* At every refit boundary the model (StandardScaler + Ridge/ElasticNet, hyperparameters chosen by
  time-series cross-validation **inside the training window only**) is retrained from scratch on
  strictly past data and then used to predict every date up to the next refit boundary -- fully
  out-of-sample relative to that fold's training set.
* `tests/test_backtest_no_lookahead.py` verifies this directly, including a test that perturbs
  returns after a cutoff date by a large amount and checks that predictions dated on or before the
  cutoff are bit-for-bit unchanged.

## 6. Transaction Cost Model

Costs are charged as `cost_bps / 10,000 * sum(|weight change|)` at each rebalance, i.e. proportional
to gross traded notional. Default: 5 bps one-way. Sensitivity is reported
at 0, 2, 5, 10, 25 bps.

## 7. Real-Market-Data Result

*Computed on live adjusted OHLCV data via yfinance (see Section 3 for exact coverage). This is the section that matters for any CV or interview claim.*

![Equity Curve](figures/equity_curve.png)

| Strategy | Net Ann. Return | Net Ann. Vol | Net Sharpe | Net Max DD | Gross Sharpe | Avg Turnover |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge (graph diffusion features) | -0.3867 | 0.2517 | -1.8139 | -0.9755 | -0.8369 | 1.9525 |
| ElasticNet (graph diffusion features) | -0.1478 | 0.2194 | -0.6194 | -0.7447 | -0.6189 | 0.0008 |
| Rank on diffusion score (no fitting) | -0.2341 | 0.2402 | -0.9897 | -0.8863 | 0.2261 | 2.3172 |
| Naive momentum baseline | -0.0688 | 0.2718 | -0.1261 | -0.6722 | 0.1376 | 0.5688 |
| Naive reversal baseline | -0.1012 | 0.2586 | -0.2836 | -0.7931 | 0.2456 | 1.0858 |
| SPY (buy & hold) | 0.1745 | 0.1945 | 0.9245 | -0.3372 | 0.9245 | n/a |
| QQQ (buy & hold) | 0.2341 | 0.2404 | 0.9958 | -0.3512 | 0.9958 | n/a |
| SMH (buy & hold) | 0.4259 | 0.3601 | 1.1662 | -0.4530 | 1.1662 | n/a |
| IYT (buy & hold) | 0.1194 | 0.2482 | 0.5787 | -0.4077 | 0.5787 | n/a |

![Drawdown](figures/drawdown.png)

### Annual Returns (Ridge, net of costs)

![Annual Returns](figures/year_by_year.png)

| Year | Return | Volatility | Sharpe | Max Drawdown | N Days |
| --- | --- | --- | --- | --- | --- |
| 2019 | -0.4922 | 0.1732 | -3.8204 | -0.4957 | 251 |
| 2020 | -0.3920 | 0.4019 | -1.0362 | -0.4027 | 253 |
| 2021 | -0.4975 | 0.1938 | -3.4490 | -0.4971 | 252 |
| 2022 | -0.4608 | 0.2405 | -2.4453 | -0.4676 | 251 |
| 2023 | -0.2354 | 0.1665 | -1.5274 | -0.2859 | 250 |
| 2024 | -0.2039 | 0.2260 | -0.8950 | -0.2906 | 252 |
| 2025 | -0.3834 | 0.2650 | -1.6896 | -0.5252 | 250 |
| 2026 | -0.3364 | 0.2812 | -1.3173 | -0.1973 | 125 |

### Transaction Cost Sensitivity (Ridge)

![Cost Sensitivity](figures/cost_sensitivity.png)

| cost_bps | sharpe | ann_return |
| --- | --- | --- |
| 0 | -0.837 | -0.215 |
| 2 | -1.228 | -0.289 |
| 5 | -1.814 | -0.387 |
| 10 | -2.789 | -0.521 |
| 25 | -5.694 | -0.772 |

### Feature Importance (Ridge, standardised coefficients)

![Average Absolute Coefficients](figures/avg_abs_coefficients.png)

![Coefficients Over Time](figures/coefficients_over_time.png)

## 8. Benchmark Comparison

| Series | Ann. Return | Ann. Vol | Sharpe | Beta to SPY | Alpha (ann.) | Information Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Ridge strategy (net) | -0.3867 | 0.2517 | -1.8139 | -0.2048 | -0.3430 | -1.8630 |
| SPY | 0.1745 | 0.1945 | 0.9245 | n/a | n/a | n/a |
| QQQ | 0.2341 | 0.2404 | 0.9958 | n/a | n/a | n/a |
| SMH | 0.4259 | 0.3601 | 1.1662 | n/a | n/a | n/a |
| IYT | 0.1194 | 0.2482 | 0.5787 | n/a | n/a | n/a |

Naive rule-based baselines (momentum on the same universe, short-term reversal on the same universe)
are included in the headline table in Section 7 for context -- the graph diffusion features are
compared against both passive benchmarks and simple systematic alternatives, not only against cash.

## 9. Robustness and Failure Cases

**Permutation test** (predictions vs. realised next-day residual return, 47100
observations, 200 permutations): observed Spearman IC =
-0.0033, p-value = 0.4627.
This does not clear conventional significance thresholds, i.e. the observed rank relationship between predictions and forward returns is statistically indistinguishable from a random reshuffling.

**Bootstrap Sharpe ratio (90% CI, 1000
resamples):** point estimate -1.81, CI [-2.39, -1.21].
The interval does not straddle zero, though see the permutation test and cost-sensitivity results before drawing conclusions.

**Parameter sensitivity (graph window x top-k, pooled IC):**

![Graph Parameter Sensitivity](figures/graph_param_sensitivity.png)

|  | pooled_ic | n_obs |
| --- | --- | --- |
| (30, 4) | -0.0011 | 71350 |
| (30, 8) | -0.0022 | 71350 |
| (30, 12) | -0.0030 | 71350 |
| (30, 16) | -0.0033 | 71350 |
| (60, 4) | -0.0074 | 70825 |
| (60, 8) | -0.0039 | 70825 |
| (60, 12) | -0.0023 | 70825 |
| (60, 16) | -0.0049 | 70825 |
| (90, 4) | -0.0058 | 70075 |
| (90, 8) | -0.0048 | 70075 |
| (90, 12) | -0.0060 | 70075 |
| (90, 16) | -0.0058 | 70075 |

**Known limitations and failure modes:**

* **Transaction costs matter a lot at this turnover.** Compare gross vs net Sharpe in Section 7 --
  average daily turnover of 1.95
  means even the default 5 bps assumption materially erodes returns, and
  the cost-sensitivity table shows the strategy's Sharpe ratio as costs rise toward realistic
  small-cap/short-borrow levels.
* **Regime instability.** The annual returns table shows the strategy's performance is not uniform
  across years; a graph-diffusion effect estimated on one volatility/correlation regime need not
  persist into another (e.g. correlations spike and diversify away in market-wide sell-offs, exactly
  when a diffusion signal might otherwise be most useful).
* **Data-snooping risk.** Feature and hyperparameter choices (graph window, top-k, lag windows) were
  set from domain reasoning and a single sensitivity sweep, not from an independent validation
  universe -- results should be treated as a research prototype, not a validated alpha.
* **Small universe.** 29 names across three sectors is not enough to diversify
  idiosyncratic risk in a long/short book with a 20%/
  20% quantile split; single-name moves can dominate.
* **Survivorship bias is not corrected.** The universe is today's list of liquid names in these
  sectors, not a point-in-time constituent list; any name that was delisted or became illiquid
  earlier in the sample is absent by construction.
* **Liquidity/shorting assumptions are simplified.** The backtest assumes shorting is always
  available at the assumed cost with no borrow fee, no market impact beyond the flat bps cost, and
  full fills at the close -- unrealistic for smaller-cap names in the universe (e.g. RBLX, BILI).
* **Institutional deployment would need:** point-in-time fundamentals and index membership, a
  proper borrow/locate feed, intraday execution data to model market impact and slippage more
  realistically, a much larger and more carefully constructed universe, and out-of-sample validation
  on data the researcher did not already see.

## 10. Conclusion

The graph diffusion signal, as implemented here, does not produce a convincing, cost-robust edge over the test period on this universe -- gross performance is thin and net performance is further eroded by transaction costs, and the permutation/bootstrap tests do not provide strong statistical support.
This is reported as a genuine research finding rather than reframed as a success: a rigorously
built, leakage-safe pipeline that finds a weak or inconclusive signal is a more useful (and more
honest) outcome than an inflated backtest. The engineering value of the project -- leakage-safe
feature construction, walk-forward validation, transaction-cost-aware evaluation, and statistical
robustness testing -- stands independently of whether this particular signal turns out to be
profitable on this particular universe and period.
