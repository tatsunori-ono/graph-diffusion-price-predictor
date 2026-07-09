# Cross-Asset Graph Diffusion Signals for Equity Return Prediction

A leakage-safe, walk-forward equity research pipeline that tests whether information shocks
diffuse across economically connected companies -- semiconductors, Japanese media/gaming/IP, and
transport/logistics -- with a short, exploitable delay.

> **This build's example results used synthetic data, not live markets.** The sandbox this
> repository was built in could not reach Yahoo Finance, so `make data` automatically fell back to
> a calibrated synthetic OHLCV generator (see `data/raw/SYNTHETIC_DATA_NOTICE.txt` and
> [Data notice](#data-notice) below). The code path for real data is identical and untouched --
> run `make data` yourself with a working connection to get genuine results. Every number in
> [Headline Results](#headline-results) below is from that synthetic placebo run and should be read
> as "the pipeline works end-to-end and is leakage-safe," not as a claim about real markets.

## 1. Project Summary

This repository builds a rolling correlation graph over a ~29-stock universe (semiconductors/tech,
Japanese media/gaming/IP, transport/logistics) plus four sector benchmarks, engineers graph-aware
features (neighbour return/residual diffusion scores, weighted-degree centrality) alongside standard
own-return, volatility, and momentum/reversal features, and feeds them into interpretable linear
models (Ridge, ElasticNet) and a zero-parameter rank baseline inside a walk-forward backtest with
realistic transaction costs. Every step -- graph construction, feature timing, model refitting,
portfolio weighting -- is built to a strict no-look-ahead discipline, checked directly by an
automated leakage test (`tests/test_backtest_no_lookahead.py`) that perturbs future returns and
confirms past predictions never change. The honest headline finding on the (synthetic, placebo)
data available in this environment: gross signal is weak and does **not** survive realistic
transaction costs -- reported as-is rather than dressed up.

## 2. Why This Hypothesis Could Exist

Companies in the same sector, supply chain, or thematic bucket share exposure to common shocks (a
foundry capacity update, a console hardware refresh, a fuel-price move), but the market doesn't
necessarily price that shared exposure into every related name at the same instant. A name with
lower analyst coverage, lower liquidity, or a less obvious read-through to the news event may react
with a lag as flows, arbitrage desks, and index/ETF rebalancing slowly connect the dots. A **rolling
correlation graph** (rather than a static sector label) is used because economic linkages between
these companies are not fixed -- supply-chain relationships, thematic overlap ("AI capex" pulling
semiconductor and hyperscaler-adjacent names together), and investor-flow correlation all drift over
time. If a short diffusion delay exists and is even weakly persistent, a stock's lagged neighbour
returns (from the graph) should carry information about its own near-future **residual** return
(i.e. after stripping out market and sector beta).

## 3. Methodology / Pipeline

```
yfinance (or cached CSV / synthetic fallback)
        |
   data.py  --> clean + align trading calendar, drop low-history tickers
        |
   features.py --> rolling-beta residual returns (t-1 betas only)
        |
   graph.py  --> rolling correlation graph, window ends at t-1, top-k positive edges, row-normalised
        |
   features.py --> own returns, vol, volume surprise, neighbour diffusion/shock scores,
        |          centrality, sector/market return, momentum/reversal  [all known by close of f]
        |
   backtest.py --> walk-forward: train on date < boundary, predict [boundary, next boundary)
        |          models: Ridge / ElasticNet (StandardScaler refit per fold) / rank-on-diffusion-score
        |
   portfolio.py --> rank cross-section, dollar-neutral long/short quantiles, position cap, turnover
        |
   metrics.py / plots.py / report.py --> Sharpe/Sortino/drawdown/alpha/IR, permutation + bootstrap
                                          tests, cost sensitivity, parameter sensitivity, PDF report
```

Full timing/leakage conventions are documented in the module docstrings of `graph.py`,
`features.py`, and `backtest.py`, and enforced by `tests/test_backtest_no_lookahead.py`.

## 4. Setup

Requires Python 3.11+ (developed/tested against 3.10 as well). No paid data APIs.

```bash
git clone <this-repo>
cd graph-diffusion-signals
make install     # pip install -e ".[dev]"
```

## 5. Reproduce

```bash
make data        # download (or synthesize, if offline) + clean OHLCV, cache to data/raw and data/processed
make backtest     # feature engineering + walk-forward backtest + robustness stats -> reports/results/
make report       # build reports/quant_research_report.md and .pdf from saved results
make test         # run the pytest suite, including the no-look-ahead tests
make smoke        # fast (~seconds) end-to-end sanity check on a tiny universe/date range
```

Or via the CLI directly: `python -m graph_diffusion_signal.cli {data,backtest,report}`.

## 6. Headline Results

*(Synthetic/placebo data -- see the notice at the top of this file. Test period 2019-01-01 onward,
quarterly walk-forward refit, 5 bps one-way transaction costs.)*

| Strategy | Net Ann. Return | Net Sharpe | Gross Sharpe | Avg Daily Turnover |
| --- | --- | --- | --- | --- |
| Ridge (graph diffusion features) | -10.4% | -0.18 | 0.56 | 1.87x |
| ElasticNet (graph diffusion features) | -18.9% | -0.48 | 0.32 | 2.07x |
| Rank on diffusion score (no fitting) | -21.2% | -0.60 | 0.38 | 2.43x |
| Naive momentum baseline | -19.4% | -0.52 | -0.28 | 0.60x |
| Naive reversal baseline | -12.8% | -0.25 | 0.18 | 1.12x |
| SPY buy & hold | +8.3% | 0.40 | -- | -- |
| SMH buy & hold | +23.7% | 0.65 | -- | -- |

Permutation test on the Ridge signal: Spearman IC = 0.006, p = 0.15 (not significant). Bootstrap 90%
Sharpe CI: [-0.76, 0.40] (straddles zero). **Conclusion on this synthetic run: no cost-robust edge.**
Gross Sharpe is positive but modest, and is fully consumed (and reversed) by transaction costs at
realistic turnover -- this is reported honestly as the finding, not reframed as a success. Full
tables, the annual-returns breakdown, and the cost/parameter sensitivity sweeps are in
[`reports/quant_research_report.pdf`](reports/quant_research_report.pdf).

## 7. Key Plots

See `reports/figures/`: `equity_curve.png`, `drawdown.png`, `year_by_year.png`,
`cost_sensitivity.png`, `avg_abs_coefficients.png`, `coefficients_over_time.png`,
`graph_param_sensitivity.png`.

## 8. Limitations

* **Synthetic data in this environment.** See the notice at the top -- results here demonstrate
  pipeline correctness, not a real-market finding.
* **No survivorship-bias correction.** The universe is today's liquid names in each sector, not a
  point-in-time constituent list.
* **Small universe (29 names).** Not enough breadth to diversify idiosyncratic risk in a 20%/20%
  quantile long/short book.
* **Simplified execution assumptions.** Flat bps cost, no borrow fee, full fills at the close, no
  market-impact model -- unrealistic for smaller-cap names in the universe.
* **Data-snooping risk.** Graph window, top-k, and lag choices came from domain reasoning plus one
  sensitivity sweep, not an independent validation universe.
* Full discussion in `reports/quant_research_report.pdf`, Section 9.

## 9. What This Demonstrates (for Quant Recruiters)

* **Independent signal research:** a specific, falsifiable hypothesis (short-horizon cross-asset
  diffusion via a dynamic correlation graph), motivated by real sector structure rather than
  data-mined post hoc.
* **Statistical rigor:** permutation testing, bootstrap confidence intervals, year-by-year and
  cost-sensitivity breakdowns, and an explicit distinction between gross and net performance.
* **No look-ahead bias, verifiably:** every timing convention is documented and directly tested --
  including a test that perturbs future data and checks past predictions are provably unchanged.
* **Honest reporting under a weak result:** the signal does not survive costs on the available data,
  and the report says so plainly instead of cherry-picking a favourable cut.
* **Software engineering standards:** modular package (`data` / `features` / `graph` / `models` /
  `portfolio` / `backtest` / `metrics` / `plots` / `report` / `cli`), typed dataclasses for config,
  a CLI, a Makefile, unit tests (including a dedicated leakage-correctness suite), deterministic
  seeding, logging instead of print spam, and a reproducible PDF report pipeline.

## 10. Possible CV Bullet

See [CV Bullet Options](#cv-bullet-options) below.

---

## Interview Talking Points

**Why graph diffusion, specifically?** Static sector labels are a blunt instrument -- the actual
economic linkages between, say, a GPU maker and a game publisher shift over time (a console cycle,
an AI capex wave, a shared supplier). A rolling correlation graph adapts to that instead of assuming
a fixed taxonomy, while staying simple enough to compute, inspect, and reason about (row-normalised
adjacency, top-k positive edges, no learned graph structure to overfit).

**How I avoided look-ahead bias.** Three separate mechanisms, each documented and tested: (1) the
correlation graph "for date t" is built only from returns through t-1; (2) rolling-beta residuals
are estimated on a window ending at t-1 and only then applied to t's realised factor return; (3) the
walk-forward loop retrains from scratch at every refit boundary using only rows with `date <
boundary`. `tests/test_backtest_no_lookahead.py::test_perturbing_future_returns_does_not_change_past_predictions`
is the test I'd walk an interviewer through first: it builds two datasets identical up to a cutoff
and diverging sharply after it, and asserts every prediction dated on or before the cutoff is
bit-for-bit identical between the two runs.

**Why walk-forward validation matters here.** A single train/test split would let me pick a
favourable window; walk-forward with quarterly refits forces the model to prove itself repeatedly
across changing correlation and volatility regimes, and the year-by-year table shows exactly how
unstable that performance is (strongly negative in 2021 and 2023, strongly positive in 2024-2026 in
this synthetic run) -- instability a single split would have hidden.

**How transaction costs changed the result.** Gross Sharpe for the Ridge strategy was a modest but
positive ~0.56; at the default 5 bps one-way cost assumption net Sharpe flips to about -0.18, and it
degrades further (to roughly -3 at 25 bps) as costs rise. The culprit is turnover: an average daily
turnover near 1.9x (i.e. the book is nearly fully re-traded most days) at even modest per-trade
costs adds up fast. This is the single most important number in the whole report for judging
whether a signal is real-world-deployable, and it's why the headline table always shows gross and
net side by side.

**What failed.** The signal, as built, does not survive transaction costs on the data available in
this environment, and the permutation test (p ~= 0.15) and bootstrap Sharpe CI (straddling zero) do
not provide statistical support for a durable edge. The rank-only baseline (no model fitting at all)
performed similarly to the fitted models, which is itself informative -- it suggests whatever weak
structure exists in the diffusion feature isn't something the linear models are adding much value on
top of.

**What I'd improve with institutional data.** Point-in-time index membership and fundamentals to
remove survivorship bias properly; a real borrow/locate feed and market-impact model instead of a
flat bps assumption; intraday data to test whether the diffusion delay is sub-daily (this daily-bar
design could easily be missing a same-day effect); and a materially larger universe so the graph has
enough breadth to find genuine structure instead of noise.

**How this relates to quant research vs. quant development.** The research side is the hypothesis,
the feature design, and the honest statistical evaluation of whether it holds up. The engineering
side -- equally important here -- is making every timing assumption explicit and mechanically
testable, building a pipeline that a second person could pick up and extend (typed configs, a CLI,
modular stages, checkpoint-able intermediate outputs), and treating "no leakage" as a correctness
property to be tested, not just asserted in a comment.

## CV Bullet Options

1. **Quant Research version:** Designed and tested a cross-asset graph-diffusion hypothesis for
   equity residual returns across semiconductor, Japanese media/gaming, and transport/logistics
   equities; built a walk-forward backtest with permutation testing, bootstrap confidence intervals,
   and cost-sensitivity analysis, and reported a rigorously evaluated (and honestly negative-net)
   result rather than an overfit backtest.

2. **Quant Developer version:** Engineered a modular, leakage-safe Python research pipeline (rolling
   graph construction, feature engineering, walk-forward backtesting, transaction-cost modelling)
   with a typed configuration system, CLI, automated PDF reporting, and a dedicated test suite that
   directly verifies no-look-ahead correctness by perturbing future data and checking past
   predictions are unchanged.

3. **Investment Banking / Tech Sector Research version:** Built an independent research project
   analysing information diffusion across semiconductor supply chains, Japanese gaming/media IP
   names, and transport/logistics operators, combining sector domain knowledge with a systematic,
   cost-aware backtesting framework and clear benchmark comparisons (SPY, QQQ, SMH, IYT).

## Repository Structure

```
config/            universe.yaml, backtest.yaml
data/               raw/ (cached OHLCV or synthetic fallback), processed/ (cleaned wide panels)
notebooks/          01_research_exploration.ipynb
reports/            quant_research_report.{md,pdf}, figures/, results/ (intermediate CSV/JSON)
src/graph_diffusion_signal/   data, features, graph, models, portfolio, backtest, metrics, plots, report, cli
scripts/            run_pipeline.py, build_report.py, smoke_test.py
tests/              test_features.py, test_graph.py, test_backtest_no_lookahead.py, test_metrics.py, test_portfolio.py
```

## License

MIT.
