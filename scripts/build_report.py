#!/usr/bin/env python3
"""Build the Markdown + PDF research report from saved backtest results.

Reads everything from ``reports/results/`` (written by ``run_pipeline.py`` /
``graph_diffusion_signal.cli backtest``) so this script never re-runs the
backtest itself -- it only formats what was already computed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from graph_diffusion_signal import metrics as metrics_mod  # noqa: E402
from graph_diffusion_signal import plots as plots_mod  # noqa: E402
from graph_diffusion_signal import report as report_mod  # noqa: E402
from graph_diffusion_signal import cli as cli_mod  # noqa: E402

MODEL_NAMES = ["ridge", "elasticnet", "rank_signal", "naive_momentum", "naive_reversal"]
DISPLAY_NAMES = {
    "ridge": "Ridge (graph diffusion features)",
    "elasticnet": "ElasticNet (graph diffusion features)",
    "rank_signal": "Rank on diffusion score (no fitting)",
    "naive_momentum": "Naive momentum baseline",
    "naive_reversal": "Naive reversal baseline",
}


def _load_all(reports_dir: Path, uni_benchmarks: list[str]):
    results_dir = reports_dir / "results"
    results, benchmarks = cli_mod.load_backtest_results(results_dir, MODEL_NAMES, uni_benchmarks)
    summary = json.loads((results_dir / "summary.json").read_text())
    robustness = json.loads((results_dir / "robustness.json").read_text())
    cost_table = pd.read_csv(results_dir / "cost_sensitivity.csv")
    param_grid = pd.read_csv(results_dir / "graph_param_sensitivity.csv")
    year_table = pd.read_csv(results_dir / "year_by_year_ridge.csv", index_col=0)
    coef_path = results_dir / "ridge_coefficients_by_fold.csv"
    coef_by_fold = pd.read_csv(coef_path, index_col=0, parse_dates=True) if coef_path.exists() else None
    return results, benchmarks, summary, robustness, cost_table, param_grid, year_table, coef_by_fold


def build_headline_table(summary: dict, benchmark_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in MODEL_NAMES:
        if name not in summary:
            continue
        net, gross = summary[name]["net"], summary[name]["gross"]
        rows.append({
            "Strategy": DISPLAY_NAMES[name],
            "Net Ann. Return": net.get("Annualised Return"),
            "Net Ann. Vol": net.get("Annualised Volatility"),
            "Net Sharpe": net.get("Sharpe Ratio"),
            "Net Max DD": net.get("Max Drawdown"),
            "Gross Sharpe": gross.get("Sharpe Ratio"),
            "Avg Turnover": net.get("Avg Daily Turnover"),
        })
    for b in benchmark_names:
        key = f"benchmark_{b}"
        if key in summary:
            s = summary[key]
            rows.append({
                "Strategy": f"{b} (buy & hold)",
                "Net Ann. Return": s.get("Annualised Return"),
                "Net Ann. Vol": s.get("Annualised Volatility"),
                "Net Sharpe": s.get("Sharpe Ratio"),
                "Net Max DD": s.get("Max Drawdown"),
                "Gross Sharpe": s.get("Sharpe Ratio"),
                "Avg Turnover": None,
            })
    df = pd.DataFrame(rows).set_index("Strategy")
    return df


def build_benchmark_comparison_table(summary: dict, benchmark_names: list[str]) -> pd.DataFrame:
    ridge_net = summary["ridge"]["net"]
    rows = [{
        "Series": "Ridge strategy (net)",
        "Ann. Return": ridge_net.get("Annualised Return"),
        "Ann. Vol": ridge_net.get("Annualised Volatility"),
        "Sharpe": ridge_net.get("Sharpe Ratio"),
        "Beta to SPY": ridge_net.get("Beta to Benchmark"),
        "Alpha (ann.)": ridge_net.get("Annualised Alpha"),
        "Information Ratio": ridge_net.get("Information Ratio"),
    }]
    for b in benchmark_names:
        key = f"benchmark_{b}"
        if key in summary:
            s = summary[key]
            rows.append({
                "Series": b, "Ann. Return": s.get("Annualised Return"), "Ann. Vol": s.get("Annualised Volatility"),
                "Sharpe": s.get("Sharpe Ratio"), "Beta to SPY": None, "Alpha (ann.)": None, "Information Ratio": None,
            })
    return pd.DataFrame(rows).set_index("Series")


def main(args=None):
    reports_dir = Path(getattr(args, "reports_dir", ROOT / "reports"))
    config_dir = Path(getattr(args, "config_dir", ROOT / "config"))
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    from graph_diffusion_signal.config import UniverseConfig, BacktestConfig
    uni = UniverseConfig.from_yaml(config_dir / "universe.yaml")
    bt = BacktestConfig.from_yaml(config_dir / "backtest.yaml")

    (results, benchmarks, summary, robustness, cost_table, param_grid, year_table, coef_by_fold) = _load_all(
        reports_dir, uni.benchmarks
    )

    used_synthetic = summary.get("used_synthetic_data", False)

    # ---- Figures -----------------------------------------------------
    curve_series = {
        "Ridge (net)": results["ridge"]["net_returns"],
        "Ridge (gross)": results["ridge"]["gross_returns"],
        "Naive momentum (net)": results["naive_momentum"]["net_returns"],
        "Naive reversal (net)": results["naive_reversal"]["net_returns"],
    }
    for b, s in benchmarks.items():
        curve_series[f"{b} (buy & hold)"] = s
    plots_mod.plot_equity_curve(curve_series, figures_dir / "equity_curve.png",
                                 title="Equity Curve: Graph Diffusion Strategy vs Benchmarks" + (" [SYNTHETIC DATA]" if used_synthetic else ""))

    plots_mod.plot_drawdown(results["ridge"]["net_returns"], figures_dir / "drawdown.png",
                             title="Ridge Strategy Drawdown (net of costs)")

    plots_mod.plot_cost_sensitivity(cost_table, figures_dir / "cost_sensitivity.png")

    if coef_by_fold is not None and not coef_by_fold.empty:
        plots_mod.plot_coefficients(coef_by_fold, figures_dir / "coefficients_over_time.png")
        plots_mod.plot_avg_abs_coefficients(coef_by_fold.abs().mean().sort_values(),
                                             figures_dir / "avg_abs_coefficients.png")

    if not param_grid.empty:
        plots_mod.plot_parameter_sensitivity(
            param_grid, x_col="corr_window", y_col="pooled_ic", group_col="top_k",
            path=figures_dir / "graph_param_sensitivity.png",
            title="Pooled Spearman IC: Neighbour Diffusion Score vs Next-Day Residual Return",
        )

    yt = year_table.copy()
    yt.index.name = "Year"
    plots_mod.plot_year_by_year(yt, figures_dir / "year_by_year.png", title="Ridge Strategy: Annual Net Returns")

    # ---- Tables --------------------------------------------------------
    headline = build_headline_table(summary, uni.benchmarks)
    bench_table = build_benchmark_comparison_table(summary, uni.benchmarks)

    perm = robustness["permutation_test"]
    boot = robustness["bootstrap_sharpe_ci"]

    fmt = report_mod.fmt_pct
    fmtn = report_mod.fmt_num

    data_notice = ""
    if used_synthetic:
        data_notice = (
            "\n> **Data notice:** live Yahoo Finance data could not be reached from the "
            "environment this build ran in, so the results below were computed on a "
            "calibrated **synthetic** OHLCV dataset (see `data/raw/SYNTHETIC_DATA_NOTICE.txt`). "
            "The synthetic generator reproduces realistic market/sector correlation structure "
            "and volatility clustering but contains **no injected lead-lag or diffusion effect** "
            "between names -- it is a placebo dataset. Treat every number in this report as a "
            "demonstration that the pipeline runs correctly end-to-end and is leakage-safe, "
            "**not** as evidence for or against the underlying hypothesis in real markets. "
            "Running `make data` with a working internet connection fetches real data through "
            "the identical code path and will produce genuine results.\n"
        )

    md = f"""# Cross-Asset Graph Diffusion Signals for Equity Return Prediction

*A walk-forward equity research study across semiconductor, Japanese media/gaming, and transport/logistics equities.*
{data_notice}
## 1. Abstract

This project tests whether short-horizon information diffusion across economically connected
equities -- driven by supply-chain, thematic, and investor-flow linkages -- contains exploitable
predictive information for next-day residual equity returns. A rolling correlation graph is built
strictly from past returns, and lagged neighbour return / residual-shock features derived from
that graph are fed into interpretable linear models (Ridge, ElasticNet) inside a walk-forward
backtest with realistic transaction costs. The headline out-of-sample net Sharpe ratio for the
Ridge strategy over the test period is **{fmtn(summary['ridge']['net'].get('Sharpe Ratio'))}**,
versus **{fmtn(summary.get('benchmark_SPY', {}).get('Sharpe Ratio', float('nan')))}** for buy-and-hold SPY.
{"This is a weak/negative result and is reported honestly -- see Sections 9 and the failure-case discussion." if (summary['ridge']['net'].get('Sharpe Ratio') or 0) < 0.3 else "Results are modest and are discussed critically alongside their statistical significance below."}

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

* **Source:** `yfinance` (Yahoo Finance) adjusted daily OHLCV, {uni.start_date} to latest available.
* **Universe:** {len(uni.all_tickers)} configured tickers across three sectors (semiconductors/tech,
  Japanese media/gaming/IP, transport/logistics) plus benchmarks {", ".join(uni.benchmarks)}.
  See `config/universe.yaml` for the exact list.
* **Cleaning:** tickers with fewer than {uni.min_history_days} trading days of history are dropped
  (a data-sufficiency filter, not a survivorship-bias correction -- this project does **not** claim
  to be survivorship-bias-free; see Section 9). Trading calendars are aligned across the surviving
  universe; short (<=3 day) gaps are forward-filled, longer gaps are dropped.
* **Returns:** simple daily returns from adjusted close. Residual returns are computed via a rolling
  beta regression (window={bt['returns']['beta_window']} days, method="{bt['returns']['residual_method']}")
  re-estimated **using only data through t-1** at every date, so the beta applied at date t never saw
  date t's own return.

## 4. Feature Construction

All features are computed as of the close of a "feature date" `f` and are only ever paired with a
target return realised strictly after `f` (see `src/graph_diffusion_signal/features.py` module
docstring for the exact timing convention). The rolling correlation graph (window=
{bt['graph']['correlation_window']} days, top-k={bt['graph']['top_k']}, min |corr|=
{bt['graph']['min_abs_corr']}) is built from returns up to `f-1` only, keeps each node's strongest
positive-correlation neighbours, and is row-normalised. Feature families:

* Own lagged returns ({", ".join(str(w) + "d" for w in bt['features']['lag_windows'])})
* Rolling realised volatility ({bt['features']['vol_window']}d) and volume surprise (z-score vs
  {bt['features']['volume_window']}d rolling mean/std)
* Neighbour return diffusion score and neighbour residual-shock diffusion score (graph-weighted
  sums of neighbours' same-day returns/residuals)
* Graph weighted-degree centrality
* Sector benchmark return and broad market return
* Momentum ({bt['features']['momentum_window']}d) and short-term reversal ({bt['features']['reversal_window']}d)
* Volatility-normalised versions of the return and diffusion features

## 5. Walk-Forward Backtest Design

* Initial training window: {bt['walkforward']['train_years']} years, expanding thereafter.
* Refit frequency: {bt['walkforward']['refit_frequency']} (monthly="M", quarterly="Q").
* First eligible test date: {bt['walkforward']['test_start']}.
* At every refit boundary the model (StandardScaler + Ridge/ElasticNet, hyperparameters chosen by
  time-series cross-validation **inside the training window only**) is retrained from scratch on
  strictly past data and then used to predict every date up to the next refit boundary -- fully
  out-of-sample relative to that fold's training set.
* `tests/test_backtest_no_lookahead.py` verifies this directly, including a test that perturbs
  returns after a cutoff date by a large amount and checks that predictions dated on or before the
  cutoff are bit-for-bit unchanged.

## 6. Transaction Cost Model

Costs are charged as `cost_bps / 10,000 * sum(|weight change|)` at each rebalance, i.e. proportional
to gross traded notional. Default: {bt['costs']['default_bps']} bps one-way. Sensitivity is reported
at {", ".join(str(b) for b in bt['costs']['sensitivity_bps'])} bps.

## 7. Results

![Equity Curve](figures/equity_curve.png)

{report_mod.df_to_markdown_table(headline, float_format="{:.4f}")}

![Drawdown](figures/drawdown.png)

### Annual Returns (Ridge, net of costs)

![Annual Returns](figures/year_by_year.png)

{report_mod.df_to_markdown_table(year_table.rename_axis("Year"), float_format="{:.4f}")}

### Transaction Cost Sensitivity (Ridge)

![Cost Sensitivity](figures/cost_sensitivity.png)

{report_mod.df_to_markdown_table(cost_table.set_index("cost_bps"), float_format="{:.3f}")}

### Feature Importance (Ridge, standardised coefficients)

![Average Absolute Coefficients](figures/avg_abs_coefficients.png)

![Coefficients Over Time](figures/coefficients_over_time.png)

## 8. Benchmark Comparison

{report_mod.df_to_markdown_table(bench_table, float_format="{:.4f}")}

Naive rule-based baselines (momentum on the same universe, short-term reversal on the same universe)
are included in the headline table in Section 7 for context -- the graph diffusion features are
compared against both passive benchmarks and simple systematic alternatives, not only against cash.

## 9. Robustness and Failure Cases

**Permutation test** (predictions vs. realised next-day residual return, {perm.get('n_obs', 'n/a')}
observations, {bt['robustness']['n_permutations']} permutations): observed Spearman IC =
{fmtn(perm.get('observed_ic'), 4)}, p-value = {fmtn(perm.get('p_value'), 4)}.
{"This does not clear conventional significance thresholds, i.e. the observed rank relationship between predictions and forward returns is statistically indistinguishable from a random reshuffling." if (perm.get('p_value') or 1) > 0.05 else "This clears the conventional p<0.05 threshold, though a single permutation test on one universe/period is not sufficient evidence of a durable edge -- see caveats below."}

**Bootstrap Sharpe ratio ({int(boot.get('ci', 0.9) * 100)}% CI, {bt['robustness']['n_bootstrap']}
resamples):** point estimate {fmtn(boot.get('sharpe'))}, CI [{fmtn(boot.get('lower'))}, {fmtn(boot.get('upper'))}].
{"The interval straddles zero, so we cannot reject the possibility that the true out-of-sample Sharpe ratio is zero or negative." if (boot.get('lower') or -1) < 0 < (boot.get('upper') or 1) else "The interval does not straddle zero, though see the permutation test and cost-sensitivity results before drawing conclusions."}

**Parameter sensitivity (graph window x top-k, pooled IC):**

![Graph Parameter Sensitivity](figures/graph_param_sensitivity.png)

{report_mod.df_to_markdown_table(param_grid.set_index(['corr_window','top_k']) if not param_grid.empty else param_grid, float_format="{:.4f}")}

**Known limitations and failure modes:**

* **Transaction costs matter a lot at this turnover.** Compare gross vs net Sharpe in Section 7 --
  average daily turnover of {fmtn(headline.loc[DISPLAY_NAMES['ridge'], 'Avg Turnover'] if DISPLAY_NAMES['ridge'] in headline.index else float('nan'))}
  means even the default {bt['costs']['default_bps']} bps assumption materially erodes returns, and
  the cost-sensitivity table shows the strategy's Sharpe ratio as costs rise toward realistic
  small-cap/short-borrow levels.
* **Regime instability.** The annual returns table shows the strategy's performance is not uniform
  across years; a graph-diffusion effect estimated on one volatility/correlation regime need not
  persist into another (e.g. correlations spike and diversify away in market-wide sell-offs, exactly
  when a diffusion signal might otherwise be most useful).
* **Data-snooping risk.** Feature and hyperparameter choices (graph window, top-k, lag windows) were
  set from domain reasoning and a single sensitivity sweep, not from an independent validation
  universe -- results should be treated as a research prototype, not a validated alpha.
* **Small universe.** {len(uni.all_tickers)} names across three sectors is not enough to diversify
  idiosyncratic risk in a long/short book with a {bt['portfolio']['long_quantile']:.0%}/
  {bt['portfolio']['short_quantile']:.0%} quantile split; single-name moves can dominate.
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

{"The graph diffusion signal, as implemented here, does not produce a convincing, cost-robust edge over the test period on this universe -- gross performance is thin and net performance is further eroded by transaction costs, and the permutation/bootstrap tests do not provide strong statistical support." if (summary['ridge']['net'].get('Sharpe Ratio') or 0) < 0.3 else "The graph diffusion signal shows a modest edge over the test period; the permutation and bootstrap results provide some, though not overwhelming, statistical support, and the effect is materially reduced (though not eliminated) by realistic transaction costs."}
This is reported as a genuine research finding rather than reframed as a success: a rigorously
built, leakage-safe pipeline that finds a weak or inconclusive signal is a more useful (and more
honest) outcome than an inflated backtest. The engineering value of the project -- leakage-safe
feature construction, walk-forward validation, transaction-cost-aware evaluation, and statistical
robustness testing -- stands independently of whether this particular signal turns out to be
profitable on this particular universe and period.
"""

    md_path = reports_dir / "quant_research_report.md"
    report_mod.write_markdown(md, md_path)

    pdf_path = reports_dir / "quant_research_report.pdf"
    success, method = report_mod.build_pdf(md_path, pdf_path)
    print(f"Report written to {md_path}")
    if success:
        print(f"PDF written to {pdf_path} (method={method})")
    else:
        print(f"PDF generation failed. To build it manually once pandoc + pdflatex are installed, run:\n"
              f"  pandoc {md_path} -o {pdf_path} --pdf-engine=pdflatex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    main(parser.parse_args())
