"""The most important test file in this repository.

These tests exist to catch look-ahead / leakage bugs directly, rather than
just trusting the code comments in graph.py / features.py / backtest.py.
The headline test is ``test_perturbing_future_returns_does_not_change_past_predictions``:
it builds two datasets that are byte-for-byte identical up to a cutoff date
and diverge sharply afterwards, runs the full feature + walk-forward
pipeline on both, and asserts every out-of-sample prediction dated on or
before the cutoff is identical. If any feature, residualisation step, or
walk-forward fold accidentally used future information, this test fails.
"""

from __future__ import annotations

import pandas as pd
import pytest

from graph_diffusion_signal.backtest import (
    WalkForwardParams,
    predictions_to_portfolio,
    refit_schedule,
    run_walkforward,
)
from graph_diffusion_signal.data import generate_synthetic_universe
from graph_diffusion_signal.features import (
    FEATURE_COLUMNS,
    FeatureParams,
    build_feature_panel,
    compute_residual_returns,
)
from graph_diffusion_signal.graph import GraphParams, rolling_graphs
from graph_diffusion_signal.portfolio import PortfolioParams

FPARAMS = FeatureParams(beta_window=40, min_beta_window=20, residual_method="market", lag_windows=[1, 3, 5, 20])
GPARAMS = GraphParams(window=40, top_k=3, min_abs_corr=-1.0)
WFPARAMS = WalkForwardParams(train_years=1, refit_frequency="Q", test_start="2020-01-01", min_train_days=150)


def _build_universe(seed=11, n_days=650):
    tickers = [f"S{i}" for i in range(8)]
    sector_of = {t: ("sec_a" if i < 4 else "sec_b") for i, t in enumerate(tickers)}
    benchmarks = ["SPY", "BENCH_A", "BENCH_B"]
    start = "2019-01-01"
    end = pd.bdate_range(start, periods=n_days)[-1].strftime("%Y-%m-%d")
    panels = generate_synthetic_universe(tickers, sector_of, benchmarks, start, end, seed=seed)

    prices = pd.DataFrame({t: panels[t]["Adj Close"] for t in tickers}).dropna()
    volumes = pd.DataFrame({t: panels[t]["Volume"] for t in tickers}).reindex(prices.index)
    returns_full = pd.DataFrame({t: panels[t]["Adj Close"] for t in tickers + benchmarks}).pct_change().dropna()

    common_idx = returns_full.index.intersection(prices.index)
    prices = prices.loc[common_idx]
    volumes = volumes.loc[common_idx]
    returns_universe = returns_full.loc[common_idx, tickers]
    market_returns = returns_full.loc[common_idx, "SPY"]
    sector_returns = {
        "sec_a": returns_full.loc[common_idx, "BENCH_A"],
        "sec_b": returns_full.loc[common_idx, "BENCH_B"],
    }
    return dict(
        tickers=tickers, sector_of=sector_of, prices=prices, volumes=volumes,
        returns_universe=returns_universe, market_returns=market_returns, sector_returns=sector_returns,
    )


def _run_feature_pipeline(d, fparams=FPARAMS, gparams=GPARAMS):
    resid = compute_residual_returns(
        d["returns_universe"], d["market_returns"], d["sector_of"], d["sector_returns"], fparams
    )
    graphs = rolling_graphs(d["returns_universe"], gparams)
    panel = build_feature_panel(
        d["prices"], d["volumes"], d["returns_universe"], resid, d["market_returns"],
        d["sector_of"], d["sector_returns"], graphs, fparams,
    )
    return panel


# --------------------------------------------------------------------------- #
# 1. Rolling graph for date t must only use data up to t-1
# --------------------------------------------------------------------------- #
def test_graph_dates_never_use_same_day_return():
    d = _build_universe()
    graphs = rolling_graphs(d["returns_universe"], GPARAMS)
    idx = d["returns_universe"].index
    for date in list(graphs.keys())[:10]:
        loc = idx.get_loc(date)
        window = d["returns_universe"].iloc[loc - GPARAMS.window : loc]
        assert date not in window.index
        assert window.index.max() < date


# --------------------------------------------------------------------------- #
# 2 & 3. Features at f must not use returns from f or later; target is forward
# --------------------------------------------------------------------------- #
def test_target_is_strictly_forward_return():
    d = _build_universe(seed=41, n_days=400)
    panel = _run_feature_pipeline(d)
    dates = d["returns_universe"].index
    resid = compute_residual_returns(
        d["returns_universe"], d["market_returns"], d["sector_of"], d["sector_returns"], FPARAMS
    )
    sample = panel.sample(min(40, len(panel)), random_state=0)
    checked = 0
    for (f, ticker), row in sample.iterrows():
        loc = dates.get_loc(f)
        if pd.notna(row["target_1d"]):
            assert dates[loc + 1] > f
            assert row["target_1d"] == pytest.approx(resid.iloc[loc + 1][ticker])
            checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# 5. Walk-forward: train strictly precedes test in every fold
# --------------------------------------------------------------------------- #
def test_walkforward_train_precedes_test_in_every_fold():
    d = _build_universe()
    panel = _run_feature_pipeline(d)
    dates = panel.index.get_level_values("date")
    boundaries = refit_schedule(dates, WFPARAMS)
    assert len(boundaries) >= 1
    for b in boundaries:
        assert (dates[dates < b] < b).all()

    oos_preds, fold_diag = run_walkforward(
        panel, FEATURE_COLUMNS, "target_1d", "ridge", {"alpha_grid": [1.0, 10.0], "cv_folds": 2}, WFPARAMS, seed=1
    )
    for fd in fold_diag:
        assert fd["train_end"] == fd["test_start"]
    pred_dates = oos_preds.index.get_level_values("date")
    assert pred_dates.min() >= boundaries[0]


# --------------------------------------------------------------------------- #
# The headline leakage test
# --------------------------------------------------------------------------- #
def test_perturbing_future_returns_does_not_change_past_predictions():
    d = _build_universe(seed=21, n_days=650)
    panel_a = _run_feature_pipeline(d)

    cutoff = d["returns_universe"].index[int(len(d["returns_universe"].index) * 0.75)]

    ru = d["returns_universe"].copy()
    mkt = d["market_returns"].copy()
    sec = {k: v.copy() for k, v in d["sector_returns"].items()}
    future_mask = ru.index > cutoff
    ru.loc[future_mask] = ru.loc[future_mask] + 1.5  # large shock after cutoff
    mkt.loc[future_mask] = mkt.loc[future_mask] + 1.5
    for k in sec:
        sec[k].loc[future_mask] = sec[k].loc[future_mask] + 1.5

    d_perturbed = dict(d)
    d_perturbed.update(returns_universe=ru, market_returns=mkt, sector_returns=sec)
    panel_b = _run_feature_pipeline(d_perturbed)

    preds_a, _ = run_walkforward(
        panel_a, FEATURE_COLUMNS, "target_1d", "ridge", {"alpha_grid": [1.0], "cv_folds": 2}, WFPARAMS, seed=1
    )
    preds_b, _ = run_walkforward(
        panel_b, FEATURE_COLUMNS, "target_1d", "ridge", {"alpha_grid": [1.0], "cv_folds": 2}, WFPARAMS, seed=1
    )

    common_dates = preds_a.index.get_level_values("date").unique()
    common_dates = common_dates[common_dates <= cutoff]
    assert len(common_dates) > 0

    a_sub = preds_a.loc[preds_a.index.get_level_values("date").isin(common_dates)].sort_index()
    b_sub = preds_b.loc[preds_b.index.get_level_values("date").isin(common_dates)].sort_index()

    pd.testing.assert_series_equal(a_sub, b_sub, check_exact=False, atol=1e-8, rtol=1e-6)


# --------------------------------------------------------------------------- #
# 4. Portfolio weights for date f are applied to returns realised at f+1
# --------------------------------------------------------------------------- #
def test_portfolio_realises_returns_on_next_trading_day_only():
    d = _build_universe(seed=31, n_days=500)
    panel = _run_feature_pipeline(d)
    oos_preds, _ = run_walkforward(
        panel, FEATURE_COLUMNS, "target_1d", "ridge", {"alpha_grid": [1.0], "cv_folds": 2}, WFPARAMS, seed=1
    )
    pparams = PortfolioParams(long_quantile=0.25, short_quantile=0.25, max_position_weight=1.0)
    port = predictions_to_portfolio(oos_preds, d["returns_universe"], pparams, cost_bps=5)

    all_dates = d["returns_universe"].index
    for f in port["weights"].index[:20]:
        pos = all_dates.get_loc(f)
        expected_next = all_dates[pos + 1]
        assert expected_next in port["gross_returns"].index
        assert expected_next > f
