import numpy as np
import pandas as pd
import pytest

from graph_diffusion_signal.data import generate_synthetic_universe
from graph_diffusion_signal.features import (
    FeatureParams,
    build_feature_panel,
    compute_residual_returns,
)
from graph_diffusion_signal.graph import GraphParams, rolling_graphs


@pytest.fixture(scope="module")
def small_panel_inputs():
    tickers = ["A1", "A2", "A3", "B1", "B2", "B3"]
    sector_of = {"A1": "sec_a", "A2": "sec_a", "A3": "sec_a", "B1": "sec_b", "B2": "sec_b", "B3": "sec_b"}
    benchmarks = ["SPY", "BENCH_A", "BENCH_B"]
    panels = generate_synthetic_universe(
        tickers=tickers, sector_of=sector_of, benchmarks=benchmarks,
        start="2020-01-01", end="2021-06-30", seed=7,
    )
    prices = pd.DataFrame({t: panels[t]["Adj Close"] for t in tickers}).dropna()
    volumes = pd.DataFrame({t: panels[t]["Volume"] for t in tickers}).reindex(prices.index)
    returns_full = pd.DataFrame({t: panels[t]["Adj Close"] for t in tickers + benchmarks}).pct_change().dropna()
    returns_universe = returns_full[tickers]
    market_returns = returns_full["SPY"]
    sector_returns = {"sec_a": returns_full["BENCH_A"], "sec_b": returns_full["BENCH_B"]}
    return {
        "tickers": tickers,
        "sector_of": sector_of,
        "prices": prices.loc[returns_universe.index[0]:],
        "volumes": volumes.loc[returns_universe.index[0]:],
        "returns_universe": returns_universe,
        "market_returns": market_returns,
        "sector_returns": sector_returns,
    }


def test_residual_returns_shape_and_finite(small_panel_inputs):
    d = small_panel_inputs
    params = FeatureParams(beta_window=40, min_beta_window=20, residual_method="sector")
    resid = compute_residual_returns(d["returns_universe"], d["market_returns"], d["sector_of"], d["sector_returns"], params)
    assert resid.shape == d["returns_universe"].shape
    # Early rows before min_beta_window should be NaN (no estimate yet).
    assert resid.iloc[0].isna().all()
    # Later rows should be finite.
    assert resid.iloc[-1].notna().any()


def test_residual_beta_uses_only_past_data(small_panel_inputs):
    """Perturbing returns strictly AFTER date t must not change the residual
    computed AT date t (rolling beta is shifted by one day)."""
    d = small_panel_inputs
    params = FeatureParams(beta_window=40, min_beta_window=20, residual_method="market")

    resid_original = compute_residual_returns(
        d["returns_universe"], d["market_returns"], d["sector_of"], d["sector_returns"], params
    )

    cutoff = d["returns_universe"].index[150]
    perturbed_returns = d["returns_universe"].copy()
    perturbed_market = d["market_returns"].copy()
    future_mask = perturbed_returns.index > cutoff
    perturbed_returns.loc[future_mask] = perturbed_returns.loc[future_mask] + 0.5
    perturbed_market.loc[future_mask] = perturbed_market.loc[future_mask] + 0.5

    resid_perturbed = compute_residual_returns(
        perturbed_returns, perturbed_market, d["sector_of"], d["sector_returns"], params
    )

    before = resid_original.loc[:cutoff]
    before_perturbed = resid_perturbed.loc[:cutoff]
    pd.testing.assert_frame_equal(before, before_perturbed)


def test_feature_panel_has_no_same_day_target_leak(small_panel_inputs):
    d = small_panel_inputs
    fparams = FeatureParams(beta_window=40, min_beta_window=20, residual_method="market", lag_windows=[1, 3, 5, 20])
    resid = compute_residual_returns(d["returns_universe"], d["market_returns"], d["sector_of"], d["sector_returns"], fparams)
    gparams = GraphParams(window=40, top_k=2, min_abs_corr=-1.0)
    graphs = rolling_graphs(d["returns_universe"], gparams)

    panel = build_feature_panel(
        prices_wide=d["prices"],
        volumes_wide=d["volumes"],
        returns_raw=d["returns_universe"],
        residual_returns=resid,
        market_returns=d["market_returns"],
        sector_of=d["sector_of"],
        sector_returns=d["sector_returns"],
        graphs=graphs,
        params=fparams,
    )
    assert not panel.empty

    dates = d["returns_universe"].index
    for (f, ticker), row in panel.sample(min(30, len(panel)), random_state=0).iterrows():
        loc = dates.get_loc(f)
        expected_target = resid.iloc[loc + 1][ticker]
        if pd.notna(row["target_1d"]) and pd.notna(expected_target):
            assert row["target_1d"] == pytest.approx(expected_target)


def test_feature_panel_columns_present(small_panel_inputs):
    d = small_panel_inputs
    fparams = FeatureParams(beta_window=40, min_beta_window=20, residual_method="market")
    resid = compute_residual_returns(d["returns_universe"], d["market_returns"], d["sector_of"], d["sector_returns"], fparams)
    gparams = GraphParams(window=40, top_k=2, min_abs_corr=-1.0)
    graphs = rolling_graphs(d["returns_universe"], gparams)
    panel = build_feature_panel(
        d["prices"], d["volumes"], d["returns_universe"], resid, d["market_returns"],
        d["sector_of"], d["sector_returns"], graphs, fparams,
    )
    from graph_diffusion_signal.features import FEATURE_COLUMNS

    for col in FEATURE_COLUMNS:
        assert col in panel.columns
