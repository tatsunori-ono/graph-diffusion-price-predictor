import numpy as np
import pandas as pd
import pytest

from graph_diffusion_signal import metrics


def _flat_positive_returns(n=252, daily=0.0004):
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(daily, index=dates)


def test_annualised_return_of_constant_daily_return():
    r = _flat_positive_returns(252, 0.0004)
    ar = metrics.annualised_return(r)
    expected = (1.0004) ** 252 - 1
    assert ar == pytest.approx(expected, rel=1e-6)


def test_sharpe_ratio_zero_vol_is_nan():
    r = _flat_positive_returns(100, 0.0)
    assert np.isnan(metrics.sharpe_ratio(r))


def test_max_drawdown_is_nonpositive():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=500)
    r = pd.Series(rng.normal(0, 0.01, 500), index=dates)
    mdd = metrics.max_drawdown(r)
    assert mdd <= 0


def test_max_drawdown_known_path():
    dates = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([0.10, -0.20, 0.05, 0.0], index=dates)
    # curve: 1.10, 0.88, 0.924, 0.924 -> peak 1.10 -> trough 0.88 -> dd = -0.2
    mdd = metrics.max_drawdown(r)
    assert mdd == pytest.approx(-0.2, abs=1e-9)


def test_hit_rate_bounds():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=300)
    r = pd.Series(rng.normal(0, 0.01, 300), index=dates)
    hr = metrics.hit_rate(r)
    assert 0 <= hr <= 1


def test_beta_alpha_regression_recovers_known_beta():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=1000)
    bench = pd.Series(rng.normal(0, 0.01, 1000), index=dates)
    true_beta = 0.6
    strat = true_beta * bench + rng.normal(0, 0.001, 1000)
    strat = pd.Series(strat.values, index=dates)
    beta, alpha, r2 = metrics.beta_alpha_vs_benchmark(strat, bench)
    assert beta == pytest.approx(true_beta, abs=0.05)
    assert r2 > 0.8


def test_permutation_test_null_case_high_pvalue():
    rng = np.random.default_rng(4)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    preds = pd.Series(rng.normal(0, 1, n), index=dates)
    fwd = pd.Series(rng.normal(0, 1, n), index=dates)  # independent of preds
    result = metrics.permutation_test_signal(preds, fwd, n_permutations=100, seed=0)
    assert result["p_value"] > 0.05


def test_permutation_test_detects_real_signal():
    rng = np.random.default_rng(5)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    preds = pd.Series(rng.normal(0, 1, n), index=dates)
    fwd = pd.Series(0.9 * preds.values + rng.normal(0, 0.2, n), index=dates)
    result = metrics.permutation_test_signal(preds, fwd, n_permutations=200, seed=0)
    assert result["p_value"] < 0.05
    assert result["observed_ic"] > 0.5


def test_bootstrap_sharpe_ci_contains_point_estimate():
    r = _flat_positive_returns(400, 0.0003)
    rng_r = pd.Series(np.random.default_rng(6).normal(0.0003, 0.01, 400), index=r.index)
    result = metrics.bootstrap_sharpe_ci(rng_r, n_bootstrap=500, seed=0)
    assert result["lower"] <= result["sharpe"] <= result["upper"]
