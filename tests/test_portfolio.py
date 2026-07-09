import numpy as np
import pandas as pd
import pytest

from graph_diffusion_signal.portfolio import (
    PortfolioParams,
    apply_transaction_costs,
    compute_turnover,
    construct_weights,
    exposures,
)


def test_construct_weights_dollar_neutral():
    preds = pd.Series([5, 4, 3, 2, 1, 0, -1, -2, -3, -4], index=[f"T{i}" for i in range(10)])
    params = PortfolioParams(long_quantile=0.2, short_quantile=0.2, max_position_weight=1.0)
    w = construct_weights(preds, params)
    assert w.sum() == pytest.approx(0.0, abs=1e-9)
    assert (w > 0).sum() == 2
    assert (w < 0).sum() == 2


def test_construct_weights_respects_max_position_cap():
    preds = pd.Series([5, 4, 3, 2, 1, 0, -1, -2, -3, -4], index=[f"T{i}" for i in range(10)])
    params = PortfolioParams(long_quantile=0.2, short_quantile=0.2, max_position_weight=0.1)
    w = construct_weights(preds, params)
    assert w.abs().max() <= 0.1 + 1e-9


def test_construct_weights_picks_correct_names():
    preds = pd.Series([10, 9, 8, 1, 0, -1, -8, -9, -10, -20], index=list("ABCDEFGHIJ"))
    params = PortfolioParams(long_quantile=0.3, short_quantile=0.3, max_position_weight=1.0)
    w = construct_weights(preds, params)
    longs = w[w > 0].index.tolist()
    shorts = w[w < 0].index.tolist()
    assert set(longs) == {"A", "B", "C"}
    assert set(shorts) == {"H", "I", "J"}


def test_construct_weights_small_universe_returns_zero():
    preds = pd.Series([1, 2, 3], index=["A", "B", "C"])
    params = PortfolioParams()
    w = construct_weights(preds, params)
    assert (w == 0).all()


def test_compute_turnover_first_row_equals_gross_entry():
    dates = pd.bdate_range("2020-01-01", periods=3)
    w = pd.DataFrame(
        [[0.5, -0.5, 0.0], [0.5, 0.0, -0.5], [0.0, 0.0, 0.0]], index=dates, columns=["A", "B", "C"]
    )
    turnover = compute_turnover(w)
    assert turnover.iloc[0] == pytest.approx(1.0)  # entering 0.5 long + 0.5 short from cash
    assert turnover.iloc[1] == pytest.approx(abs(0.0) + abs(0.5) + abs(0.5))  # A unchanged, B and C moved


def test_apply_transaction_costs_scales_linearly():
    turnover = pd.Series([1.0, 2.0, 0.5])
    cost = apply_transaction_costs(turnover, cost_bps=10)
    expected = turnover * 0.001
    pd.testing.assert_series_equal(cost, expected)


def test_exposures_split_long_short():
    dates = pd.bdate_range("2020-01-01", periods=2)
    w = pd.DataFrame([[0.3, -0.2, 0.1], [0.0, -0.5, 0.0]], index=dates, columns=["A", "B", "C"])
    long_exp, short_exp = exposures(w)
    assert long_exp.iloc[0] == pytest.approx(0.4)
    assert short_exp.iloc[0] == pytest.approx(0.2)
    assert long_exp.iloc[1] == pytest.approx(0.0)
    assert short_exp.iloc[1] == pytest.approx(0.5)
