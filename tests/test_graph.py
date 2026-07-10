import numpy as np
import pandas as pd
import pytest

from graph_diffusion_signal.graph import (
    GraphParams,
    _adjacency_from_window,
    diffusion_score,
    rolling_graphs,
    weighted_degree_centrality,
)


def _make_returns(n=200, tickers=("A", "B", "C", "D", "E"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = rng.normal(0, 0.01, size=(n, len(tickers)))
    return pd.DataFrame(data, index=dates, columns=list(tickers))


def test_adjacency_row_normalised_and_nonnegative():
    returns = _make_returns()
    params = GraphParams(window=60, top_k=2, min_abs_corr=-1.0)  # accept all positive corr
    adj = _adjacency_from_window(returns, params)
    assert (adj.values >= 0).all()
    row_sums = adj.sum(axis=1)
    # Rows with at least one edge should sum to ~1 (row-normalised)
    has_edges = row_sums > 0
    assert np.allclose(row_sums[has_edges], 1.0, atol=1e-8)


def test_adjacency_respects_top_k():
    returns = _make_returns(tickers=[f"T{i}" for i in range(10)])
    params = GraphParams(window=60, top_k=3, min_abs_corr=-1.0)
    adj = _adjacency_from_window(returns, params)
    nonzero_counts = (adj > 0).sum(axis=1)
    assert (nonzero_counts <= 3).all()


def test_no_self_loops_by_default():
    returns = _make_returns()
    params = GraphParams(window=60, top_k=4, min_abs_corr=-1.0, self_loops=False)
    adj = _adjacency_from_window(returns, params)
    assert np.allclose(np.diag(adj.values), 0.0)


def test_min_abs_corr_threshold_can_zero_out_all_edges():
    returns = _make_returns()
    params = GraphParams(window=60, top_k=4, min_abs_corr=0.999)
    adj = _adjacency_from_window(returns, params)
    # With near-independent random returns, essentially nothing clears 0.999 corr.
    assert (adj.values == 0).all()


def test_diffusion_score_matches_matrix_multiplication():
    idx = ["A", "B", "C"]
    adj = pd.DataFrame(
        [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.3, 0.7, 0.0]], index=idx, columns=idx
    )
    lag_returns = pd.Series([0.01, -0.02, 0.03], index=idx)
    score = diffusion_score(adj, lag_returns)
    expected = pd.Series(adj.values @ lag_returns.values, index=idx)
    pd.testing.assert_series_equal(score.sort_index(), expected.sort_index())


def test_weighted_degree_centrality_nonnegative():
    returns = _make_returns()
    params = GraphParams(window=60, top_k=3, min_abs_corr=-1.0)
    adj = _adjacency_from_window(returns, params)
    centrality = weighted_degree_centrality(adj)
    assert (centrality >= 0).all()


def test_rolling_graphs_only_uses_past_window():
    returns = _make_returns(n=150)
    params = GraphParams(window=60, top_k=3, min_abs_corr=-1.0)
    graphs = rolling_graphs(returns, params)
    idx = returns.index
    for date, adj in graphs.items():
        i = idx.get_loc(date)
        expected_window = returns.iloc[i - params.window : i]
        recomputed = _adjacency_from_window(expected_window, params)
        pd.testing.assert_frame_equal(adj.sort_index().sort_index(axis=1), recomputed.sort_index().sort_index(axis=1))
        # The graph for `date` must not have been built using a window that
        # includes `date` itself.
        assert date not in expected_window.index
