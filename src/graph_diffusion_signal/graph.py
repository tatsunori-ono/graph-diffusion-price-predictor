"""Rolling correlation graph construction.

Leakage discipline
-------------------
The adjacency matrix "as of" date ``t`` is built from a trailing correlation
window that ends at ``t - 1`` (i.e. the last trading day strictly before t).
It never uses returns realised on ``t`` or later. Callers must respect this:
``adjacency_for_date[t]`` is safe to combine with any feature/target pair
where the feature uses information available at the close of ``t - 1`` and
the target is a return realised on or after ``t``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GraphParams:
    window: int = 60
    top_k: int = 8
    min_abs_corr: float = 0.15
    self_loops: bool = False


def _adjacency_from_window(window_returns: pd.DataFrame, params: GraphParams) -> pd.DataFrame:
    """Build one row-normalised adjacency matrix from a trailing-return window.

    Only positive correlations are eligible (the hypothesis is about positive
    co-movement / shock diffusion, not hedging pairs), keeping each node's
    top-k strongest neighbours above ``min_abs_corr``.
    """
    corr = window_returns.corr()
    tickers = corr.columns
    if not params.self_loops:
        values = corr.to_numpy(copy=True)
        np.fill_diagonal(values, np.nan)
        corr = pd.DataFrame(values, index=tickers, columns=tickers)

    adj = pd.DataFrame(0.0, index=tickers, columns=tickers)
    for name in tickers:
        row = corr[name].drop(index=name, errors="ignore") if not params.self_loops else corr[name]
        row = row[row > 0]  # only positive co-movement is eligible (diffusion, not hedging pairs)
        row = row[row > params.min_abs_corr]
        if row.empty:
            continue
        top = row.sort_values(ascending=False).head(params.top_k)
        adj.loc[name, top.index] = top.values

    row_sums = adj.sum(axis=1)
    nonzero = row_sums > 0
    adj.loc[nonzero] = adj.loc[nonzero].div(row_sums[nonzero], axis=0)
    return adj


def _adjacency_from_corr_matrix(corr: np.ndarray, cols: pd.Index, params: GraphParams) -> pd.DataFrame:
    """Numpy-array equivalent of ``_adjacency_from_window``'s selection logic,
    operating directly on a precomputed correlation matrix. Kept numerically
    equivalent to ``_adjacency_from_window`` so the two code paths agree."""
    n = len(cols)
    c = corr.copy()
    if not params.self_loops:
        np.fill_diagonal(c, np.nan)

    adj = np.zeros((n, n))
    thresh = max(0.0, params.min_abs_corr)
    for i in range(n):
        row = c[i]
        eligible = np.where((row > thresh) & ~np.isnan(row))[0]
        if eligible.size == 0:
            continue
        if eligible.size > params.top_k:
            order = eligible[np.argsort(-row[eligible])][: params.top_k]
        else:
            order = eligible[np.argsort(-row[eligible])]
        adj[i, order] = row[order]

    row_sums = adj.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    adj[nonzero] = adj[nonzero] / row_sums[nonzero]
    return pd.DataFrame(adj, index=cols, columns=cols)


def rolling_graphs(
    returns_wide: pd.DataFrame, params: GraphParams
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Compute an adjacency matrix for every date with enough trailing history.

    The graph attributed to date ``t`` (``idx[i]``) is built strictly from
    ``returns_wide`` rows ``idx[i - window] .. idx[i - 1]`` -- i.e. up to and
    including t-1, and never t itself.

    Implementation note: correlation matrices are updated incrementally
    (rolling sums of first and second moments) rather than recomputed from
    scratch with ``DataFrame.corr()`` at every date. This is purely a
    performance optimisation -- it is numerically equivalent to calling
    ``_adjacency_from_window`` at every date (see ``tests/test_graph.py``)
    but roughly two orders of magnitude faster on a multi-thousand-day panel,
    which matters for walk-forward backtests re-run many times.
    """
    idx = returns_wide.index
    w = params.window
    if len(idx) <= w:
        return {}

    if returns_wide.isna().any().any():
        # Safety fallback for panels with residual gaps -- correctness over
        # speed in this (expected to be rare, post-cleaning) case.
        graphs: dict[pd.Timestamp, pd.DataFrame] = {}
        for i in range(w, len(idx)):
            window = returns_wide.iloc[i - w : i]
            if window.isna().any().any():
                window = window.dropna(axis=1)
                if window.shape[1] < 3:
                    continue
            graphs[idx[i]] = _adjacency_from_window(window, params)
        return graphs

    values = returns_wide.to_numpy(dtype=float)
    cols = returns_wide.columns
    T = values.shape[0]

    S = values[:w].sum(axis=0)
    M = values[:w].T @ values[:w]

    graphs = {}
    for i in range(w, T):
        mean = S / w
        cov = M / w - np.outer(mean, mean)
        std = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        denom = np.outer(std, std)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.where(denom > 0, cov / denom, 0.0)
        graphs[idx[i]] = _adjacency_from_corr_matrix(corr, cols, params)

        if i < T - 1:
            old, new = values[i - w], values[i]
            S = S - old + new
            M = M - np.outer(old, old) + np.outer(new, new)
    return graphs


def diffusion_score(
    adjacency: pd.DataFrame, lagged_returns: pd.Series
) -> pd.Series:
    """Neighbour return diffusion score: A @ r_lag, restricted to common tickers.

    ``lagged_returns`` must already be information available strictly before
    the date the score is attributed to (e.g. t-1 returns for a t-dated score).
    """
    common = adjacency.columns.intersection(lagged_returns.index)
    a = adjacency.loc[common, common].fillna(0.0)
    r = lagged_returns.loc[common].fillna(0.0)
    return pd.Series(a.values @ r.values, index=common)


def weighted_degree_centrality(adjacency: pd.DataFrame) -> pd.Series:
    """Row-sum-of-raw-weight style centrality before row normalisation is not
    recoverable from the normalised matrix alone, so we approximate centrality
    here using the column sums of the (already row-normalised) adjacency --
    i.e. how often / how strongly a node is chosen as a strong neighbour by
    others, which is a reasonable in-degree centrality proxy."""
    return adjacency.sum(axis=0)
