"""Feature engineering and target construction.

Timing convention (read this before touching the file)
--------------------------------------------------------
* ``returns.loc[t, ticker]`` is the simple return realised BETWEEN the close
  of ``t-1`` and the close of ``t``. It becomes fully known exactly at the
  close of day ``t``.
* A "feature row" for ``(feature_date=f, ticker=i)`` may use any information
  that is fully known by the close of day ``f`` -- i.e. returns/volumes up to
  and including ``f``, and graph structure built from correlations up to
  ``f-1`` (see ``graph.py``).
* The prediction target for that row is a return realised strictly AFTER
  ``f``: ``target_1d = returns.loc[f+1]`` and ``target_5d = cumulative
  return over (f+1 .. f+5]``. The target is never used to build the feature
  row it is attached to.
* Rolling-beta residualisation estimates betas on a trailing window ending at
  ``f-1`` (via ``.shift(1)``) and only then applies them to the realised
  factor return at ``f`` -- so the beta estimate itself never sees day f's
  stock return.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from graph_diffusion_signal.graph import diffusion_score, weighted_degree_centrality

logger = logging.getLogger(__name__)


@dataclass
class FeatureParams:
    lag_windows: list[int] = field(default_factory=lambda: [1, 3, 5, 20])
    vol_window: int = 20
    volume_window: int = 20
    momentum_window: int = 20
    reversal_window: int = 5
    beta_window: int = 60
    min_beta_window: int = 40
    residual_method: str = "sector"  # "market" or "sector"
    target_horizons: tuple[int, int] = (1, 5)


# --------------------------------------------------------------------------- #
# Rolling-beta residualisation
# --------------------------------------------------------------------------- #
def _rolling_beta_univariate(r_i: pd.Series, r_f: pd.Series, window: int, min_periods: int) -> pd.Series:
    cov = r_i.rolling(window, min_periods=min_periods).cov(r_f)
    var = r_f.rolling(window, min_periods=min_periods).var()
    beta = (cov / var.replace(0, np.nan)).shift(1)  # estimated on data through t-1
    return beta


def _rolling_beta_bivariate(
    r_i: pd.Series, r_mkt: pd.Series, r_sec: pd.Series, window: int, min_periods: int
) -> tuple[pd.Series, pd.Series]:
    var_m = r_mkt.rolling(window, min_periods=min_periods).var()
    var_s = r_sec.rolling(window, min_periods=min_periods).var()
    cov_ms = r_mkt.rolling(window, min_periods=min_periods).cov(r_sec)
    cov_mi = r_i.rolling(window, min_periods=min_periods).cov(r_mkt)
    cov_si = r_i.rolling(window, min_periods=min_periods).cov(r_sec)

    det = var_m * var_s - cov_ms**2
    det = det.replace(0, np.nan)

    beta_mkt = (var_s * cov_mi - cov_ms * cov_si) / det
    beta_sec = (var_m * cov_si - cov_ms * cov_mi) / det
    return beta_mkt.shift(1), beta_sec.shift(1)  # estimated on data through t-1


def compute_residual_returns(
    returns_raw: pd.DataFrame,
    market_returns: pd.Series,
    sector_of: dict[str, str],
    sector_returns: dict[str, pd.Series],
    params: FeatureParams,
) -> pd.DataFrame:
    """Rolling-beta residual returns for every ticker.

    ``residual_method='market'`` regresses each stock only on the market
    factor. ``residual_method='sector'`` additionally includes the stock's
    own sector benchmark ETF return.
    """
    out = {}
    for ticker in returns_raw.columns:
        r_i = returns_raw[ticker]
        sector = sector_of.get(ticker)
        if params.residual_method == "sector" and sector in sector_returns:
            r_sec = sector_returns[sector]
            beta_mkt, beta_sec = _rolling_beta_bivariate(
                r_i, market_returns, r_sec, params.beta_window, params.min_beta_window
            )
            fitted = beta_mkt * market_returns + beta_sec * r_sec
        else:
            beta_mkt = _rolling_beta_univariate(
                r_i, market_returns, params.beta_window, params.min_beta_window
            )
            fitted = beta_mkt * market_returns
        out[ticker] = r_i - fitted
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Cross-sectional feature panel
# --------------------------------------------------------------------------- #
def _lagged_return_features(prices: pd.DataFrame, windows: list[int]) -> dict[str, pd.DataFrame]:
    return {f"own_ret_{w}d": prices.pct_change(w) for w in windows}


def _rolling_vol(returns_raw: pd.DataFrame, window: int) -> pd.DataFrame:
    return returns_raw.rolling(window, min_periods=max(5, window // 2)).std()


def _volume_surprise(volumes: pd.DataFrame, window: int) -> pd.DataFrame:
    roll_mean = volumes.rolling(window, min_periods=max(5, window // 2)).mean().shift(1)
    roll_std = volumes.rolling(window, min_periods=max(5, window // 2)).std().shift(1)
    return (volumes - roll_mean) / roll_std.replace(0, np.nan)


def build_feature_panel(
    prices_wide: pd.DataFrame,
    volumes_wide: pd.DataFrame,
    returns_raw: pd.DataFrame,
    residual_returns: pd.DataFrame,
    market_returns: pd.Series,
    sector_of: dict[str, str],
    sector_returns: dict[str, pd.Series],
    graphs: dict[pd.Timestamp, pd.DataFrame],
    params: FeatureParams,
) -> pd.DataFrame:
    """Assemble the full leakage-safe long-format feature + target panel.

    Returns a DataFrame indexed by (date, ticker) MultiIndex.
    """
    tickers = list(returns_raw.columns)

    own_ret = _lagged_return_features(prices_wide, params.lag_windows)
    vol20 = _rolling_vol(returns_raw, params.vol_window)
    vol_surprise = _volume_surprise(volumes_wide, params.volume_window)
    momentum = prices_wide.pct_change(params.momentum_window)
    reversal = -returns_raw.rolling(params.reversal_window).sum()

    records = []
    dates = sorted(graphs.keys())
    max_h = max(params.target_horizons)

    for f in dates:
        if f not in returns_raw.index:
            continue
        loc = returns_raw.index.get_loc(f)
        if loc + max_h >= len(returns_raw.index):
            continue  # not enough future data left for the largest horizon

        adj = graphs[f]
        common = [t for t in tickers if t in adj.columns]
        if len(common) < 3:
            continue

        neighbour_raw = diffusion_score(adj, returns_raw.loc[f, common])
        neighbour_resid = diffusion_score(adj, residual_returns.loc[f, common])
        centrality = weighted_degree_centrality(adj)

        target_1d = residual_returns.iloc[loc + 1][common] if 1 in params.target_horizons else None
        if max_h >= 5:
            target_5d = residual_returns.iloc[loc + 1 : loc + 6][common].sum()
        else:
            target_5d = None

        vol_f = vol20.loc[f, common]

        for t in common:
            sector = sector_of.get(t)
            row = {
                "date": f,
                "ticker": t,
                "sector": sector if sector else "benchmark",
                "own_ret_1d": own_ret["own_ret_1d"].loc[f, t] if 1 in params.lag_windows else np.nan,
                "own_ret_3d": own_ret.get("own_ret_3d", pd.DataFrame()).loc[f, t] if 3 in params.lag_windows else np.nan,
                "own_ret_5d": own_ret.get("own_ret_5d", pd.DataFrame()).loc[f, t] if 5 in params.lag_windows else np.nan,
                "own_ret_20d": own_ret.get("own_ret_20d", pd.DataFrame()).loc[f, t] if 20 in params.lag_windows else np.nan,
                "volatility_20d": vol_f.get(t, np.nan),
                "volume_surprise": vol_surprise.loc[f, t] if t in vol_surprise.columns else np.nan,
                "neighbour_diffusion_raw": neighbour_raw.get(t, 0.0),
                "neighbour_diffusion_resid": neighbour_resid.get(t, 0.0),
                "graph_centrality": centrality.get(t, 0.0),
                "sector_bench_return": sector_returns.get(sector, pd.Series(dtype=float)).get(f, np.nan)
                if sector else np.nan,
                "market_return": market_returns.get(f, np.nan),
                "momentum_20d": momentum.loc[f, t],
                "reversal_5d": reversal.loc[f, t],
                "target_1d": target_1d.get(t, np.nan) if target_1d is not None else np.nan,
                "target_5d": target_5d.get(t, np.nan) if target_5d is not None else np.nan,
            }
            v = row["volatility_20d"]
            row["own_ret_1d_volnorm"] = row["own_ret_1d"] / v if v and v > 0 else np.nan
            row["neighbour_diffusion_raw_volnorm"] = row["neighbour_diffusion_raw"] / v if v and v > 0 else np.nan
            row["neighbour_diffusion_resid_volnorm"] = row["neighbour_diffusion_resid"] / v if v and v > 0 else np.nan
            records.append(row)

    panel = pd.DataFrame.from_records(records)
    if panel.empty:
        raise ValueError("Feature panel is empty -- check date ranges / min history filters.")
    panel = panel.set_index(["date", "ticker"]).sort_index()
    return panel


FEATURE_COLUMNS = [
    "own_ret_1d",
    "own_ret_3d",
    "own_ret_5d",
    "own_ret_20d",
    "volatility_20d",
    "volume_surprise",
    "neighbour_diffusion_raw",
    "neighbour_diffusion_resid",
    "graph_centrality",
    "sector_bench_return",
    "market_return",
    "momentum_20d",
    "reversal_5d",
    "own_ret_1d_volnorm",
    "neighbour_diffusion_raw_volnorm",
    "neighbour_diffusion_resid_volnorm",
]
