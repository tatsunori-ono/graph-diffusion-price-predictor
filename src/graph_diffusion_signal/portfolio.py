"""Portfolio construction: cross-sectional ranking, dollar-neutral long/short
weights, position caps, optional volatility targeting, and turnover.

Timing convention
------------------
``weights.loc[f]`` denotes a portfolio DECIDED using information available at
the close of day ``f`` (i.e. built from a prediction that only used data up
to and including f). That portfolio is assumed executed at (or immediately
after) the close of ``f`` and therefore earns the return realised on the NEXT
trading day, ``f+1``. ``backtest.py`` is responsible for the date shift; this
module only builds weights for a given cross-section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioParams:
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    dollar_neutral: bool = True
    max_position_weight: float = 0.15
    vol_target_annual: float | None = None
    rebalance_frequency: str = "D"


def construct_weights(
    predictions: pd.Series,
    params: PortfolioParams,
    realised_vol: pd.Series | None = None,
) -> pd.Series:
    """Build long/short weights for one date's cross-section of predictions.

    ``predictions`` index = ticker. Names with NaN predictions are excluded.
    """
    preds = predictions.dropna()
    n = len(preds)
    if n < 5:
        return pd.Series(0.0, index=predictions.index)

    n_long = max(1, int(np.floor(n * params.long_quantile)))
    n_short = max(1, int(np.floor(n * params.short_quantile)))
    ranked = preds.sort_values(ascending=False)
    longs = ranked.index[:n_long]
    shorts = ranked.index[-n_short:]

    weights = pd.Series(0.0, index=predictions.index)
    weights.loc[longs] = 1.0 / n_long
    weights.loc[shorts] = -1.0 / n_short

    if not params.dollar_neutral:
        weights = weights - weights.mean()

    weights = weights.clip(-params.max_position_weight, params.max_position_weight)

    if params.vol_target_annual and realised_vol is not None:
        port_vol_proxy = float(
            np.sqrt((weights.reindex(realised_vol.index).fillna(0) ** 2 * realised_vol**2).sum())
        ) * np.sqrt(252)
        if port_vol_proxy > 0:
            scale = min(params.vol_target_annual / port_vol_proxy, 3.0)
            weights = weights * scale

    return weights


def compute_turnover(weights_df: pd.DataFrame) -> pd.Series:
    """Gross traded notional per rebalance: sum(|w_t - w_{t-1}|).
    The first row is charged as if entering from an all-cash book."""
    w = weights_df.fillna(0.0)
    diffs = w.diff().abs().sum(axis=1)
    diffs.iloc[0] = w.iloc[0].abs().sum()
    return diffs


def apply_transaction_costs(gross_turnover: pd.Series, cost_bps: float) -> pd.Series:
    return gross_turnover * (cost_bps / 10_000.0)


def exposures(weights_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    w = weights_df.fillna(0.0)
    long_exp = w.clip(lower=0).sum(axis=1)
    short_exp = w.clip(upper=0).abs().sum(axis=1)
    return long_exp, short_exp
