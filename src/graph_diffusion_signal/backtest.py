"""Walk-forward backtest orchestration.

This module is the one place where "no look-ahead" has to be enforced most
carefully, so the rules are repeated here:

1. At refit boundary ``b``, the model is trained ONLY on feature/target rows
   with ``date < b``.
2. The freshly trained model then produces predictions for every date in
   ``[b, next_b)`` -- strictly out-of-sample relative to its own training
   set.
3. Predictions at feature-date ``f`` are turned into portfolio weights at
   ``f``, which are assumed executed at the close of ``f`` and earn the
   return realised on the NEXT trading day ``f+1`` (handled by shifting the
   realised-return index by one trading day when computing P&L).
4. Scalers are refit from scratch at every boundary using only that fold's
   training rows (see ``models.py`` -- StandardScaler lives inside the
   sklearn Pipeline that is re-fit each fold).

``tests/test_backtest_no_lookahead.py`` checks these properties directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from graph_diffusion_signal import models as models_mod
from graph_diffusion_signal.portfolio import (
    PortfolioParams,
    apply_transaction_costs,
    compute_turnover,
    construct_weights,
    exposures,
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardParams:
    train_years: int = 3
    refit_frequency: str = "M"
    test_start: str = "2019-01-01"
    min_train_days: int = 500


def refit_schedule(
    dates: pd.DatetimeIndex, params: WalkForwardParams
) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    start = dates.min()
    initial_train_end = start + pd.DateOffset(years=params.train_years)
    test_start_ts = pd.Timestamp(params.test_start)
    first_refit = max(initial_train_end, test_start_ts)

    freq = "MS" if params.refit_frequency == "M" else "QS"
    anchors = pd.date_range(first_refit, dates.max(), freq=freq)

    refit_dates: list[pd.Timestamp] = []
    for a in anchors:
        candidates = dates[dates >= a]
        if len(candidates) == 0:
            continue
        refit_dates.append(candidates[0])
    refit_dates = sorted(set(refit_dates))
    if refit_dates and refit_dates[-1] != dates.max():
        pass  # last fold simply runs to the end of the sample
    return refit_dates


def run_walkforward(
    panel: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    model_kwargs: dict,
    wf_params: WalkForwardParams,
    seed: int = 42,
) -> tuple[pd.Series, list[dict]]:
    """Returns (oos_predictions, fold_diagnostics).

    ``oos_predictions`` is indexed like ``panel`` (date, ticker) and contains
    a prediction ONLY for rows that fell in some fold's out-of-sample test
    window (rows before the first refit boundary are left out entirely,
    matching the requirement that test always follows training).
    """
    dates = panel.index.get_level_values("date")
    boundaries = refit_schedule(dates, wf_params)
    if not boundaries:
        raise ValueError("No valid refit boundaries -- check walk-forward config vs data range.")

    all_preds = []
    fold_diagnostics = []

    for i, b in enumerate(boundaries):
        train_mask = dates < b
        next_b = boundaries[i + 1] if i + 1 < len(boundaries) else None
        if next_b is not None:
            test_mask = (dates >= b) & (dates < next_b)
        else:
            test_mask = dates >= b

        train_df = panel.loc[train_mask]
        test_df = panel.loc[test_mask]

        n_train_dates = train_df.index.get_level_values("date").nunique()
        if n_train_dates < wf_params.min_train_days or test_df.empty:
            continue

        X_train = train_df[feature_cols].dropna()
        y_train = train_df.loc[X_train.index, target_col].dropna()
        common_idx = X_train.index.intersection(y_train.index)
        X_train, y_train = X_train.loc[common_idx], y_train.loc[common_idx]
        if len(X_train) < 200:
            continue

        X_test = test_df[feature_cols]
        valid_test = X_test.dropna().index
        X_test = X_test.loc[valid_test]
        if X_test.empty:
            continue

        if model_name == "rank_signal":
            model = models_mod.RankSignalModel(**model_kwargs)
            model.fit(X_train, y_train)
            coef = None
        else:
            fit_fn = models_mod.MODEL_REGISTRY[model_name]
            fitted = fit_fn(X_train, y_train, seed=seed, **model_kwargs)
            model = fitted
            coef = fitted.coefficients

        preds = pd.Series(model.predict(X_test), index=X_test.index, name="prediction")
        all_preds.append(preds)
        fold_diagnostics.append(
            {
                "fold": i,
                "train_start": train_df.index.get_level_values("date").min(),
                "train_end": b,
                "test_start": b,
                "test_end": next_b,
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "coefficients": coef,
            }
        )

    if not all_preds:
        raise ValueError("Walk-forward produced no out-of-sample predictions.")

    oos_predictions = pd.concat(all_preds).sort_index()
    return oos_predictions, fold_diagnostics


def predictions_to_portfolio(
    predictions: pd.Series,
    returns_raw_wide: pd.DataFrame,
    portfolio_params: PortfolioParams,
    cost_bps: float,
    realised_vol_wide: pd.DataFrame | None = None,
) -> dict:
    """Turn a (date, ticker)-indexed prediction Series into a full backtest:
    weights, gross/net daily returns, turnover, exposures."""
    pred_wide = predictions.unstack("ticker")
    decision_dates = pred_wide.index

    weight_rows = []
    for f in decision_dates:
        vol_cs = realised_vol_wide.loc[f] if realised_vol_wide is not None and f in realised_vol_wide.index else None
        w = construct_weights(pred_wide.loc[f], portfolio_params, realised_vol=vol_cs)
        weight_rows.append(w.rename(f))
    weights_df = pd.DataFrame(weight_rows)
    weights_df.index.name = "date"

    # Shift decision dates to the next available trading date in the returns
    # index -- this is where "decide at close of f, earn return at f+1" is
    # implemented.
    all_dates = returns_raw_wide.index
    realise_on = []
    for f in weights_df.index:
        pos = all_dates.searchsorted(f)
        nxt = pos + 1
        realise_on.append(all_dates[nxt] if nxt < len(all_dates) else pd.NaT)
    weights_df["_realise_on"] = realise_on
    weights_df = weights_df.dropna(subset=["_realise_on"])
    realise_dates = weights_df.pop("_realise_on")

    turnover = compute_turnover(weights_df)
    cost = apply_transaction_costs(turnover, cost_bps)
    long_exp, short_exp = exposures(weights_df)

    gross_returns = []
    for decision_date, realise_date in zip(weights_df.index, realise_dates):
        w = weights_df.loc[decision_date]
        r = returns_raw_wide.loc[realise_date].reindex(w.index).fillna(0.0)
        gross_returns.append((w * r).sum())
    gross = pd.Series(gross_returns, index=realise_dates.values, name="gross_return")
    gross = gross.groupby(level=0).sum().sort_index()

    cost_realised = cost.copy()
    cost_realised.index = realise_dates.values
    cost_realised = cost_realised.groupby(level=0).sum().reindex(gross.index).fillna(0.0)

    net = gross - cost_realised
    turnover_realised = turnover.copy()
    turnover_realised.index = realise_dates.values
    turnover_realised = turnover_realised.groupby(level=0).sum().reindex(gross.index).fillna(0.0)

    long_exp.index = weights_df.index
    short_exp.index = weights_df.index

    return {
        "weights": weights_df,
        "gross_returns": gross,
        "net_returns": net,
        "turnover": turnover_realised,
        "long_exposure": long_exp,
        "short_exposure": short_exp,
        "predictions": predictions,
    }
