"""Prediction models.

Kept deliberately simple and interpretable: Ridge and ElasticNet linear
models (scaled with a StandardScaler fit ONLY on the training fold), plus a
zero-parameter cross-sectional rank baseline that just uses the neighbour
diffusion feature directly. Hyperparameters are chosen by time-series cross
-validation confined to the training window, never touching test data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RankSignalModel:
    """Stateless baseline: prediction = raw value of one feature column.

    Used to test whether the graph diffusion feature has cross-sectional rank
    information on its own, without any fitted weights.
    """

    def __init__(self, feature_name: str = "neighbour_diffusion_resid"):
        self.feature_name = feature_name

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankSignalModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.feature_name].to_numpy()


@dataclass
class FittedLinearModel:
    pipeline: Pipeline
    best_alpha: float
    feature_names: list[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[self.feature_names])

    @property
    def coefficients(self) -> pd.Series:
        model = self.pipeline.named_steps["model"]
        return pd.Series(model.coef_, index=self.feature_names)


def _time_series_grid_search(estimator_factory, param_grid, X, y, cv_folds, seed):
    """Simple time-series CV grid search that respects row (=date) order.

    The feature panel is already sorted by date, so a plain TimeSeriesSplit
    over row order approximates date-respecting expanding-window CV without
    needing a custom per-date splitter.
    """
    n = len(X)
    n_splits = min(cv_folds, max(2, n // 100))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = -np.inf
    best_param = param_grid[0]
    for p in param_grid:
        scores = []
        for train_idx, val_idx in tscv.split(X):
            est = estimator_factory(p, seed)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipe.predict(X.iloc[val_idx])
            # Spearman-style rank correlation is more relevant for a
            # cross-sectional ranking strategy than raw R^2, but R^2 is
            # cheap and stable for hyperparameter selection here.
            yv = y.iloc[val_idx]
            if yv.std() == 0 or np.std(pred) == 0:
                scores.append(-np.inf)
                continue
            corr = np.corrcoef(pred, yv)[0, 1]
            scores.append(corr if not np.isnan(corr) else -np.inf)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_param = p
    return best_param


def fit_ridge(
    X: pd.DataFrame, y: pd.Series, alpha_grid: list[float], cv_folds: int = 4, seed: int = 42
) -> FittedLinearModel:
    feature_names = list(X.columns)
    best_alpha = _time_series_grid_search(
        lambda a, s: Ridge(alpha=a, random_state=s), alpha_grid, X, y, cv_folds, seed
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=best_alpha, random_state=seed))])
    pipe.fit(X, y)
    return FittedLinearModel(pipeline=pipe, best_alpha=best_alpha, feature_names=feature_names)


def fit_elasticnet(
    X: pd.DataFrame, y: pd.Series, alpha_grid: list[float], l1_ratio: float = 0.5,
    cv_folds: int = 4, seed: int = 42,
) -> FittedLinearModel:
    feature_names = list(X.columns)
    best_alpha = _time_series_grid_search(
        lambda a, s: ElasticNet(alpha=a, l1_ratio=l1_ratio, random_state=s, max_iter=5000),
        alpha_grid, X, y, cv_folds, seed,
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=best_alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=5000)),
    ])
    pipe.fit(X, y)
    return FittedLinearModel(pipeline=pipe, best_alpha=best_alpha, feature_names=feature_names)


MODEL_REGISTRY = {
    "ridge": fit_ridge,
    "elasticnet": fit_elasticnet,
}
