"""Performance metrics and statistical robustness tests.

All functions here are pure (take return series in, return numbers/tables
out) and operate strictly on already-realised historical returns, so there
is nothing forward-looking about them by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

TRADING_DAYS = 252


def annualised_return(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return np.nan
    growth = (1 + daily_returns).prod()
    years = len(daily_returns) / TRADING_DAYS
    if years <= 0 or growth <= 0:
        return np.nan
    return growth ** (1 / years) - 1


def annualised_vol(daily_returns: pd.Series) -> float:
    return daily_returns.dropna().std(ddof=1) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, rf_annual: float = 0.0) -> float:
    r = daily_returns.dropna()
    if r.std(ddof=1) == 0 or r.empty:
        return np.nan
    excess = r - rf_annual / TRADING_DAYS
    return (excess.mean() / excess.std(ddof=1)) * np.sqrt(TRADING_DAYS)


def sortino_ratio(daily_returns: pd.Series, rf_annual: float = 0.0) -> float:
    r = daily_returns.dropna()
    excess = r - rf_annual / TRADING_DAYS
    downside = excess[excess < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd) or r.empty:
        return np.nan
    return (excess.mean() / dd) * np.sqrt(TRADING_DAYS)


def max_drawdown(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    curve = (1 + r).cumprod()
    peak = curve.cummax()
    dd = curve / peak - 1
    return dd.min()


def calmar_ratio(daily_returns: pd.Series) -> float:
    mdd = max_drawdown(daily_returns)
    ar = annualised_return(daily_returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return ar / abs(mdd)


def hit_rate(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if r.empty:
        return np.nan
    return (r > 0).mean()


def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    r = daily_returns.dropna()
    curve = (1 + r).cumprod()
    peak = curve.cummax()
    return curve / peak - 1


def beta_alpha_vs_benchmark(
    daily_returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[float, float, float]:
    """OLS regression of strategy returns on benchmark returns.
    Returns (beta, annualised_alpha, r_squared)."""
    df = pd.concat([daily_returns, benchmark_returns], axis=1).dropna()
    df.columns = ["strategy", "benchmark"]
    if len(df) < 30:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(df["benchmark"])
    model = sm.OLS(df["strategy"], X).fit()
    beta = model.params["benchmark"]
    alpha_daily = model.params["const"]
    alpha_annual = (1 + alpha_daily) ** TRADING_DAYS - 1
    return beta, alpha_annual, model.rsquared


def information_ratio(daily_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    active = (daily_returns - benchmark_returns).dropna()
    if active.std(ddof=1) == 0 or active.empty:
        return np.nan
    return (active.mean() / active.std(ddof=1)) * np.sqrt(TRADING_DAYS)


def summary_table(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    long_exposure: pd.Series | None = None,
    short_exposure: pd.Series | None = None,
) -> dict:
    out = {
        "Annualised Return": annualised_return(daily_returns),
        "Annualised Volatility": annualised_vol(daily_returns),
        "Sharpe Ratio": sharpe_ratio(daily_returns),
        "Sortino Ratio": sortino_ratio(daily_returns),
        "Max Drawdown": max_drawdown(daily_returns),
        "Calmar Ratio": calmar_ratio(daily_returns),
        "Hit Rate": hit_rate(daily_returns),
    }
    if turnover is not None:
        out["Avg Daily Turnover"] = turnover.mean()
    if long_exposure is not None:
        out["Avg Long Exposure"] = long_exposure.mean()
    if short_exposure is not None:
        out["Avg Short Exposure"] = short_exposure.mean()
    if benchmark_returns is not None:
        beta, alpha, r2 = beta_alpha_vs_benchmark(daily_returns, benchmark_returns)
        out["Beta to Benchmark"] = beta
        out["Annualised Alpha"] = alpha
        out["Alpha R-squared"] = r2
        out["Information Ratio"] = information_ratio(daily_returns, benchmark_returns)
    return out


def year_by_year_table(daily_returns: pd.Series) -> pd.DataFrame:
    r = daily_returns.dropna()
    rows = []
    for year, grp in r.groupby(r.index.year):
        rows.append(
            {
                "Year": year,
                "Return": annualised_return(grp) if len(grp) > 20 else (1 + grp).prod() - 1,
                "Volatility": annualised_vol(grp),
                "Sharpe": sharpe_ratio(grp),
                "Max Drawdown": max_drawdown(grp),
                "N Days": len(grp),
            }
        )
    return pd.DataFrame(rows).set_index("Year")


# --------------------------------------------------------------------------- #
# Statistical robustness
# --------------------------------------------------------------------------- #
def permutation_test_signal(
    predictions: pd.Series, forward_returns: pd.Series, n_permutations: int = 200, seed: int = 42
) -> dict:
    """Permutation test for whether the (predictions, forward returns)
    relationship is stronger than chance. Shuffles the prediction labels
    within each date's cross-section is not done here (this operates on a
    pooled series); the null is generated by randomly permuting predictions
    against realised forward returns and recomputing the Spearman rank
    correlation, repeated ``n_permutations`` times.
    """
    df = pd.concat([predictions, forward_returns], axis=1).dropna()
    df.columns = ["pred", "fwd"]
    if len(df) < 50:
        return {"observed_ic": np.nan, "p_value": np.nan, "n_obs": len(df)}
    observed_ic, _ = stats.spearmanr(df["pred"], df["fwd"])
    rng = np.random.default_rng(seed)
    perm_ics = np.empty(n_permutations)
    fwd_vals = df["fwd"].values
    pred_vals = df["pred"].values
    for i in range(n_permutations):
        shuffled = rng.permutation(fwd_vals)
        perm_ics[i], _ = stats.spearmanr(pred_vals, shuffled)
    p_value = (np.sum(np.abs(perm_ics) >= np.abs(observed_ic)) + 1) / (n_permutations + 1)
    return {
        "observed_ic": observed_ic,
        "p_value": p_value,
        "n_obs": len(df),
        "perm_ic_mean": np.mean(perm_ics),
        "perm_ic_std": np.std(perm_ics),
    }


def bootstrap_sharpe_ci(
    daily_returns: pd.Series, n_bootstrap: int = 1000, seed: int = 42, ci: float = 0.90
) -> dict:
    r = daily_returns.dropna().values
    if len(r) < 60:
        return {"sharpe": np.nan, "lower": np.nan, "upper": np.nan}
    rng = np.random.default_rng(seed)
    n = len(r)
    boot_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(r, size=n, replace=True)
        s = sample.std(ddof=1)
        boot_sharpes[i] = (sample.mean() / s) * np.sqrt(TRADING_DAYS) if s > 0 else np.nan
    boot_sharpes = boot_sharpes[~np.isnan(boot_sharpes)]
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(boot_sharpes, [alpha, 1 - alpha])
    return {
        "sharpe": sharpe_ratio(daily_returns),
        "lower": lower,
        "upper": upper,
        "ci": ci,
        "n_bootstrap": len(boot_sharpes),
    }
