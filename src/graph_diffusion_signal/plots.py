"""Matplotlib plotting helpers. Every function saves a PNG to disk and
returns the path, so callers (CLI, report builder, notebook) can chain them
without duplicating figure styling."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graph_diffusion_signal.metrics import drawdown_series

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def _save(fig, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_equity_curve(
    returns_by_strategy: dict[str, pd.Series], path: str | Path, title: str = "Equity Curve"
) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, r in returns_by_strategy.items():
        curve = (1 + r.fillna(0)).cumprod()
        ax.plot(curve.index, curve.values, label=name, linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", fontsize=8)
    ax.axhline(1.0, color="grey", linewidth=0.6)
    return _save(fig, path)


def plot_drawdown(returns: pd.Series, path: str | Path, title: str = "Drawdown") -> str:
    dd = drawdown_series(returns)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(dd.index, dd.values * 100, 0, color="firebrick", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    return _save(fig, path)


def plot_coefficients(coef_by_fold: pd.DataFrame, path: str | Path,
                       title: str = "Model Coefficients Over Time") -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    coef_by_fold.plot(ax=ax, marker="o", markersize=3, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel("Standardised coefficient")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    return _save(fig, path)


def plot_avg_abs_coefficients(mean_abs_coef: pd.Series, path: str | Path,
                               title: str = "Average |Coefficient| Across Folds") -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mean_abs_coef.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Mean |standardised coefficient|")
    return _save(fig, path)


def plot_cost_sensitivity(cost_table: pd.DataFrame, path: str | Path,
                           title: str = "Sharpe Ratio vs Transaction Cost") -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cost_table["cost_bps"], cost_table["sharpe"], marker="o")
    ax.set_xlabel("Cost (bps, one-way)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(title)
    ax.axhline(0, color="grey", linewidth=0.6)
    return _save(fig, path)


def plot_parameter_sensitivity(sens_table: pd.DataFrame, x_col: str, y_col: str,
                                group_col: str | None, path: str | Path, title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if group_col:
        for g, grp in sens_table.groupby(group_col):
            ax.plot(grp[x_col], grp[y_col], marker="o", label=f"{group_col}={g}")
        ax.legend(fontsize=8)
    else:
        ax.plot(sens_table[x_col], sens_table[y_col], marker="o")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    return _save(fig, path)


def plot_year_by_year(year_table: pd.DataFrame, path: str | Path,
                       title: str = "Annual Returns") -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["seagreen" if v >= 0 else "firebrick" for v in year_table["Return"]]
    ax.bar(year_table.index.astype(str), year_table["Return"] * 100, color=colors)
    ax.set_ylabel("Return (%)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.6)
    return _save(fig, path)
