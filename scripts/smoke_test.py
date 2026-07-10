#!/usr/bin/env python3
"""Fast end-to-end smoke test: tiny universe, short history, coarse
walk-forward schedule. Meant to finish in well under a minute and catch
wiring/import/shape bugs -- NOT to produce a meaningful research result.
Used by ``make smoke`` and safe to run in CI.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_diffusion_signal import backtest as backtest_mod  # noqa: E402
from graph_diffusion_signal import features as features_mod  # noqa: E402
from graph_diffusion_signal import graph as graph_mod  # noqa: E402
from graph_diffusion_signal.data import compute_simple_returns, generate_synthetic_universe  # noqa: E402
from graph_diffusion_signal.portfolio import PortfolioParams  # noqa: E402

import pandas as pd  # noqa: E402


def main() -> int:
    t0 = time.time()
    print("[smoke] Generating tiny synthetic universe...")
    tickers = ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "SONY", "TTWO", "DAL", "UPS"]
    sector_of = {
        "NVDA": "semi", "AMD": "semi", "INTC": "semi", "AVGO": "semi", "QCOM": "semi", "MU": "semi",
        "SONY": "media", "TTWO": "media", "DAL": "transport", "UPS": "transport",
    }
    benchmarks = ["SPY", "SMH", "QQQ", "IYT"]
    panels = generate_synthetic_universe(tickers, sector_of, benchmarks, "2021-01-01", "2023-06-30", seed=123)

    prices = pd.DataFrame({t: panels[t]["Adj Close"] for t in tickers}).dropna()
    volumes = pd.DataFrame({t: panels[t]["Volume"] for t in tickers}).reindex(prices.index)
    returns_full = compute_simple_returns(pd.DataFrame({t: panels[t]["Adj Close"] for t in tickers + benchmarks}))
    common = returns_full.index.intersection(prices.index)
    prices, volumes = prices.loc[common], volumes.loc[common]
    returns_universe = returns_full.loc[common, tickers]
    market_returns = returns_full.loc[common, "SPY"]
    sector_returns = {"semi": returns_full.loc[common, "SMH"], "media": returns_full.loc[common, "QQQ"],
                       "transport": returns_full.loc[common, "IYT"]}

    print("[smoke] Building residual returns + rolling graph + feature panel...")
    fparams = features_mod.FeatureParams(beta_window=30, min_beta_window=15, residual_method="sector")
    gparams = graph_mod.GraphParams(window=30, top_k=3, min_abs_corr=-1.0)

    resid = features_mod.compute_residual_returns(returns_universe, market_returns, sector_of, sector_returns, fparams)
    graphs = graph_mod.rolling_graphs(returns_universe, gparams)
    panel = features_mod.build_feature_panel(
        prices, volumes, returns_universe, resid, market_returns, sector_of, sector_returns, graphs, fparams
    )
    assert not panel.empty, "smoke test: feature panel is empty"

    print("[smoke] Running a 2-fold walk-forward Ridge backtest...")
    wf_params = backtest_mod.WalkForwardParams(train_years=1, refit_frequency="Q", test_start="2022-01-01", min_train_days=150)
    oos_preds, fold_diag = backtest_mod.run_walkforward(
        panel, features_mod.FEATURE_COLUMNS, "target_1d", "ridge",
        {"alpha_grid": [1.0, 10.0], "cv_folds": 2}, wf_params, seed=42,
    )
    assert len(oos_preds) > 0, "smoke test: no out-of-sample predictions produced"
    assert len(fold_diag) >= 1, "smoke test: no walk-forward folds ran"

    pparams = PortfolioParams(long_quantile=0.3, short_quantile=0.3, max_position_weight=0.3)
    port = backtest_mod.predictions_to_portfolio(oos_preds, returns_universe, pparams, cost_bps=5)
    assert len(port["net_returns"]) > 0, "smoke test: no portfolio returns produced"

    from graph_diffusion_signal import metrics as metrics_mod
    sharpe = metrics_mod.sharpe_ratio(port["net_returns"])
    print(f"[smoke] OK -- {len(oos_preds)} OOS predictions, {len(fold_diag)} folds, "
          f"net Sharpe on tiny/synthetic sample = {sharpe:.2f} (not meaningful, smoke only).")
    print(f"[smoke] Completed in {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
