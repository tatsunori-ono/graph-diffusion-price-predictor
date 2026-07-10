"""Command-line interface and pipeline orchestration.

Usage:
    python -m graph_diffusion_signal.cli data       [--force-refresh]
    python -m graph_diffusion_signal.cli backtest    [--smoke]
    python -m graph_diffusion_signal.cli report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from graph_diffusion_signal.config import BacktestConfig, UniverseConfig
from graph_diffusion_signal import data as data_mod
from graph_diffusion_signal import features as features_mod
from graph_diffusion_signal import graph as graph_mod
from graph_diffusion_signal import backtest as backtest_mod
from graph_diffusion_signal import metrics as metrics_mod
from graph_diffusion_signal import plots as plots_mod
from graph_diffusion_signal import report as report_mod
from graph_diffusion_signal.portfolio import PortfolioParams

logger = logging.getLogger("graph_diffusion_signal")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def load_configs(config_dir: str = "config") -> tuple[UniverseConfig, BacktestConfig]:
    uni = UniverseConfig.from_yaml(Path(config_dir) / "universe.yaml")
    bt = BacktestConfig.from_yaml(Path(config_dir) / "backtest.yaml")
    return uni, bt


# --------------------------------------------------------------------------- #
# Stage 1: data
# --------------------------------------------------------------------------- #
def stage_data(uni: UniverseConfig, bt: BacktestConfig, data_dir: str, force_refresh: bool = False):
    sector_of = {t: s for s, names in uni.sectors.items() for t in names}
    panels, used_synthetic = data_mod.download_universe(
        tickers=uni.all_tickers,
        sector_of=sector_of,
        benchmarks=uni.benchmarks,
        start=uni.start_date,
        end=uni.end_date,
        raw_dir=Path(data_dir) / "raw",
        seed=bt["random_seed"],
        force_refresh=force_refresh,
    )
    prices = data_mod.clean_and_align(
        panels, min_history_days=uni.min_history_days, price_field=bt["data"]["price_field"]
    )
    kept = list(prices.columns)
    volumes = data_mod.align_volume(panels, kept, prices.index)
    data_mod.save_processed(prices, Path(data_dir) / "processed" / "prices_wide.csv")
    data_mod.save_processed(volumes, Path(data_dir) / "processed" / "volumes_wide.csv")
    logger.info(
        "Data stage complete: %d tickers kept, %d trading days, synthetic=%s",
        prices.shape[1], prices.shape[0], used_synthetic,
    )
    return prices, volumes, used_synthetic, sector_of


# --------------------------------------------------------------------------- #
# Stage 2: features
# --------------------------------------------------------------------------- #
def stage_features(uni, bt, prices, volumes, sector_of):
    universe_tickers = [t for t in uni.all_tickers if t in prices.columns]
    bench_tickers = [t for t in uni.benchmarks if t in prices.columns]

    returns_raw_full = data_mod.compute_simple_returns(prices)
    returns_universe = returns_raw_full[universe_tickers]

    market_returns = returns_raw_full["SPY"] if "SPY" in returns_raw_full.columns else returns_raw_full.mean(axis=1)
    sector_bench_map = {s: uni.sector_benchmark(s) for s in uni.sectors}
    sector_returns = {
        s: returns_raw_full[b] for s, b in sector_bench_map.items() if b in returns_raw_full.columns
    }

    fparams = features_mod.FeatureParams(
        lag_windows=bt["features"]["lag_windows"],
        vol_window=bt["features"]["vol_window"],
        volume_window=bt["features"]["volume_window"],
        momentum_window=bt["features"]["momentum_window"],
        reversal_window=bt["features"]["reversal_window"],
        beta_window=bt["returns"]["beta_window"],
        min_beta_window=bt["returns"]["min_beta_window"],
        residual_method=bt["returns"]["residual_method"],
        target_horizons=tuple([bt["target"]["primary_horizon_days"], bt["target"]["secondary_horizon_days"]]),
    )

    residual_returns = features_mod.compute_residual_returns(
        returns_universe, market_returns, sector_of, sector_returns, fparams
    )

    gparams = graph_mod.GraphParams(
        window=bt["graph"]["correlation_window"],
        top_k=bt["graph"]["top_k"],
        min_abs_corr=bt["graph"]["min_abs_corr"],
        self_loops=bt["graph"]["self_loops"],
    )
    graphs = graph_mod.rolling_graphs(returns_universe, gparams)

    panel = features_mod.build_feature_panel(
        prices_wide=prices[universe_tickers],
        volumes_wide=volumes[universe_tickers],
        returns_raw=returns_universe,
        residual_returns=residual_returns,
        market_returns=market_returns,
        sector_of=sector_of,
        sector_returns=sector_returns,
        graphs=graphs,
        params=fparams,
    )
    return {
        "panel": panel,
        "returns_universe": returns_universe,
        "returns_full": returns_raw_full,
        "residual_returns": residual_returns,
        "market_returns": market_returns,
        "sector_returns": sector_returns,
        "graphs": graphs,
        "universe_tickers": universe_tickers,
        "bench_tickers": bench_tickers,
        "fparams": fparams,
        "gparams": gparams,
    }


# --------------------------------------------------------------------------- #
# Stage 3: walk-forward backtest across models + baselines + benchmarks
# --------------------------------------------------------------------------- #
def stage_backtest(uni, bt, feat: dict, target_col: str = "target_1d"):
    panel = feat["panel"]
    returns_universe = feat["returns_universe"]

    wf_params = backtest_mod.WalkForwardParams(
        train_years=bt["walkforward"]["train_years"],
        refit_frequency=bt["walkforward"]["refit_frequency"],
        test_start=bt["walkforward"]["test_start"],
        min_train_days=bt["walkforward"]["min_train_days"],
    )
    pparams = PortfolioParams(
        long_quantile=bt["portfolio"]["long_quantile"],
        short_quantile=bt["portfolio"]["short_quantile"],
        dollar_neutral=bt["portfolio"]["dollar_neutral"],
        max_position_weight=bt["portfolio"]["max_position_weight"],
        vol_target_annual=bt["portfolio"]["vol_target_annual"],
        rebalance_frequency=bt["portfolio"]["rebalance_frequency"],
    )
    cost_bps = bt["costs"]["default_bps"]
    seed = bt["random_seed"]

    model_specs = {
        "ridge": ("ridge", {"alpha_grid": bt["models"]["ridge_alpha_grid"], "cv_folds": bt["models"]["cv_folds"]}),
        "elasticnet": (
            "elasticnet",
            {
                "alpha_grid": bt["models"]["elasticnet_alpha_grid"],
                "l1_ratio": bt["models"]["elasticnet_l1_ratio"],
                "cv_folds": bt["models"]["cv_folds"],
            },
        ),
        "rank_signal": ("rank_signal", {"feature_name": "neighbour_diffusion_resid"}),
    }

    results = {}
    fold_diag_by_model = {}
    for name, (model_name, kwargs) in model_specs.items():
        logger.info("Running walk-forward for model=%s", name)
        oos_preds, fold_diag = backtest_mod.run_walkforward(
            panel, features_mod.FEATURE_COLUMNS, target_col, model_name, kwargs, wf_params, seed=seed
        )
        port = backtest_mod.predictions_to_portfolio(oos_preds, returns_universe, pparams, cost_bps)
        results[name] = port
        fold_diag_by_model[name] = fold_diag

    # Naive rule-based baselines over the SAME out-of-sample window as ridge.
    oos_index = results["ridge"]["predictions"].index.get_level_values("date").unique()
    for baseline_name, col in [("naive_momentum", "momentum_20d"), ("naive_reversal", "reversal_5d")]:
        preds = panel.loc[panel.index.get_level_values("date").isin(oos_index), col]
        preds = preds.rename("prediction")
        port = backtest_mod.predictions_to_portfolio(preds, returns_universe, pparams, cost_bps)
        results[baseline_name] = port

    # Buy-and-hold benchmarks over the same realised-return date range.
    date_lo, date_hi = results["ridge"]["gross_returns"].index.min(), results["ridge"]["gross_returns"].index.max()
    benchmarks = {}
    for b in uni.benchmarks:
        if b in feat["returns_full"].columns:
            benchmarks[b] = feat["returns_full"][b].loc[date_lo:date_hi]

    return {
        "results": results,
        "fold_diag_by_model": fold_diag_by_model,
        "benchmarks": benchmarks,
        "wf_params": wf_params,
        "portfolio_params": pparams,
        "cost_bps": cost_bps,
    }


# --------------------------------------------------------------------------- #
# Stage 4: robustness / statistical tests
# --------------------------------------------------------------------------- #
def cost_sensitivity_table(gross: pd.Series, turnover: pd.Series, cost_grid: list[float]) -> pd.DataFrame:
    rows = []
    for b in cost_grid:
        net = gross - turnover * (b / 10_000.0)
        rows.append({"cost_bps": b, "sharpe": metrics_mod.sharpe_ratio(net), "ann_return": metrics_mod.annualised_return(net)})
    return pd.DataFrame(rows)


def graph_parameter_sensitivity(
    returns_universe: pd.DataFrame,
    residual_returns: pd.DataFrame,
    window_grid: list[int],
    topk_grid: list[int],
    min_abs_corr: float,
) -> pd.DataFrame:
    """Pooled Spearman IC between the neighbour-residual-diffusion feature and
    next-day residual return, swept over graph hyperparameters. This is a
    lightweight sensitivity check (no model fitting / portfolio construction)
    so it stays fast across a 3x4 grid."""
    rows = []
    for window in window_grid:
        for top_k in topk_grid:
            gparams = graph_mod.GraphParams(window=window, top_k=top_k, min_abs_corr=min_abs_corr)
            graphs = graph_mod.rolling_graphs(returns_universe, gparams)
            preds, targets = [], []
            dates = sorted(graphs.keys())
            idx = residual_returns.index
            for f in dates:
                if f not in idx:
                    continue
                loc = idx.get_loc(f)
                if loc + 1 >= len(idx):
                    continue
                adj = graphs[f]
                common = [t for t in returns_universe.columns if t in adj.columns]
                if len(common) < 3:
                    continue
                score = graph_mod.diffusion_score(adj, residual_returns.loc[f, common])
                tgt = residual_returns.iloc[loc + 1][common]
                preds.append(score)
                targets.append(tgt)
            if not preds:
                continue
            pooled_pred = pd.concat(preds)
            pooled_tgt = pd.concat(targets)
            df = pd.concat([pooled_pred, pooled_tgt], axis=1).dropna()
            if len(df) < 100:
                ic = np.nan
            else:
                ic, _ = stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])
            rows.append({"corr_window": window, "top_k": top_k, "pooled_ic": ic, "n_obs": len(df)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_backtest_results(bt_out: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, port in bt_out["results"].items():
        port["gross_returns"].to_csv(out_dir / f"{name}_gross_returns.csv", header=["gross_return"])
        port["net_returns"].to_csv(out_dir / f"{name}_net_returns.csv", header=["net_return"])
        port["turnover"].to_csv(out_dir / f"{name}_turnover.csv", header=["turnover"])
        port["weights"].to_csv(out_dir / f"{name}_weights.csv")
    for name, series in bt_out["benchmarks"].items():
        series.to_csv(out_dir / f"benchmark_{name}.csv", header=["return"])
    logger.info("Saved backtest results to %s", out_dir)


def load_backtest_results(out_dir: str | Path, model_names: list[str], benchmark_names: list[str]) -> dict:
    out_dir = Path(out_dir)
    results = {}
    for name in model_names:
        results[name] = {
            "gross_returns": pd.read_csv(out_dir / f"{name}_gross_returns.csv", index_col=0, parse_dates=True)["gross_return"],
            "net_returns": pd.read_csv(out_dir / f"{name}_net_returns.csv", index_col=0, parse_dates=True)["net_return"],
            "turnover": pd.read_csv(out_dir / f"{name}_turnover.csv", index_col=0, parse_dates=True)["turnover"],
        }
    benchmarks = {}
    for name in benchmark_names:
        p = out_dir / f"benchmark_{name}.csv"
        if p.exists():
            benchmarks[name] = pd.read_csv(p, index_col=0, parse_dates=True)["return"]
    return results, benchmarks


# --------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------- #
def cmd_data(args):
    setup_logging(args.verbose)
    uni, bt = load_configs(args.config_dir)
    stage_data(uni, bt, args.data_dir, force_refresh=args.force_refresh)


def cmd_backtest(args):
    setup_logging(args.verbose)
    uni, bt = load_configs(args.config_dir)
    prices, volumes, used_synthetic, sector_of = stage_data(uni, bt, args.data_dir, force_refresh=args.force_refresh)
    feat = stage_features(uni, bt, prices, volumes, sector_of)
    bt_out = stage_backtest(uni, bt, feat)
    save_backtest_results(bt_out, Path(args.reports_dir) / "results")

    summary = {"used_synthetic_data": bool(used_synthetic), "n_tickers": prices.shape[1], "n_days": int(prices.shape[0])}
    for name, port in bt_out["results"].items():
        s = metrics_mod.summary_table(
            port["net_returns"],
            benchmark_returns=bt_out["benchmarks"].get("SPY"),
            turnover=port["turnover"],
            long_exposure=port["long_exposure"],
            short_exposure=port["short_exposure"],
        )
        s_gross = metrics_mod.summary_table(port["gross_returns"])
        summary[name] = {"net": s, "gross": s_gross}
    for name, series in bt_out["benchmarks"].items():
        summary[f"benchmark_{name}"] = metrics_mod.summary_table(series)

    cost_grid = bt["costs"]["sensitivity_bps"]
    cost_table = cost_sensitivity_table(bt_out["results"]["ridge"]["gross_returns"], bt_out["results"]["ridge"]["turnover"], cost_grid)
    cost_table.to_csv(Path(args.reports_dir) / "results" / "cost_sensitivity.csv", index=False)

    param_grid = graph_parameter_sensitivity(
        feat["returns_universe"], feat["residual_returns"],
        bt["robustness"]["corr_window_grid"],
        bt["robustness"]["graph_topk_grid"],
        bt["graph"]["min_abs_corr"],
    )
    param_grid.to_csv(Path(args.reports_dir) / "results" / "graph_param_sensitivity.csv", index=False)

    perm = metrics_mod.permutation_test_signal(
        bt_out["results"]["ridge"]["predictions"], feat["panel"]["target_1d"],
        n_permutations=bt["robustness"]["n_permutations"], seed=bt["random_seed"],
    )
    boot = metrics_mod.bootstrap_sharpe_ci(
        bt_out["results"]["ridge"]["net_returns"], n_bootstrap=bt["robustness"]["n_bootstrap"], seed=bt["random_seed"]
    )
    year_table = metrics_mod.year_by_year_table(bt_out["results"]["ridge"]["net_returns"])
    year_table.to_csv(Path(args.reports_dir) / "results" / "year_by_year_ridge.csv")

    with open(Path(args.reports_dir) / "results" / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(Path(args.reports_dir) / "results" / "robustness.json", "w") as fh:
        json.dump({"permutation_test": perm, "bootstrap_sharpe_ci": boot}, fh, indent=2, default=str)

    # Feature coefficients across folds (ridge)
    coef_rows = []
    for d in bt_out["fold_diag_by_model"]["ridge"]:
        if d["coefficients"] is not None:
            row = d["coefficients"].to_dict()
            row["fold_end"] = d["test_start"]
            coef_rows.append(row)
    if coef_rows:
        pd.DataFrame(coef_rows).set_index("fold_end").to_csv(Path(args.reports_dir) / "results" / "ridge_coefficients_by_fold.csv")

    logger.info("Backtest stage complete. Summary written to reports/results/summary.json")
    return bt_out, feat


def cmd_report(args):
    setup_logging(args.verbose)
    # Delegate to the dedicated report-building script to keep this module
    # focused on data/backtest orchestration. ``scripts/`` is not a package,
    # so it is added to sys.path explicitly.
    project_root = Path(__file__).resolve().parents[2]
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import build_report  # type: ignore  # noqa: E402

    build_report.main(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graph-diffusion-signal", description="Cross-Asset Graph Diffusion Signals CLI")
    p.add_argument("--config-dir", default="config")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("data", help="Download/cache and clean OHLCV data.")
    d.add_argument("--force-refresh", action="store_true")
    d.set_defaults(func=cmd_data)

    b = sub.add_parser("backtest", help="Run feature engineering + walk-forward backtest.")
    b.add_argument("--force-refresh", action="store_true")
    b.set_defaults(func=cmd_backtest)

    r = sub.add_parser("report", help="Build the markdown + PDF research report from saved results.")
    r.set_defaults(func=cmd_report)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
