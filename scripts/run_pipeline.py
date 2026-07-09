#!/usr/bin/env python3
"""End-to-end pipeline entrypoint: data -> features -> walk-forward backtest.

Equivalent to running:
    python -m graph_diffusion_signal.cli data
    python -m graph_diffusion_signal.cli backtest

but as a single script for convenience (this is what ``make backtest`` calls).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graph_diffusion_signal import cli  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run the full graph-diffusion-signal pipeline.")
    parser.add_argument("--config-dir", default=str(ROOT / "config"))
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cli.setup_logging(args.verbose)
    t0 = time.time()

    uni, bt = cli.load_configs(args.config_dir)
    prices, volumes, used_synthetic, sector_of = cli.stage_data(
        uni, bt, args.data_dir, force_refresh=args.force_refresh
    )
    feat = cli.stage_features(uni, bt, prices, volumes, sector_of)
    bt_out = cli.stage_backtest(uni, bt, feat)
    cli.save_backtest_results(bt_out, Path(args.reports_dir) / "results")

    import json
    import pandas as pd
    from graph_diffusion_signal import metrics as metrics_mod

    summary = {
        "used_synthetic_data": bool(used_synthetic),
        "n_tickers": int(prices.shape[1]),
        "n_days": int(prices.shape[0]),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    for name, port in bt_out["results"].items():
        summary[name] = {
            "net": metrics_mod.summary_table(
                port["net_returns"], benchmark_returns=bt_out["benchmarks"].get("SPY"),
                turnover=port["turnover"], long_exposure=port["long_exposure"], short_exposure=port["short_exposure"],
            ),
            "gross": metrics_mod.summary_table(port["gross_returns"]),
        }
    for name, series in bt_out["benchmarks"].items():
        summary[f"benchmark_{name}"] = metrics_mod.summary_table(series)

    cost_grid = bt["costs"]["sensitivity_bps"]
    cost_table = cli.cost_sensitivity_table(
        bt_out["results"]["ridge"]["gross_returns"], bt_out["results"]["ridge"]["turnover"], cost_grid
    )
    cost_table.to_csv(Path(args.reports_dir) / "results" / "cost_sensitivity.csv", index=False)

    param_grid = cli.graph_parameter_sensitivity(
        feat["returns_universe"], feat["residual_returns"],
        bt["robustness"]["corr_window_grid"], bt["robustness"]["graph_topk_grid"], bt["graph"]["min_abs_corr"],
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

    coef_rows = []
    for d in bt_out["fold_diag_by_model"]["ridge"]:
        if d["coefficients"] is not None:
            row = d["coefficients"].to_dict()
            row["fold_end"] = d["test_start"]
            coef_rows.append(row)
    if coef_rows:
        pd.DataFrame(coef_rows).set_index("fold_end").to_csv(
            Path(args.reports_dir) / "results" / "ridge_coefficients_by_fold.csv"
        )

    print(f"\nPipeline complete in {time.time() - t0:.1f}s. Synthetic data used: {used_synthetic}")
    print(f"Results written to {Path(args.reports_dir) / 'results'}")


if __name__ == "__main__":
    main()
