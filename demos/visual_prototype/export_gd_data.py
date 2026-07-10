#!/usr/bin/env python3
"""
Export graph-diffusion-driven features and next-day price predictions for a small universe of stocks.

This script is intentionally:
- Stable on NumPy 2.x (no np.trapz)
- Stable numerically (sanitizes NaNs/Infs, clips z-scores)
- Simple and transparent (graph diffusion + Ridge regression)

Outputs:
- gd_data.npz     (arrays for Manim + Streamlit)
- gd_meta.json    (human-readable metadata)

Typical usage:
  python3 export_gd_data.py \
    --start 2022-01-03 --end 2022-06-17 \
    --tickers AAPL MSFT AMZN GOOG NVDA \
    --target AAPL \
    --csv demo_prices.csv \
    --out gd_data.npz
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Optional import (only needed for live download)
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

import networkx as nx  # type: ignore
from sklearn.linear_model import Ridge  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def _ensure_2d_prices(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        return df.to_frame()
    return df


def download_prices_yf(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not available. Install it or use --csv/--no-yf.")
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    # yfinance returns:
    # - multiindex columns for many tickers
    # - or single column for one ticker
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close" in data.columns.get_level_values(0)):
            close = data["Close"]
        else:
            # fallback
            close = data.xs("Close", axis=1, level=0, drop_level=True)
    else:
        # single ticker
        close = data["Close"]
    close = _ensure_2d_prices(close)
    close = close.sort_index()
    close = close.dropna(how="all")
    close = close.ffill().dropna()
    # Ensure correct column order
    close = close[[c for c in tickers if c in close.columns]]
    return close


def load_prices_csv(path: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("CSV must have a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if tickers is not None and len(tickers) > 0:
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            raise ValueError(f"CSV missing tickers: {missing}")
        df = df[tickers]
    df = df.ffill().dropna()
    return df


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # log returns, stable for multiplicative modeling
    lr = np.log(prices).diff()
    lr = lr.dropna()
    return lr


def safe_corr(window_returns: np.ndarray) -> np.ndarray:
    """
    window_returns: shape (W, N)
    returns correlation matrix (N, N) with NaNs -> 0 and diag=1.
    """
    corr = np.corrcoef(window_returns, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def corr_to_adj(corr: np.ndarray, mode: str = "abs", thr: float = 0.2) -> np.ndarray:
    A = corr.copy()
    np.fill_diagonal(A, 0.0)
    if mode == "abs":
        A = np.abs(A)
    elif mode == "pos":
        A = np.clip(A, 0.0, None)
    elif mode == "signed":
        # keep sign but threshold on abs
        pass
    else:
        raise ValueError("mode must be one of: abs, pos, signed")
    if mode == "signed":
        mask = np.abs(A) < thr
        A[mask] = 0.0
    else:
        A[A < thr] = 0.0
    # ensure symmetry
    A = 0.5 * (A + A.T)
    return A


def row_normalize(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = W.sum(axis=1, keepdims=True)
    return W / (s + eps)


def clip_finite(x: np.ndarray, clip: float = 6.0) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -clip, clip)


def zscore_cross_section(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x)) + eps
    return (x - m) / s


def diffusion_iter(A_rw: np.ndarray, x0: np.ndarray, mu: float, steps: int) -> np.ndarray:
    """
    Iterative diffusion/message passing:
      x_{s+1} = (1-mu) x_s + mu A x_s
    returns array (steps+1, N)
    """
    xs = [clip_finite(x0)]
    x = xs[0].copy()
    for _ in range(steps):
        x = (1.0 - mu) * x + mu * (A_rw @ x)
        x = clip_finite(x)
        xs.append(x.copy())
    return np.stack(xs, axis=0)


def build_layout_from_average(W_avg: np.ndarray, tickers: List[str], seed: int = 7) -> np.ndarray:
    """
    Build a stable 2D layout with networkx spring_layout from the average adjacency.
    Returns (N, 2), centered and scaled to roughly [-1, 1].
    """
    N = len(tickers)
    G = nx.Graph()
    for i, t in enumerate(tickers):
        G.add_node(i, label=t)
    for i in range(N):
        for j in range(i + 1, N):
            w = float(W_avg[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)
    pos = nx.spring_layout(G, seed=seed, weight="weight")  # dict idx->(x,y)
    pts = np.array([pos[i] for i in range(N)], dtype=float)
    pts -= pts.mean(axis=0, keepdims=True)
    mx = float(np.max(np.abs(pts)) + 1e-9)
    pts /= mx
    return pts


def make_model(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD), exclusive-ish for yfinance")
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--target", default=None, help="Target ticker for display; defaults to first ticker")
    ap.add_argument("--csv", default=None, help="Optional local CSV with Date + ticker columns")
    ap.add_argument("--no-yf", action="store_true", help="Do not use yfinance (requires --csv)")
    ap.add_argument("--window", type=int, default=60, help="Rolling window size for correlations")
    ap.add_argument("--thr", type=float, default=0.20, help="Adjacency threshold on corr magnitude")
    ap.add_argument("--adj-mode", type=str, default="abs", choices=["abs", "pos", "signed"])
    ap.add_argument("--mu", type=float, default=0.55, help="Diffusion mixing (0..1)")
    ap.add_argument("--steps", type=int, default=6, help="Number of diffusion steps to animate")
    ap.add_argument("--ridge-alpha", type=float, default=5.0, help="Ridge regularization")
    ap.add_argument("--min-train", type=int, default=80, help="Min samples before first walk-forward prediction")
    ap.add_argument("--stride", type=int, default=1, help="Stride for exported animation frames")
    ap.add_argument("--layout-seed", type=int, default=7)
    ap.add_argument("--out", default="gd_data.npz")
    args = ap.parse_args()

    tickers: List[str] = list(args.tickers)
    target = args.target or tickers[0]
    if target not in tickers:
        raise ValueError(f"--target {target} not in tickers list.")

    # Load prices
    if args.csv is not None:
        prices = load_prices_csv(args.csv, tickers)
    else:
        if args.no_yf:
            raise ValueError("--no-yf requires --csv.")
        prices = download_prices_yf(tickers, args.start, args.end)

    if prices.shape[0] < args.window + 5:
        raise ValueError(f"Not enough data: have {prices.shape[0]} rows; need at least window+5.")

    rets = log_returns(prices)  # (T_ret, N)

    # We'll build samples on rets index; sample at time t predicts t+1
    dates = rets.index
    N = len(tickers)

    # Rolling correlations aligned with rets dates from window-1 onward
    corr_list: List[np.ndarray] = []
    W_list: List[np.ndarray] = []
    A_list: List[np.ndarray] = []
    x_steps_list: List[np.ndarray] = []
    x0_list: List[np.ndarray] = []
    h_list: List[np.ndarray] = []
    e_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []         # next-day raw log returns
    p_list: List[np.ndarray] = []         # current prices
    p_next_list: List[np.ndarray] = []    # next prices
    sample_dates: List[pd.Timestamp] = []

    # align prices to returns index (prices has one more row; but we can locate by date)
    prices_aligned = prices.reindex(rets.index).ffill().dropna()

    for t in range(args.window - 1, len(rets) - 1):
        date_t = dates[t]
        # corr window ending at t
        win = rets.iloc[t - args.window + 1 : t + 1].values  # (W, N)
        corr = safe_corr(win)
        W = corr_to_adj(corr, mode=args.adj_mode, thr=args.thr)
        A_rw = row_normalize(W)

        r_t = rets.iloc[t].values.astype(float)  # raw log returns
        x0 = zscore_cross_section(r_t)
        x0 = clip_finite(x0)

        xs = diffusion_iter(A_rw, x0, mu=args.mu, steps=args.steps)  # (steps+1,N)
        h = xs[-1]
        e = x0 - h
        e = clip_finite(e)

        y_next = rets.iloc[t + 1].values.astype(float)  # raw log returns

        p_t = prices_aligned.loc[date_t].values.astype(float)
        p_next = prices_aligned.loc[dates[t + 1]].values.astype(float)

        corr_list.append(corr)
        W_list.append(W)
        A_list.append(A_rw)
        x_steps_list.append(xs)
        x0_list.append(x0)
        h_list.append(h)
        e_list.append(e)
        y_list.append(y_next)
        p_list.append(p_t)
        p_next_list.append(p_next)
        sample_dates.append(date_t)

    # stack
    corr_arr = np.stack(corr_list, axis=0)           # (S, N, N)
    W_arr = np.stack(W_list, axis=0)                # (S, N, N)
    A_arr = np.stack(A_list, axis=0)                # (S, N, N)
    x_steps = np.stack(x_steps_list, axis=0)        # (S, steps+1, N)
    x0_arr = np.stack(x0_list, axis=0)              # (S, N)
    h_arr = np.stack(h_list, axis=0)                # (S, N)
    e_arr = np.stack(e_list, axis=0)                # (S, N)
    Y = np.stack(y_list, axis=0)                    # (S, N)
    P = np.stack(p_list, axis=0)                    # (S, N)
    P_next = np.stack(p_next_list, axis=0)          # (S, N)
    S = x0_arr.shape[0]

    # Features: concat(x0, h, e)
    X = np.concatenate([x0_arr, h_arr, e_arr], axis=1)  # (S, 3N)
    X = clip_finite(X)

    # Walk-forward predictions
    model = make_model(alpha=args.ridge_alpha)
    yhat = np.full_like(Y, fill_value=np.nan, dtype=float)

    min_train = max(10, int(args.min_train))
    if S <= min_train + 2:
        # fallback: fit once on first half
        split = max(10, S // 2)
        model.fit(X[:split], Y[:split])
        yhat[:] = model.predict(X)
        pred_mode = "single_fit"
    else:
        for i in range(min_train, S):
            model.fit(X[:i], Y[:i])
            yhat[i] = model.predict(X[i : i + 1])[0]
        # fill early region using first fitted model to avoid NaNs in animation
        model.fit(X[:min_train], Y[:min_train])
        yhat[:min_train] = model.predict(X[:min_train])
        pred_mode = "walk_forward"

    # Convert target predictions to price forecasts for display
    tgt_idx = tickers.index(target)
    pred_lr = yhat[:, tgt_idx]         # predicted log return
    true_lr = Y[:, tgt_idx]            # actual next log return
    p_tgt = P[:, tgt_idx]
    p_next_tgt = P_next[:, tgt_idx]
    p_hat_next = p_tgt * np.exp(pred_lr)

    # Layout from average adjacency
    W_avg = np.mean(W_arr, axis=0)
    layout = build_layout_from_average(W_avg, tickers, seed=int(args.layout_seed))  # (N,2)

    # Animation indices (stride)
    anim_idx = np.arange(0, S, int(args.stride))
    # keep as python int list for json friendliness
    anim_idx_list = [int(i) for i in anim_idx]

    # Export arrays
    out_path = args.out
    np.savez_compressed(
        out_path,
        dates=np.array([d.strftime("%Y-%m-%d") for d in sample_dates], dtype=object),
        tickers=np.array(tickers, dtype=object),
        target=np.array(target, dtype=object),
        target_idx=np.array(tgt_idx, dtype=np.int64),
        layout=layout.astype(np.float32),
        corr=corr_arr.astype(np.float32),
        W=W_arr.astype(np.float32),
        A=A_arr.astype(np.float32),
        x_steps=x_steps.astype(np.float32),
        x0=x0_arr.astype(np.float32),
        h=h_arr.astype(np.float32),
        e=e_arr.astype(np.float32),
        prices=P.astype(np.float32),
        prices_next=P_next.astype(np.float32),
        y_true=Y.astype(np.float32),
        y_pred=yhat.astype(np.float32),
        p_hat_next=p_hat_next.astype(np.float32),
        anim_idx=np.array(anim_idx_list, dtype=np.int64),
        mu=np.array(args.mu, dtype=np.float32),
        steps=np.array(args.steps, dtype=np.int64),
        thr=np.array(args.thr, dtype=np.float32),
        window=np.array(args.window, dtype=np.int64),
        ridge_alpha=np.array(args.ridge_alpha, dtype=np.float32),
        pred_mode=np.array(pred_mode, dtype=object),
    )

    meta = {
        "start": args.start,
        "end": args.end,
        "tickers": tickers,
        "target": target,
        "N": N,
        "samples": S,
        "window": int(args.window),
        "thr": float(args.thr),
        "adj_mode": args.adj_mode,
        "mu": float(args.mu),
        "steps": int(args.steps),
        "ridge_alpha": float(args.ridge_alpha),
        "min_train": int(args.min_train),
        "pred_mode": pred_mode,
        "stride": int(args.stride),
        "out": out_path,
        "equity_demo_note": "This project focuses on diffusion+prediction visualization, not a full trading backtest.",
        "target_price_last": float(p_tgt[-1]),
        "target_price_next_actual_last": float(p_next_tgt[-1]),
        "target_price_next_pred_last": float(p_hat_next[-1]),
    }
    with open("gd_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_path}")
    print("Saved: gd_meta.json")
    print(f"Samples: {S} | N={N} | steps={int(args.steps)} | pred_mode={pred_mode}")
    print(f"Target {target}: last P={p_tgt[-1]:.2f} | next actual={p_next_tgt[-1]:.2f} | next pred={p_hat_next[-1]:.2f}")


if __name__ == "__main__":
    main()
