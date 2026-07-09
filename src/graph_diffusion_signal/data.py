"""Data acquisition, caching, and cleaning.

Design goals
------------
1. Prefer live data from ``yfinance`` (Yahoo Finance), which is free and needs
   no API key.
2. Cache every ticker's raw OHLCV to ``data/raw/<TICKER>.csv`` so re-runs are
   fast and reproducible without hitting the network again.
3. If Yahoo Finance is unreachable (e.g. no network, or the deployment
   environment blocks outbound calls to finance endpoints), fall back to a
   clearly-labelled *synthetic* data generator so the rest of the pipeline
   remains runnable end to end.

Synthetic fallback and honesty
-------------------------------
The synthetic generator is a calibrated multi-factor stochastic model (market
factor + sector factor + idiosyncratic Student-t noise, with simple GARCH-like
volatility clustering). It reproduces realistic *contemporaneous* sector
correlation and volatility, but it does **not** inject any artificial
lagged/lead-lag predictability between names. This is a deliberate choice: it
would be easy to hard-code a fake diffusion effect into synthetic data and
"discover" it later, but that would defeat the point of an honest research
project. When synthetic data is used, a ``SYNTHETIC_DATA_NOTICE.txt`` file is
written into ``data/raw/`` and every downstream artefact (README, PDF report)
must say so explicitly. Whoever runs this pipeline with working internet
access will transparently get real Yahoo Finance data instead -- the code
path is identical.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
SYNTHETIC_NOTICE_FILE = "SYNTHETIC_DATA_NOTICE.txt"


class DataUnavailableError(RuntimeError):
    """Raised when neither live data nor a cache nor synthetic fallback works."""


# --------------------------------------------------------------------------- #
# Live download via yfinance
# --------------------------------------------------------------------------- #
def _download_one_yfinance(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise DataUnavailableError(f"yfinance returned no rows for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataUnavailableError(f"yfinance response for {ticker} missing columns {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df.index.name = "Date"
    return df


def _probe_network(timeout: float = 6.0) -> bool:
    """Cheap check of whether Yahoo Finance is reachable before looping over
    the whole universe -- avoids a slow per-ticker timeout cascade."""
    import urllib.request

    try:
        req = urllib.request.Request(
            "https://query1.finance.yahoo.com/v8/finance/chart/SPY",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Yahoo Finance endpoint not reachable (%s).", exc)
        return False


# --------------------------------------------------------------------------- #
# Synthetic fallback generator
# --------------------------------------------------------------------------- #
def _stable_seed(ticker: str, base_seed: int) -> int:
    return (base_seed + sum(ord(c) for c in ticker) * 97) % (2**32 - 1)


def _garch_like_vol(n: int, rng: np.random.Generator, base_vol: float) -> np.ndarray:
    """Simple GARCH(1,1)-flavoured volatility path for realistic vol clustering."""
    omega, alpha, beta = base_vol**2 * 0.05, 0.08, 0.90
    var = np.empty(n)
    var[0] = base_vol**2
    shocks = rng.standard_normal(n)
    for t in range(1, n):
        var[t] = omega + alpha * (var[t - 1] * shocks[t - 1] ** 2) + beta * var[t - 1]
        var[t] = max(var[t], 1e-8)
    return np.sqrt(var)


def generate_synthetic_universe(
    tickers: list[str],
    sector_of: dict[str, str],
    benchmarks: list[str],
    start: str,
    end: str | None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate a calibrated synthetic OHLCV panel for the whole universe.

    Structure: r_i(t) = beta_mkt_i * market(t) + beta_sec_i * sector_factor(t)
    + idio_i(t), with idio_i drawn from a Student-t distribution and its own
    GARCH-like volatility path. No cross-sectional lag structure is injected,
    so any predictive signal later found by the pipeline on this dataset
    reflects noise/estimation artefacts, not a real (or fake) planted effect.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    rng_master = np.random.default_rng(seed)

    # Market factor: small positive drift, GARCH-like vol clustering.
    mkt_vol = _garch_like_vol(n, np.random.default_rng(seed + 1), base_vol=0.010)
    market = rng_master.standard_normal(n) * mkt_vol + 0.00025

    sectors = sorted(set(sector_of.values())) if sector_of else []
    sector_factors: dict[str, np.ndarray] = {}
    for si, sector in enumerate(sectors):
        rng_s = np.random.default_rng(seed + 100 + si)
        sec_vol = _garch_like_vol(n, np.random.default_rng(seed + 200 + si), base_vol=0.009)
        beta_to_mkt = rng_s.uniform(0.5, 0.9)
        idio_sector = rng_s.standard_normal(n) * sec_vol
        sector_factors[sector] = beta_to_mkt * market + idio_sector

    panels: dict[str, pd.DataFrame] = {}

    def _simulate_one(ticker: str, beta_mkt: float, beta_sec: float | None,
                       sector_factor: np.ndarray | None, base_vol: float,
                       start_price: float) -> pd.DataFrame:
        rng_t = np.random.default_rng(_stable_seed(ticker, seed))
        idio_vol = _garch_like_vol(n, rng_t, base_vol=base_vol)
        # Student-t idiosyncratic noise for fat tails, scaled to target vol.
        t_dof = 5
        idio_raw = rng_t.standard_t(t_dof, size=n)
        idio_raw = idio_raw / idio_raw.std()
        idio = idio_raw * idio_vol
        drift = rng_t.normal(0.0002, 0.0002)
        r = beta_mkt * market + drift + idio
        if beta_sec is not None and sector_factor is not None:
            r = r + beta_sec * sector_factor
        price = start_price * np.exp(np.cumsum(r))
        # Simple realistic OHLC construction from close-to-close returns.
        close = price
        open_ = np.empty(n)
        open_[0] = start_price
        open_[1:] = close[:-1] * (1 + rng_t.normal(0, 0.0015, n - 1))
        intraday_range = np.abs(rng_t.normal(0, 0.007, n)) * close
        high = np.maximum(open_, close) + intraday_range * rng_t.uniform(0.2, 1.0, n)
        low = np.minimum(open_, close) - intraday_range * rng_t.uniform(0.2, 1.0, n)
        low = np.maximum(low, 0.01)
        base_volume = rng_t.uniform(2e6, 3e7)
        vol_ar = np.empty(n)
        vol_ar[0] = base_volume
        vshock = rng_t.lognormal(mean=0.0, sigma=0.35, size=n)
        for t in range(1, n):
            vol_ar[t] = 0.85 * vol_ar[t - 1] + 0.15 * base_volume
        volume = np.maximum(vol_ar * vshock * (1 + 2.5 * np.abs(r)), 1000).astype(np.int64)

        df = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Adj Close": close,
                "Volume": volume,
            },
            index=dates,
        )
        df.index.name = "Date"
        return df

    for ticker in tickers:
        sector = sector_of.get(ticker)
        rng_p = np.random.default_rng(_stable_seed(ticker, seed) + 1)
        beta_mkt = rng_p.uniform(0.7, 1.5)
        beta_sec = rng_p.uniform(0.4, 0.9) if sector else None
        base_vol = rng_p.uniform(0.016, 0.032)
        start_price = rng_p.uniform(20, 400)
        panels[ticker] = _simulate_one(
            ticker, beta_mkt, beta_sec,
            sector_factors.get(sector) if sector else None,
            base_vol, start_price,
        )

    # Benchmarks: broad market-like blends of the simulated factors so their
    # correlation with the universe is realistic.
    bench_specs = {
        "SPY": (1.0, None, 0.011, 450.0),
        "QQQ": (1.15, "semiconductors_tech", 0.013, 380.0),
        "SMH": (1.25, "semiconductors_tech", 0.020, 150.0),
        "IYT": (0.95, "transport_logistics", 0.014, 220.0),
    }
    for b in benchmarks:
        beta_mkt, sec_key, base_vol, start_price = bench_specs.get(
            b, (1.0, None, 0.012, 100.0)
        )
        sf = sector_factors.get(sec_key) if sec_key in sector_factors else None
        beta_sec = 0.3 if sf is not None else None
        panels[b] = _simulate_one(b, beta_mkt, beta_sec, sf, base_vol, start_price)

    return panels


# --------------------------------------------------------------------------- #
# Cache I/O
# --------------------------------------------------------------------------- #
def _cache_path(raw_dir: Path, ticker: str) -> Path:
    return raw_dir / f"{ticker.replace('/', '_')}.csv"


def load_cached(raw_dir: Path, ticker: str) -> pd.DataFrame | None:
    path = _cache_path(raw_dir, ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def save_cached(raw_dir: Path, ticker: str, df: pd.DataFrame) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(_cache_path(raw_dir, ticker))


def download_universe(
    tickers: list[str],
    sector_of: dict[str, str],
    benchmarks: list[str],
    start: str,
    end: str | None,
    raw_dir: str | Path,
    seed: int = 42,
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], bool]:
    """Fetch (or load-from-cache) OHLCV for every ticker + benchmark.

    Returns ``(panels, used_synthetic)``.
    """
    raw_dir = Path(raw_dir)
    all_names = list(dict.fromkeys(list(tickers) + list(benchmarks)))
    notice_path = raw_dir / SYNTHETIC_NOTICE_FILE

    if not force_refresh and all(_cache_path(raw_dir, t).exists() for t in all_names):
        logger.info("Loading %d tickers from cache at %s", len(all_names), raw_dir)
        panels = {t: load_cached(raw_dir, t) for t in all_names}
        return panels, notice_path.exists()

    network_ok = _probe_network()
    panels: dict[str, pd.DataFrame] = {}
    used_synthetic = False

    if network_ok:
        logger.info("Yahoo Finance reachable -- downloading %d tickers live.", len(all_names))
        for t in all_names:
            try:
                df = _download_one_yfinance(t, start, end)
                panels[t] = df
                save_cached(raw_dir, t, df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Live download failed for %s (%s); will fall back.", t, exc)
                network_ok = False
                break

    if not network_ok or len(panels) < len(all_names):
        logger.warning(
            "Falling back to calibrated SYNTHETIC data for %d tickers. "
            "This happens automatically when Yahoo Finance cannot be reached "
            "(e.g. sandboxed / offline environments). Re-run with internet "
            "access to fetch real data -- the rest of the pipeline is identical.",
            len(all_names),
        )
        panels = generate_synthetic_universe(
            tickers=list(tickers), sector_of=sector_of, benchmarks=list(benchmarks),
            start=start, end=end, seed=seed,
        )
        used_synthetic = True
        raw_dir.mkdir(parents=True, exist_ok=True)
        notice_path.write_text(
            "SYNTHETIC DATA IN USE\n"
            "======================\n"
            "The CSV files in this directory were generated by "
            "graph_diffusion_signal.data.generate_synthetic_universe because "
            "live Yahoo Finance data could not be reached from this "
            "environment.\n\n"
            "The synthetic model uses a market factor + sector factor + "
            "Student-t idiosyncratic noise with GARCH-like volatility "
            "clustering, calibrated to look like realistic equity returns. "
            "It deliberately does NOT contain any injected lead-lag / "
            "diffusion effect between names, so any signal the pipeline "
            "finds on this data should be interpreted purely as a "
            "pipeline-correctness demo, not as evidence about real markets.\n\n"
            "Delete this directory (or pass --force-refresh) and re-run "
            "`make data` with a working internet connection to fetch real "
            "adjusted OHLCV data via yfinance instead.\n"
        )
        for t, df in panels.items():
            save_cached(raw_dir, t, df)

    return panels, used_synthetic


# --------------------------------------------------------------------------- #
# Cleaning / alignment
# --------------------------------------------------------------------------- #
def clean_and_align(
    panels: dict[str, pd.DataFrame],
    min_history_days: int = 500,
    price_field: str = "Adj Close",
) -> pd.DataFrame:
    """Align all tickers onto a common trading calendar and return a wide
    price DataFrame (columns = tickers, index = dates).

    Cleaning steps (all using only each series' own historical values, never
    future values, so this step introduces no look-ahead bias):
    1. Drop tickers with fewer than ``min_history_days`` observed rows.
    2. Outer-join onto the union of trading calendars actually observed
       across the universe (not a fabricated calendar), then forward-fill
       short gaps (<=3 consecutive days) to handle isolated missing-data
       points such as data-vendor holes; longer gaps are left as NaN.
    3. Drop any date where the surviving cross-section is mostly missing
       (kept at the very start of history before enough names have IPO'd).
    """
    kept = {}
    for t, df in panels.items():
        if price_field not in df.columns:
            logger.warning("Ticker %s missing field %s -- dropping.", t, price_field)
            continue
        n_obs = df[price_field].dropna().shape[0]
        if n_obs < min_history_days:
            logger.warning(
                "Dropping %s: only %d observations (< min_history_days=%d). "
                "Note this is a data-sufficiency filter, not a survivorship claim "
                "-- see README limitations.",
                t, n_obs, min_history_days,
            )
            continue
        kept[t] = df[price_field]

    if not kept:
        raise DataUnavailableError("No tickers survived the min_history_days filter.")

    wide = pd.DataFrame(kept).sort_index()
    wide = wide.ffill(limit=3)
    # Require at least 70% of the (kept) universe present to keep a date row;
    # this trims the very early period before most names have data.
    min_names = max(1, int(0.7 * wide.shape[1]))
    wide = wide.dropna(thresh=min_names, axis=0)
    wide = wide.dropna(axis=1, how="any") if wide.isna().any().any() else wide
    # Any remaining sparse names (still NaN after ffill/threshold trim) are
    # dropped outright rather than interpolated, to avoid inventing prices.
    still_bad = wide.columns[wide.isna().any()].tolist()
    if still_bad:
        logger.warning("Dropping %s after alignment: residual gaps remain.", still_bad)
        wide = wide.drop(columns=still_bad)

    return wide


def align_volume(
    panels: dict[str, pd.DataFrame], kept_tickers: list[str], common_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Align raw Volume onto the same (tickers, dates) grid used for prices."""
    cols = {}
    for t in kept_tickers:
        if t in panels and "Volume" in panels[t].columns:
            cols[t] = panels[t]["Volume"]
    vol = pd.DataFrame(cols).reindex(common_index)
    vol = vol.ffill(limit=3)
    return vol


def compute_simple_returns(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from adjusted close prices. ``returns.loc[t]`` is
    the return realised BETWEEN t-1 and t, i.e. it is fully known only as of
    the close of day t -- this convention is used consistently everywhere
    downstream to keep the leakage bookkeeping simple."""
    return prices_wide.pct_change().dropna(how="all")


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    logger.info("Saved processed file: %s (%s rows)", path, len(df))
