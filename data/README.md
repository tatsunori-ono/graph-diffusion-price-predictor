# Data directory

## `raw/`

One CSV per ticker (`<TICKER>.csv`), columns `Open, High, Low, Close, Adj Close, Volume`,
indexed by trading date. Populated by `make data` (or `python -m graph_diffusion_signal.cli data`).

* If Yahoo Finance is reachable, these are real adjusted OHLCV data via `yfinance`.
* If it is not reachable (e.g. a sandboxed/offline environment), the pipeline automatically falls
  back to a calibrated **synthetic** generator and writes `SYNTHETIC_DATA_NOTICE.txt` into this
  directory. Check for that file before treating anything downstream as real market data.
* Delete this directory (or pass `--force-refresh`) to force a fresh download attempt.

## `processed/`

Cleaned, aligned wide-format CSVs (`prices_wide.csv`, `volumes_wide.csv`) produced by the data stage
-- common trading calendar, insufficient-history tickers dropped, short gaps forward-filled. See
`src/graph_diffusion_signal/data.py::clean_and_align` for the exact cleaning logic and its
limitations (no survivorship-bias correction; see README "Limitations").

Both subdirectories are gitignored by default (see `.gitignore`) since they are derived/cache data,
not source -- regenerate with `make data`.
