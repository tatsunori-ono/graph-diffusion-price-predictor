"""Shared configuration loading utilities.

Both ``universe.yaml`` and ``backtest.yaml`` are plain YAML files. Keeping the
loading logic in one place avoids every module reimplementing path handling
and keeps error messages consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and raise a clear error if it is missing or malformed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Run commands from the repository "
            f"root, or pass an explicit --config path."
        )
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping.")
    return data


@dataclass
class UniverseConfig:
    start_date: str
    end_date: Optional[str]
    sectors: dict[str, list[str]]
    benchmarks: list[str]
    min_history_days: int = 500

    @classmethod
    def from_yaml(cls, path: str | Path = "config/universe.yaml") -> "UniverseConfig":
        raw = load_yaml(path)
        return cls(
            start_date=raw["start_date"],
            end_date=raw.get("end_date"),
            sectors=raw["sectors"],
            benchmarks=raw["benchmarks"],
            min_history_days=int(raw.get("min_history_days", 500)),
        )

    @property
    def all_tickers(self) -> list[str]:
        out: list[str] = []
        for names in self.sectors.values():
            out.extend(names)
        return out

    @property
    def all_tickers_with_benchmarks(self) -> list[str]:
        return self.all_tickers + list(self.benchmarks)

    def ticker_sector(self, ticker: str) -> Optional[str]:
        for sector, names in self.sectors.items():
            if ticker in names:
                return sector
        return None

    def sector_benchmark(self, sector: str) -> str:
        """Pick a representative benchmark ETF for a sector's residualisation."""
        mapping = {
            "semiconductors_tech": "SMH",
            "japan_media_gaming_ip": "QQQ",
            "transport_logistics": "IYT",
        }
        return mapping.get(sector, "SPY")


@dataclass
class BacktestConfig:
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path = "config/backtest.yaml") -> "BacktestConfig":
        return cls(raw=load_yaml(path))

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)
