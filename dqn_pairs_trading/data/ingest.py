"""Data ingestion utilities for the pairs trading project.

This module centralises the notebook logic for pulling and cleaning
historical equity data so the rest of the pipeline can reuse a single
implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler


DEFAULT_LOOKBACK_YEARS = 10


@dataclass(frozen=True)
class SectorUniverse:
    """Collection of tickers grouped by sector."""

    sectors: Mapping[str, Tuple[str, ...]]

    @classmethod
    def default(cls) -> "SectorUniverse":
        """Return the default sector universe used in the original notebook."""

        return cls(
            sectors={
                "Technology": (
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "NVDA",
                    "META",
                    "TSM",
                    "ORCL",
                    "ADBE",
                    "CSCO",
                    "INTC",
                    "IBM",
                    "SAP",
                    "CRM",
                    "AMD",
                    "TXN",
                    "QCOM",
                    "AVGO",
                    "NOW",
                    "INTU",
                    "SHOP",
                    "WDAY",
                    "CDNS",
                    "PANW",
                    "SNOW",
                    "SQ",
                    "ZM",
                    "ASML",
                    "MU",
                    "XLNX",
                    "DOCU",
                ),
                "Healthcare": (
                    "JNJ",
                    "PFE",
                    "MRK",
                    "ABBV",
                    "LLY",
                    "TMO",
                    "UNH",
                    "ABT",
                    "BMY",
                    "CVS",
                    "DHR",
                    "AMGN",
                    "GILD",
                    "BIIB",
                    "ZTS",
                    "BDX",
                    "ISRG",
                    "SYK",
                    "BSX",
                    "CI",
                    "HCA",
                    "REGN",
                    "VRTX",
                    "MDT",
                    "MCK",
                    "ILMN",
                    "RHHBY",
                    "ABC",
                    "COO",
                    "TFX",
                ),
                "Real Estate": (
                    "AMT",
                    "PLD",
                    "CCI",
                    "DLR",
                    "EQIX",
                    "O",
                    "SPG",
                    "WELL",
                    "AVB",
                    "EQR",
                    "EXR",
                    "IRM",
                    "ESS",
                    "BXP",
                    "ARE",
                    "PSA",
                    "VTR",
                    "MAA",
                    "HST",
                    "UDR",
                    "SUI",
                    "CPT",
                    "STOR",
                    "KIM",
                    "SLG",
                    "PEAK",
                    "REG",
                    "SRC",
                    "AIV",
                    "BRX",
                ),
            }
        )


def compute_default_date_range(lookback_years: int = DEFAULT_LOOKBACK_YEARS) -> Tuple[str, str]:
    """Return `(start_date, end_date)` strings for `yfinance.download`."""

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.DateOffset(years=lookback_years)).strftime("%Y-%m-%d")
    return start_date, end_date


def download_sector_data(
    universe: SectorUniverse,
    start_date: str,
    end_date: str,
    *,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Download adjusted close and volume data for each sector."""

    downloaded: Dict[str, pd.DataFrame] = {}
    for sector, tickers in universe.sectors.items():
        data = yf.download(
            list(tickers), start=start_date, end=end_date, progress=progress
        )[["Adj Close", "Volume"]]
        downloaded[sector] = data.dropna(how="all")
    return downloaded


def _drop_inconsistent_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Remove tickers that contain missing observations."""

    adj_close = panel["Adj Close"].copy()
    volume = panel["Volume"].copy()

    valid_columns = adj_close.columns[adj_close.notna().all()]
    adj_close = adj_close[valid_columns]
    volume = volume[valid_columns]

    adj_close = adj_close.dropna()
    volume = volume.loc[adj_close.index]

    return pd.concat({"Adj Close": adj_close, "Volume": volume}, axis=1)


def clean_sector_data(raw_data: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Enforce consistent history across tickers for every sector."""

    cleaned: Dict[str, pd.DataFrame] = {}
    for sector, panel in raw_data.items():
        cleaned[sector] = _drop_inconsistent_columns(panel)
    return cleaned


def add_standardised_prices(
    sector_data: Mapping[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Append z-scored adjusted closes to each sector panel."""

    enriched: Dict[str, pd.DataFrame] = {}
    for sector, panel in sector_data.items():
        raw_close = panel["Adj Close"].copy()
        scaler = StandardScaler()
        standardised = pd.DataFrame(
            scaler.fit_transform(raw_close),
            index=raw_close.index,
            columns=raw_close.columns,
        )
        enriched[sector] = pd.concat(
            {"Adj Close": raw_close, "Standardized Close": standardised, "Volume": panel["Volume"]},
            axis=1,
        )
    return enriched


def load_sector_data(
    *,
    universe: SectorUniverse | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Download, clean, and standardise sector price data."""

    universe = universe or SectorUniverse.default()
    if start_date is None or end_date is None:
        start_date, end_date = compute_default_date_range()

    downloaded = download_sector_data(universe, start_date, end_date, progress=progress)
    cleaned = clean_sector_data(downloaded)
    return add_standardised_prices(cleaned)
