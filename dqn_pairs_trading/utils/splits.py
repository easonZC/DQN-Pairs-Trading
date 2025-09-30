"""Utility helpers for splitting price histories."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def train_test_split_prices(prices: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split a two-column price DataFrame into train/test dictionaries."""

    if not {"stock_A", "stock_B"}.issubset(prices.columns):
        raise ValueError("prices DataFrame must contain 'stock_A' and 'stock_B' columns.")

    split_index = len(prices) // 2
    train = prices.iloc[:split_index]
    test = prices.iloc[split_index:]

    train_dict = {
        "stock_A": train["stock_A"].to_numpy(),
        "stock_B": train["stock_B"].to_numpy(),
        "dates": train.index.to_numpy(),
    }
    test_dict = {
        "stock_A": test["stock_A"].to_numpy(),
        "stock_B": test["stock_B"].to_numpy(),
        "dates": test.index.to_numpy(),
    }
    return train_dict, test_dict
