"""Baseline mean-reversion strategy for benchmarking."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ..evaluation.evaluate import calculate_trade_profit


def traditional_pairs_trading(
    prices: pd.DataFrame,
    *,
    trading_boundaries: float = 2.0,
    stop_loss: float = 4.0,
) -> Dict[str, object]:
    """Simple threshold-based pairs trading baseline."""

    prices_A = prices["stock_A"].values
    prices_B = prices["stock_B"].values

    model = LinearRegression()
    model.fit(prices_B.reshape(-1, 1), prices_A)
    hedge_ratio = model.coef_[0]

    spread = prices_A - hedge_ratio * prices_B
    spread_mean = np.mean(spread)
    spread_std = np.std(spread) or 1e-8

    position = 0
    entry_price_A = None
    entry_price_B = None
    trades: List[Dict[str, object]] = []

    for price_A, price_B, current_date in zip(
        prices["stock_A"], prices["stock_B"], prices.index
    ):
        zscore = (price_A - hedge_ratio * price_B - spread_mean) / spread_std

        if position == 0:
            if trading_boundaries <= zscore < stop_loss:
                position = -1
                entry_price_A = price_A
                entry_price_B = price_B
                entry_date = current_date
            elif -stop_loss < zscore <= -trading_boundaries:
                position = 1
                entry_price_A = price_A
                entry_price_B = price_B
                entry_date = current_date
            continue

        exit_reason = None
        if position == 1 and zscore >= 0:
            exit_reason = "mean_revert"
        elif position == 1 and zscore <= -stop_loss:
            exit_reason = "stop_loss"
        elif position == -1 and zscore <= 0:
            exit_reason = "mean_revert"
        elif position == -1 and zscore >= stop_loss:
            exit_reason = "stop_loss"

        if exit_reason is None:
            continue

        trade = {
            "entry_date": entry_date,
            "exit_date": current_date,
            "position": position,
            "entry_price_A": entry_price_A,
            "entry_price_B": entry_price_B,
            "exit_price_A": price_A,
            "exit_price_B": price_B,
            "volume_A": 1.0,
            "volume_B": abs(hedge_ratio),
            "action": exit_reason,
        }
        trade["profit"] = calculate_trade_profit(trade)
        trades.append(trade)
        position = 0

    profits = np.array([trade["profit"] for trade in trades])
    return {
        "trades": trades,
        "total_profit": float(np.sum(profits)),
        "equity_curve": np.cumsum(profits),
    }
