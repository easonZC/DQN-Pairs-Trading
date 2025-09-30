"""Cointegration utilities extracted from the exploratory notebook."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def _johansen_significance_index(alpha: float) -> int:
    if np.isclose(alpha, 0.01):
        return 0
    if np.isclose(alpha, 0.05):
        return 1
    if np.isclose(alpha, 0.1):
        return 2
    raise ValueError("Johansen critical values only available for 1%, 5% and 10% levels.")


def find_cointegrated_pairs(
    sector_data: Dict[str, pd.DataFrame],
    sector: str,
    *,
    p_value_threshold: float = 0.02,
    t_stat_threshold: float = -3.5,
    johansen_alpha: float = 0.05,
    make_plots: bool = False,
) -> List[Dict[str, object]]:
    """Return validated cointegrated pairs for the requested sector."""

    if sector not in sector_data:
        raise KeyError(f"Sector '{sector}' not found in provided sector data.")

    close_prices = sector_data[sector]["Adj Close"]
    standardized_close = sector_data[sector]["Standardized Close"]

    validated_pairs_data: List[Dict[str, object]] = []
    significance_index = _johansen_significance_index(johansen_alpha)

    tickers = close_prices.columns
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            stock_a = tickers[i]
            stock_b = tickers[j]
            prices_a = close_prices[stock_a]
            prices_b = close_prices[stock_b]

            try:
                t_statistic, p_value, _ = sm.tsa.coint(prices_a, prices_b)
                if p_value >= p_value_threshold or t_statistic >= t_stat_threshold:
                    continue

                pair_prices = pd.concat([prices_a, prices_b], axis=1)
                johansen_test = coint_johansen(pair_prices, det_order=1, k_ar_diff=1)
                if johansen_test.lr1[0] <= johansen_test.cvt[0, significance_index]:
                    continue

                hedge_ratio = -johansen_test.evec[:, 0][1] / johansen_test.evec[:, 0][0]
                spread_standardized = standardized_close[stock_a] - hedge_ratio * standardized_close[stock_b]

                result: Dict[str, object] = {
                    "pair": (stock_a, stock_b),
                    "prices": pd.DataFrame({"stock_A": prices_a, "stock_B": prices_b}),
                    "hedge_ratio": float(hedge_ratio),
                    "spread_standardized": spread_standardized,
                }

                if make_plots:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(
                        spread_standardized.index,
                        spread_standardized,
                        label=f"Spread: {stock_a} - {hedge_ratio:.2f}*{stock_b}",
                    )
                    ax.axhline(spread_standardized.mean(), color="red", linestyle="--", label="Mean")
                    ax.set_title(f"Standardized Spread: {stock_a} vs {stock_b}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Standardized Spread")
                    ax.legend()
                    ax.grid(True)
                    fig.autofmt_xdate()
                    result["figure"] = fig

                validated_pairs_data.append(result)
            except Exception as exc:  # pragma: no cover - defensive guard
                # Continue scanning pairs if a statistical test fails to converge
                print(f"Error processing pair {stock_a}-{stock_b}: {exc}")
                continue

    return validated_pairs_data
