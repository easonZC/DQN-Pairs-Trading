"""Evaluation helpers for trained agents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from ..env.pairs_trading_env import PairsTradingEnv


@dataclass
class EvaluationResult:
    trades: List[Dict[str, object]]
    total_profit: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: np.ndarray


def calculate_trade_profit(trade: Dict[str, object]) -> float:
    position = int(trade["position"])
    volume_A = float(trade["volume_A"])
    volume_B = float(trade["volume_B"])
    entry_price_A = float(trade["entry_price_A"])
    entry_price_B = float(trade["entry_price_B"])
    exit_price_A = float(trade["exit_price_A"])
    exit_price_B = float(trade["exit_price_B"])

    if position == 1:
        profit = volume_A * (exit_price_A - entry_price_A) / entry_price_A + volume_B * (
            (entry_price_B - exit_price_B) / entry_price_B
        )
    elif position == -1:
        profit = volume_A * (entry_price_A - exit_price_A) / entry_price_A + volume_B * (
            (exit_price_B - entry_price_B) / entry_price_B
        )
    else:
        profit = 0.0
    return float(profit)


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    peaks = np.maximum.accumulate(cumulative_returns)
    drawdowns = (peaks - cumulative_returns) / peaks
    return float(np.max(drawdowns)) if len(drawdowns) else 0.0


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    std = np.std(excess_returns)
    if std == 0:
        return 0.0
    return float(np.mean(excess_returns) / std * np.sqrt(252))


def evaluate_agent(
    agent: torch.nn.Module,
    test_prices: Dict[str, np.ndarray],
    formation_window_size: int,
    trading_window_size: int,
    *,
    device: torch.device | None = None,
) -> EvaluationResult:
    """Evaluate a trained agent on the held-out price segment."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = agent.to(device)
    agent.eval()

    env = PairsTradingEnv(
        prices=test_prices,
        formation_window_size=formation_window_size,
        trading_window_size=trading_window_size,
    )

    state = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = agent(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            next_state, _, done, _ = env.step(action)
            state = next_state if next_state is not None else state

    profits = np.array([calculate_trade_profit(trade) for trade in env.trades])
    cumulative_returns = np.cumsum(profits)
    total_profit = float(np.sum(profits))
    max_drawdown = calculate_max_drawdown(cumulative_returns + 1)
    sharpe = calculate_sharpe_ratio(profits)

    return EvaluationResult(
        trades=env.trades,
        total_profit=total_profit,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        equity_curve=cumulative_returns,
    )
