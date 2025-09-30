"""Gym environment implementing the pairs trading simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from gym import spaces
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class BoundarySetting:
    trade: float
    stop_loss: float


DEFAULT_BOUNDARIES: Tuple[BoundarySetting, ...] = (
    BoundarySetting(1.0, 2.0),
    BoundarySetting(1.5, 2.5),
    BoundarySetting(2.0, 3.0),
    BoundarySetting(2.5, 3.5),
    BoundarySetting(3.0, 4.0),
    BoundarySetting(3.5, 4.5),
    BoundarySetting(4.0, 5.0),
)


class PairsTradingEnv(gym.Env):
    """Environment mirroring the behaviour from the original notebook."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        prices: Dict[str, np.ndarray],
        formation_window_size: int,
        trading_window_size: int,
        *,
        boundary_settings: Tuple[BoundarySetting, ...] = DEFAULT_BOUNDARIES,
    ) -> None:
        super().__init__()

        self.prices_A = np.asarray(prices["stock_A"], dtype=np.float64)
        self.prices_B = np.asarray(prices["stock_B"], dtype=np.float64)
        self.dates = np.asarray(prices["dates"])

        if not (len(self.prices_A) == len(self.prices_B) == len(self.dates)):
            raise ValueError("Prices and date arrays must have the same length.")

        self.formation_window_size = formation_window_size
        self.trading_window_size = trading_window_size
        self.boundary_settings = boundary_settings

        self.total_steps = len(self.prices_A) - (formation_window_size + trading_window_size)
        if self.total_steps <= 0:
            raise ValueError("Not enough data for the requested window sizes.")

        self.action_space = spaces.Discrete(len(boundary_settings))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.current_step: int = 0
        self.total_reward: float = 0.0
        self.trades: List[Dict[str, object]] = []
        self.position: int = 0
        self.entry_price_A: Optional[float] = None
        self.entry_price_B: Optional[float] = None
        self.hedge_ratio: float = 0.0
        self.spread_mean: float = 0.0
        self.spread_std: float = 1.0
        self.normal_close_count = 0
        self.stop_loss_count = 0
        self.exit_count = 0

    def reset(self):  # type: ignore[override]
        self.current_step = 0
        self.total_reward = 0.0
        self.trades = []
        self.position = 0
        self.entry_price_A = None
        self.entry_price_B = None
        self.normal_close_count = 0
        self.stop_loss_count = 0
        self.exit_count = 0
        return self._get_state()

    def _window_indices(self) -> Tuple[int, int, int]:
        start_index = self.current_step
        formation_end = start_index + self.formation_window_size
        trading_end = formation_end + self.trading_window_size
        return start_index, formation_end, trading_end

    def _prepare_windows(self) -> None:
        start_index, formation_end, trading_end = self._window_indices()

        formation_prices_A = self.prices_A[start_index:formation_end]
        formation_prices_B = self.prices_B[start_index:formation_end]

        regression = LinearRegression()
        regression.fit(formation_prices_B.reshape(-1, 1), formation_prices_A)
        self.hedge_ratio = float(regression.coef_[0])

        formation_spread = formation_prices_A - self.hedge_ratio * formation_prices_B
        self.spread_mean = float(np.mean(formation_spread))
        self.spread_std = float(np.std(formation_spread)) or 1e-8

        self.trading_prices_A = self.prices_A[formation_end:trading_end]
        self.trading_prices_B = self.prices_B[formation_end:trading_end]
        self.trading_dates = self.dates[formation_end:trading_end]

        self.start_index = start_index
        self.end_index = trading_end

    def _get_state(self) -> np.ndarray:
        self._prepare_windows()
        return np.array(
            [self.hedge_ratio, self.spread_mean, self.spread_std], dtype=np.float32
        )

    def step(self, action: int):  # type: ignore[override]
        boundaries = self.boundary_settings[action]
        reward = self._simulate_trading(boundaries.trade, boundaries.stop_loss)
        self.total_reward += reward

        self.current_step += self.trading_window_size
        done = self.current_step > self.total_steps
        next_state = None if done else self._get_state()

        return next_state, reward, done, {}

    def _simulate_trading(self, trade_boundary: float, stop_loss_boundary: float) -> float:
        total_reward = 0.0
        self.position = 0
        self.entry_price_A = None
        self.entry_price_B = None
        volume_A = 1.0
        volume_B = abs(self.hedge_ratio)
        entry_time: Optional[object] = None

        for price_A, price_B, current_date in zip(
            self.trading_prices_A, self.trading_prices_B, self.trading_dates
        ):
            spread_t = price_A - self.hedge_ratio * price_B
            zscore_t = (spread_t - self.spread_mean) / self.spread_std

            if self.position == 0:
                if trade_boundary <= zscore_t < stop_loss_boundary:
                    self.position = -1
                    self.entry_price_A = float(price_A)
                    self.entry_price_B = float(price_B)
                    entry_time = current_date
                elif -stop_loss_boundary < zscore_t <= -trade_boundary:
                    self.position = 1
                    self.entry_price_A = float(price_A)
                    self.entry_price_B = float(price_B)
                    entry_time = current_date
                continue

            action = None
            if self.position == 1:
                if zscore_t >= 0:
                    action = "normal_close"
                    self.normal_close_count += 1
                elif zscore_t <= -stop_loss_boundary:
                    action = "stop_loss"
                    self.stop_loss_count += 1
            elif self.position == -1:
                if zscore_t <= 0:
                    action = "normal_close"
                    self.normal_close_count += 1
                elif zscore_t >= stop_loss_boundary:
                    action = "stop_loss"
                    self.stop_loss_count += 1

            if action is None:
                continue

            reward = self._calculate_reward(
                self.entry_price_A,
                self.entry_price_B,
                float(price_A),
                float(price_B),
                volume_A,
                volume_B,
                self.position,
                action,
            )
            total_reward += reward

            self.trades.append(
                {
                    "entry_date": entry_time,
                    "exit_date": current_date,
                    "position": self.position,
                    "entry_price_A": self.entry_price_A,
                    "entry_price_B": self.entry_price_B,
                    "exit_price_A": float(price_A),
                    "exit_price_B": float(price_B),
                    "volume_A": volume_A,
                    "volume_B": volume_B,
                    "action": action,
                    "profit": reward,
                }
            )

            self.position = 0
            self.entry_price_A = None
            self.entry_price_B = None
            entry_time = None

        if self.position != 0:
            self.exit_count += 1
            reward = self._calculate_reward(
                self.entry_price_A,
                self.entry_price_B,
                float(self.trading_prices_A[-1]),
                float(self.trading_prices_B[-1]),
                volume_A,
                volume_B,
                self.position,
                "exit",
            )
            total_reward += reward
            self.trades.append(
                {
                    "entry_date": entry_time,
                    "exit_date": self.trading_dates[-1],
                    "position": self.position,
                    "entry_price_A": self.entry_price_A,
                    "entry_price_B": self.entry_price_B,
                    "exit_price_A": float(self.trading_prices_A[-1]),
                    "exit_price_B": float(self.trading_prices_B[-1]),
                    "volume_A": volume_A,
                    "volume_B": volume_B,
                    "action": "exit",
                    "profit": reward,
                }
            )
            self.position = 0
            self.entry_price_A = None
            self.entry_price_B = None

        return total_reward

    def _calculate_reward(
        self,
        entry_price_A: Optional[float],
        entry_price_B: Optional[float],
        current_price_A: float,
        current_price_B: float,
        volume_A: float,
        volume_B: float,
        position: int,
        action: str,
    ) -> float:
        if entry_price_A is None or entry_price_B is None:
            raise ValueError("Entry prices must be defined when calculating rewards.")

        if position == 1:
            pnl = volume_A * ((current_price_A - entry_price_A) / entry_price_A) + volume_B * (
                (entry_price_B - current_price_B) / entry_price_B
            )
        elif position == -1:
            pnl = volume_A * ((entry_price_A - current_price_A) / entry_price_A) + volume_B * (
                (current_price_B - entry_price_B) / entry_price_B
            )
        else:
            pnl = 0.0

        scale = 1000.0
        if action == "normal_close":
            reward = scale * pnl
        elif action == "stop_loss":
            penalty_multiplier = 1.5 if pnl < 0 else 1.0
            reward = scale * pnl * penalty_multiplier
        elif action == "exit":
            penalty_multiplier = 1.2 if pnl < 0 else 1.0
            reward = scale * pnl * penalty_multiplier
        else:
            reward = 0.0
        return float(reward)
