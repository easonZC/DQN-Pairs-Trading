"""Training helpers for the DQN pairs trading agent."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..env.pairs_trading_env import PairsTradingEnv
from ..models.dqn import DQN


@dataclass
class TrainingHyperparameters:
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.95
    learning_rate: float = 1e-3
    batch_size: int = 128
    target_update_freq: int = 100
    num_episodes: int = 200
    replay_buffer_size: int = 10_000


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, Optional[np.ndarray], bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_array = np.asarray(states, dtype=np.float32)

        return (
            torch.from_numpy(states_array),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            next_states,
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class TrainingHistory:
    rewards: List[float]
    avg_q_values: List[float]
    normal_closes: List[int]
    stop_losses: List[int]
    forced_exits: List[int]


def train_agent(
    training_prices: Dict[str, np.ndarray],
    formation_window_size: int,
    trading_window_size: int,
    *,
    hyperparameters: TrainingHyperparameters | None = None,
    device: torch.device | None = None,
) -> Tuple[DQN, TrainingHistory]:
    """Train a DQN agent on the provided price dictionary."""

    hyperparameters = hyperparameters or TrainingHyperparameters()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dim = 3
    action_dim = 7

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=hyperparameters.learning_rate)
    replay_buffer = ReplayBuffer(hyperparameters.replay_buffer_size)

    epsilon = hyperparameters.epsilon_start
    epsilon_min = hyperparameters.epsilon_end
    epsilon_decay = hyperparameters.epsilon_decay

    rewards_history: List[float] = []
    avg_q_history: List[float] = []
    normal_close_history: List[int] = []
    stop_loss_history: List[int] = []
    exit_history: List[int] = []

    step_count = 0
    for _ in range(hyperparameters.num_episodes):
        env = PairsTradingEnv(
            prices=training_prices,
            formation_window_size=formation_window_size,
            trading_window_size=trading_window_size,
        )
        state = env.reset()
        done = False
        episode_q_values: List[float] = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state_tensor)
            episode_q_values.append(q_values.mean().item())

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) >= hyperparameters.batch_size:
                (
                    states,
                    actions_tensor,
                    rewards_tensor,
                    next_states,
                    dones_tensor,
                ) = replay_buffer.sample(hyperparameters.batch_size)

                states = states.to(device)
                actions_tensor = actions_tensor.to(device)
                rewards_tensor = rewards_tensor.to(device)
                dones_tensor = dones_tensor.to(device)

                q_pred = policy_net(states).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = torch.zeros(hyperparameters.batch_size, device=device)
                    for idx, next_state_np in enumerate(next_states):
                        if next_state_np is not None:
                            next_state_tensor = (
                                torch.tensor(next_state_np, dtype=torch.float32, device=device)
                                .unsqueeze(0)
                            )
                            next_q[idx] = target_net(next_state_tensor).max(1)[0]
                    target_q = rewards_tensor + (1 - dones_tensor) * hyperparameters.gamma * next_q

                loss = F.mse_loss(q_pred, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step_count % hyperparameters.target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                step_count += 1

            state = next_state if next_state is not None else state
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        rewards_history.append(env.total_reward)
        avg_q_history.append(float(np.mean(episode_q_values)))
        normal_close_history.append(env.normal_close_count)
        stop_loss_history.append(env.stop_loss_count)
        exit_history.append(env.exit_count)

    history = TrainingHistory(
        rewards=rewards_history,
        avg_q_values=avg_q_history,
        normal_closes=normal_close_history,
        stop_losses=stop_loss_history,
        forced_exits=exit_history,
    )
    return policy_net, history
