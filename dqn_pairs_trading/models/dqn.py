"""Neural network models for the DQN agent."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Feed-forward network used to approximate the Q-function."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_sizes: Sequence[int] = (128, 128, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(state)
