"""Neural network models for the DQN agent."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DQN(nn.Module):
    """Dueling feed-forward network used to approximate the Q-function."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        hidden_sizes: Sequence[int] = (128, 128, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        feature_layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            feature_layers.append(nn.Linear(input_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            if dropout > 0:
                feature_layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.feature_extractor = (
            nn.Sequential(*feature_layers)
            if feature_layers
            else nn.Identity()
        )

        feature_dim = input_dim
        value_hidden_dim = max(feature_dim // 2, 1)
        advantage_hidden_dim = max(feature_dim, 1)

        value_layers: list[nn.Module] = [nn.Linear(feature_dim, value_hidden_dim), nn.ReLU()]
        if dropout > 0:
            value_layers.append(nn.Dropout(p=dropout))
        value_layers.append(nn.Linear(value_hidden_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

        advantage_layers: list[nn.Module] = [
            nn.Linear(feature_dim, advantage_hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            advantage_layers.append(nn.Dropout(p=dropout))
        advantage_layers.append(nn.Linear(advantage_hidden_dim, action_dim))
        self.advantage_head = nn.Sequential(*advantage_layers)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(state)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        return value + advantage - advantage_mean
