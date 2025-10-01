# Model stability and improvement guide

This document outlines pragmatic changes that strengthen the neural
network architecture and reinforcement-learning workflow extracted from
`PTDQN1.ipynb`/`PTDQN2.ipynb`.

## Architectural upgrades

- **Dueling heads (implemented)**: The production model now splits the
  final layer into value/advantage streams (`DuelingDQN`) so the agent can
  learn state values independent of trading boundary choices. This
  reduces overestimation bias and stabilises convergence; refer to
  `dqn_pairs_trading/models/dqn.py` for the concrete architecture.
- **Layer normalisation (planned)**: Apply `nn.LayerNorm` after hidden
  layers to combat covariate shift caused by regime changes in the
  underlying spread series.
- **Feature gating (planned)**: Concatenate additional engineered features
  (spread z-score slope, realised volatility) and learn a small attention
  gate that adaptively weights them before the fully-connected trunk.

## Replay-buffer refinements

- **Prioritised replay**: Swap the uniform buffer with proportional
  prioritisation so large temporal-difference errors are revisited more
  often. This accelerates learning when trades are sparse.
- **N-step returns**: Augment the replay tuples with multi-step returns to
  propagate reward information across longer holding periods without
  sacrificing Markov structure.

## Exploration and stability

- **Noisy networks**: Replace epsilon-greedy with parameterised noise in
  the final layers (`NoisyLinear`) once the agent enters a fine-tuning
  phase; this avoids deterministic loops when spreads oscillate near the
  decision boundary.
- **Soft target updates**: Use Polyak averaging (`tau ~= 0.005`) instead of
  periodic hard copies. This smooths the target network trajectory and
  reduces training variance.

## Reward modelling

- **Transaction costs**: Deduct a configurable commission per trade to
  prevent the policy from over-trading when spreads oscillate.
- **Risk-adjusted returns**: Optimise a Sharpe-like metric by dividing PnL
  by rolling volatility, encouraging the agent to seek smoother equity
  curves rather than raw profit.

## Evaluation extensions

- **Walk-forward validation**: Refit hedge ratios and re-evaluate the
  agent across rolling windows to detect drift.
- **Scenario analysis**: Stress-test the policy under widened spreads and
  increased volatility using historical simulations from different market
  regimes (e.g. 2008 crisis, 2020 pandemic).

Implementing the above roadmap incrementally will improve both the sample
efficiency and robustness of the trading agent.
