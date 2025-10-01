# DQN Pairs Trading

This repository implements a Deep Q-Network driven pairs trading
pipeline. The original research notebooks have been modularised into a
reusable Python package with dedicated modules for data ingestion,
cointegration analysis, environment simulation, model training, and
evaluation.

## Current architecture

- **Dueling DQN policy** (`dqn_pairs_trading/models/dqn.py`): the network
  now separates state value and action advantage heads to reduce Q-value
  estimation variance.
- **CUDA-ready training**: `train_agent` automatically selects GPU when a
  CUDA device is available; otherwise it falls back to CPU.
- **Modular evaluation**: the CLI and helper scripts reuse
  `evaluate_agent` to produce cumulative PnL, drawdown, and Sharpe metrics.

## Project layout

```
dqn_pairs_trading/
├── analysis/           # Cointegration utilities
├── data/               # Data download and preprocessing helpers
├── env/                # Custom Gym environment with reward shaping
├── evaluation/         # Metrics and evaluation logic
├── models/             # Neural network architectures
├── strategies/         # Baseline mean-reversion strategies
├── training/           # Replay buffer and DQN training loop
└── utils/              # Helper utilities (e.g., train/test split)
```

`docs/model_improvement_guide.md` tracks future enhancements such as
LayerNorm, prioritised replay, and soft target updates.

## Quickstart

1. **Install dependencies** (minimal example):

   ```bash
   pip install -r requirements.txt
   ```

   Required packages include `yfinance`, `pandas`, `numpy`, `torch`,
   `statsmodels`, `scikit-learn`, and `gym`.

2. **Run the end-to-end pipeline**:

   ```bash
   python cli.py Technology --episodes 200
   ```

   The CLI downloads sector data, validates cointegrated pairs, trains
   four agents with different formation/trading windows, evaluates them
   on the held-out test split, and stores weights under `outputs/`. By
   default it uses the first validated pair (often `AMD/ASML`); adjust
   `cli.py` if you want to iterate all pairs.

   > Note: yfinance may warn about delisted tickers. The pipeline skips
   > missing series, but you can edit `SectorUniverse` to remove them.

3. **Visualise evaluation results**:

   After training, generate cumulative return curves with:

   ```bash
   python visualize_results.py
   ```

   This script reloads the saved agents, evaluates them on the same test
   prices, and writes `outputs/equity_curves.png` plus a numerical
   summary in `outputs/evaluation_summary.json`.

## Tuning tips

- Increase the CLI timeout (or run the command directly in a terminal)
  for long training sessions.
- `train.py` now batches replay samples with `np.asarray` before
tensor conversion to avoid PyTorch warnings during large runs.
- For more robust results, consider expanding feature engineering,
  testing multiple pairs, or enabling the planned upgrades listed in the
  improvement guide.

## Legacy notebooks

The original notebooks (`PTDQN1.ipynb`, `PTDQN2.ipynb`) remain for
reference and can import the package modules to avoid duplicated logic.
