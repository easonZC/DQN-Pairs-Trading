# DQN Pairs Trading

This repository implements a Deep Q-Network driven pairs trading
pipeline. The original research notebooks have been modularised into a
reusable Python package with dedicated modules for data ingestion,
cointegration analysis, environment simulation, model training, and
evaluation.

## Project layout

```
dqn_pairs_trading/
├── analysis/           # Cointegration utilities
├── data/               # Data download and preprocessing helpers
├── env/                # Custom Gym environment with reward shaping
├── evaluation/         # Metrics and evaluation logic
├── models/             # Neural network architectures
├── strategies/         # Baseline mean-reversion strategies
├── training/           # DQN training loop
└── utils/              # Helper utilities (e.g., train/test split)
```

The `docs/model_improvement_guide.md` file summarises recommended
advancements for the neural network and reinforcement-learning setup.

## Quickstart

1. Install dependencies (a minimal example):

   ```bash
   pip install -r requirements.txt
   ```

   The project depends on `yfinance`, `pandas`, `numpy`, `torch`,
   `statsmodels`, `scikit-learn`, and `gym`.

2. Run the full pipeline from the command line:

   ```bash
   python cli.py Healthcare --episodes 100
   ```

   The script will download data, validate cointegrated pairs, train
   agents across four window configurations, evaluate them on the
   held-out test split, and store model weights under `outputs/`.

## Legacy notebooks

The original notebooks (`PTDQN1.ipynb`, `PTDQN2.ipynb`) remain for
reference. They can now import the package modules to avoid duplicating
logic.
