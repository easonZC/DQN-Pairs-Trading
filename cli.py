"""Command line interface for running the DQN pairs trading pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch

from dqn_pairs_trading.analysis.cointegration import find_cointegrated_pairs
from dqn_pairs_trading.data.ingest import load_sector_data
from dqn_pairs_trading.evaluation.evaluate import EvaluationResult, evaluate_agent
from dqn_pairs_trading.training.train import TrainingHyperparameters, train_agent
from dqn_pairs_trading.utils.splits import train_test_split_prices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DQN pairs trading workflow.")
    parser.add_argument("sector", choices=["Technology", "Healthcare", "Real Estate"], help="Sector to scan for pairs.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    parser.add_argument(
        "--p-value", type=float, default=0.02, help="Engle-Granger p-value threshold for candidate pairs."
    )
    parser.add_argument(
        "--t-stat", type=float, default=-3.5, help="Engle-Granger t-statistic threshold for candidate pairs."
    )
    parser.add_argument(
        "--johansen-alpha", type=float, default=0.05, help="Significance level for the Johansen test."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs"), help="Directory to store trained models."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sector_data = load_sector_data()
    validated_pairs = find_cointegrated_pairs(
        sector_data,
        args.sector,
        p_value_threshold=args.p_value,
        t_stat_threshold=args.t_stat,
        johansen_alpha=args.johansen_alpha,
    )

    if not validated_pairs:
        raise SystemExit("No validated pairs found for the selected sector.")

    pair = validated_pairs[0]
    train_prices, test_prices = train_test_split_prices(pair["prices"])

    formation_window_sizes = [30, 60, 90, 120]
    trading_window_sizes = [15, 20, 30, 60]

    args.output.mkdir(parents=True, exist_ok=True)

    for idx, (formation_window, trading_window) in enumerate(
        zip(formation_window_sizes, trading_window_sizes), start=1
    ):
        hyperparams = TrainingHyperparameters(num_episodes=args.episodes)
        agent, history = train_agent(
            train_prices,
            formation_window,
            trading_window,
            hyperparameters=hyperparams,
        )
        torch.save(agent.state_dict(), args.output / f"trained_agent_{idx}.pth")

        evaluation: EvaluationResult = evaluate_agent(
            agent,
            test_prices,
            formation_window,
            trading_window,
        )
        print(
            f"Run {idx}: profit={evaluation.total_profit:.4f}, drawdown={evaluation.max_drawdown:.4f}, "
            f"sharpe={evaluation.sharpe_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
