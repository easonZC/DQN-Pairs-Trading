import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from dqn_pairs_trading.data.ingest import load_sector_data
from dqn_pairs_trading.analysis.cointegration import find_cointegrated_pairs
from dqn_pairs_trading.utils.splits import train_test_split_prices
from dqn_pairs_trading.models.dqn import DQN
from dqn_pairs_trading.evaluation.evaluate import evaluate_agent

pair_sector = 'Technology'
formation_window_sizes = [30, 60, 90, 120]
trading_window_sizes = [15, 20, 30, 60]

sector_data = load_sector_data()
validated_pairs = find_cointegrated_pairs(
    sector_data,
    pair_sector,
    p_value_threshold=0.02,
    t_stat_threshold=-3.5,
    johansen_alpha=0.05,
)
if not validated_pairs:
    raise SystemExit('No validated pairs found')

selected = validated_pairs[0]
pair = selected['pair']
_, test_prices = train_test_split_prices(selected['prices'])

state_dim = 3
action_dim = 7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results = []
for idx, (formation_window, trading_window) in enumerate(zip(formation_window_sizes, trading_window_sizes), start=1):
    path = Path('outputs') / f'trained_agent_{idx}.pth'
    if not path.exists():
        continue
    agent = DQN(state_dim, action_dim).to(device)
    agent.load_state_dict(torch.load(path, map_location=device))
    res = evaluate_agent(agent, test_prices, formation_window, trading_window, device=device)
    results.append({
        'run': idx,
        'formation_window': formation_window,
        'trading_window': trading_window,
        'total_profit': res.total_profit,
        'max_drawdown': res.max_drawdown,
        'sharpe': res.sharpe_ratio,
        'equity_curve': res.equity_curve,
        'trade_count': len(res.trades),
    })

if not results:
    raise SystemExit('No evaluation results available')

plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots(figsize=(10, 6))
for result in results:
    curve = result['equity_curve']
    x = np.arange(len(curve))
    label = f"Run {result['run']} (F{result['formation_window']}/T{result['trading_window']})"
    ax.plot(x, curve, label=label)

ax.set_title(f'累积收益曲线 - Pair {pair[0]} / {pair[1]}')
ax.set_xlabel('交易序号')
ax.set_ylabel('累积收益 (累加收益率)')
ax.legend()
ax.grid(True)

output_dir = Path('outputs')
output_dir.mkdir(exist_ok=True)
fig_path = output_dir / 'equity_curves.png'
fig.tight_layout()
fig.savefig(fig_path, dpi=150)

summary_path = output_dir / 'evaluation_summary.json'
serialized = []
for result in results:
    serialized.append({
        key: (value.tolist() if key == 'equity_curve' else value)
        for key, value in result.items()
    })
for entry in serialized:
    entry['pair'] = pair

summary_path.write_text(json.dumps(serialized, indent=2), encoding='utf-8')

print(f'figure saved: {fig_path}')
print(f'summary saved: {summary_path}')
