# Tạo file: scripts/run_full_backtesting.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.inventory_backtesting import InventoryBacktester
from src import config

# Load FreshRetail50k data
print("Loading historical data...")
# Adjust path nếu cần
historical = pd.read_csv('data/2_raw/transactions.csv')  

# Load forecasts (từ previous run)
forecasts = pd.read_csv('reports/predictions_test_set.csv')

# Run backtesting
print("Running backtesting simulation...")
backtester = InventoryBacktester(historical, forecasts)
comparison = backtester.compare_strategies()

# Save results
output_dir = config.DATA_DIRS['reports'] / 'backtesting'
output_dir.mkdir(exist_ok=True, parents=True)
comparison.to_csv(output_dir / 'strategy_comparison.csv', index=False)

print("\n" + "="*70)
print("BACKTESTING RESULTS - BASELINE VS ML")
print("="*70)
print(comparison.to_string(index=False))
print("\n✅ Results saved to: reports/backtesting/strategy_comparison.csv")
