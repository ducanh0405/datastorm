#!/usr/bin/env python3
"""
Inventory Backtesting Framework
================================

Validate inventory management strategies using historical data simulation.

Capabilities:
- Simulate inventory decisions over time
- Calculate Spoilage % (overstocking waste)
- Calculate Stockout % (lost sales)
- Compare Baseline vs ML strategies
- Generate business KPIs
- Validate forecast accuracy impact on inventory

Usage:
    from src.modules.inventory_backtesting import InventoryBacktester

    backtester = InventoryBacktester(historical_data, forecasts)
    results = backtester.run_simulation()
    metrics = backtester.calculate_kpis(results)

Author: SmartGrocy Team
Date: 2025-11-15
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for inventory backtesting."""
    initial_inventory: float = 100.0  # Starting inventory level
    lead_time_days: int = 7  # Order lead time
    shelf_life_days: int = 14  # Product shelf life (for perishables)
    unit_cost: float = 10.0  # Cost per unit
    unit_price: float = 15.0  # Selling price per unit
    holding_cost_per_day: float = 0.05  # Daily holding cost per unit
    spoilage_penalty: float = 10.0  # Cost of spoiling one unit
    stockout_penalty: float = 5.0  # Lost profit per stockout unit


class InventoryBacktester:
    """
    Backtest inventory management strategies.

    Simulates day-by-day inventory decisions and calculates KPIs.
    """

    def __init__(
        self,
        historical_demand: pd.DataFrame,
        forecasts: pd.DataFrame | None = None,
        config: BacktestConfig | None = None
    ):
        """
        Initialize backtester.

        Args:
            historical_demand: DataFrame with columns:
                - date, product_id, actual_demand
            forecasts: DataFrame with forecast_q50, forecast_q95 (optional)
            config: BacktestConfig
        """
        self.historical_demand = historical_demand.sort_values('date')
        self.forecasts = forecasts
        self.config = config or BacktestConfig()
        self.simulation_results = []
        logger.info("Inventory Backtester initialized")

    def baseline_strategy(self, row: pd.Series) -> float:
        """
        Baseline strategy: Simple 7-day moving average.

        Args:
            row: Current data row

        Returns:
            Order quantity
        """
        # Use 7-day moving average as baseline forecast
        return row.get('ma_7', row.get('actual_demand', 50))

    def ml_strategy(self, row: pd.Series) -> tuple[float, float]:
        """
        ML strategy: Use forecast Q50 for order, Q95 for safety stock.

        Args:
            row: Current data row with forecasts

        Returns:
            (order_quantity, safety_stock)
        """
        q50 = row.get('forecast_q50', row.get('actual_demand', 50))
        q95 = row.get('forecast_q95', q50 * 1.5)
        safety_stock = q95 - q50
        return q50, safety_stock

    def simulate_day(
        self,
        current_inventory: float,
        actual_demand: float,
        order_quantity: float,
        safety_stock: float,
        age_distribution: list[float]
    ) -> dict[str, float]:
        """
        Simulate one day of inventory operations.

        Args:
            current_inventory: Inventory at start of day
            actual_demand: Actual customer demand
            order_quantity: Units ordered
            safety_stock: Target safety stock
            age_distribution: Age of each unit in inventory

        Returns:
            Dictionary with day's metrics
        """
        # Spoilage (units older than shelf life)
        spoiled_units = sum(1 for age in age_distribution if age > self.config.shelf_life_days)
        current_inventory -= spoiled_units

        # Sales (limited by available inventory)
        units_sold = min(actual_demand, current_inventory)
        stockout_units = max(0, actual_demand - current_inventory)

        # Update inventory
        current_inventory -= units_sold

        # Receive order (if any, after lead time)
        # Simplified: assume instant for this simulation
        current_inventory += order_quantity

        # Costs
        spoilage_cost = spoiled_units * self.config.spoilage_penalty
        stockout_cost = stockout_units * self.config.stockout_penalty
        holding_cost = current_inventory * self.config.holding_cost_per_day
        revenue = units_sold * self.config.unit_price
        cogs = units_sold * self.config.unit_cost

        return {
            'inventory': current_inventory,
            'demand': actual_demand,
            'sold': units_sold,
            'spoiled': spoiled_units,
            'stockout': stockout_units,
            'spoilage_cost': spoilage_cost,
            'stockout_cost': stockout_cost,
            'holding_cost': holding_cost,
            'revenue': revenue,
            'cogs': cogs,
            'profit': revenue - cogs - spoilage_cost - stockout_cost - holding_cost
        }

    def run_simulation(
        self,
        strategy: str = 'ml',
        product_id: str | None = None
    ) -> pd.DataFrame:
        """
        Run full backtest simulation.

        Args:
            strategy: 'baseline' or 'ml'
            product_id: Specific product (None = all)

        Returns:
            DataFrame with daily simulation results
        """
        # Filter data
        data = self.historical_demand.copy()
        if product_id:
            data = data[data['product_id'] == product_id]

        # Add forecasts if available
        if self.forecasts is not None and strategy == 'ml':
            data = data.merge(self.forecasts, on=['date', 'product_id'], how='left')

        # Calculate moving average for baseline
        data['ma_7'] = data.groupby('product_id')['actual_demand'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

        results = []
        inventory = self.config.initial_inventory
        age_dist = [0] * int(inventory)  # Simplified age tracking

        for _idx, row in data.iterrows():
            # Decide order quantity
            if strategy == 'baseline':
                order_qty = self.baseline_strategy(row)
                safety_stock = order_qty * 0.5  # Simple 50% buffer
            else:  # ml
                order_qty, safety_stock = self.ml_strategy(row)

            # Simulate day
            day_result = self.simulate_day(
                inventory,
                row['actual_demand'],
                order_qty,
                safety_stock,
                age_dist
            )

            # Update state
            inventory = day_result['inventory']
            age_dist = [age + 1 for age in age_dist if age <= self.config.shelf_life_days]
            age_dist.extend([0] * int(order_qty))  # New inventory

            # Record
            day_result.update({
                'date': row['date'],
                'product_id': row['product_id'],
                'strategy': strategy,
                'order_quantity': order_qty,
                'safety_stock': safety_stock
            })
            results.append(day_result)

        return pd.DataFrame(results)

    def calculate_kpis(self, simulation_results: pd.DataFrame) -> dict[str, float]:
        """
        Calculate business KPIs from simulation.

        Args:
            simulation_results: Output from run_simulation()

        Returns:
            Dictionary with KPIs
        """
        total_demand = simulation_results['demand'].sum()
        total_sold = simulation_results['sold'].sum()
        total_spoiled = simulation_results['spoiled'].sum()
        total_stockout = simulation_results['stockout'].sum()

        # Calculate percentages
        spoilage_rate = (total_spoiled / (total_sold + total_spoiled)) * 100 if (total_sold + total_spoiled) > 0 else 0
        stockout_rate = (total_stockout / total_demand) * 100 if total_demand > 0 else 0
        fill_rate = (total_sold / total_demand) * 100 if total_demand > 0 else 0

        # Financial metrics
        total_revenue = simulation_results['revenue'].sum()
        (simulation_results['cogs'].sum() +
                      simulation_results['spoilage_cost'].sum() +
                      simulation_results['stockout_cost'].sum() +
                      simulation_results['holding_cost'].sum())
        total_profit = simulation_results['profit'].sum()
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0

        return {
            'strategy': simulation_results['strategy'].iloc[0],
            'spoilage_rate_pct': spoilage_rate,
            'stockout_rate_pct': stockout_rate,
            'fill_rate_pct': fill_rate,
            'total_revenue': total_revenue,
            'total_profit': total_profit,
            'profit_margin_pct': profit_margin,
            'total_spoiled_units': total_spoiled,
            'total_stockout_units': total_stockout,
            'avg_inventory': simulation_results['inventory'].mean()
        }

    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare Baseline vs ML strategies.

        Returns:
            Comparison DataFrame
        """
        # Run both simulations
        logger.info("Running baseline simulation...")
        baseline_sim = self.run_simulation('baseline')
        baseline_kpis = self.calculate_kpis(baseline_sim)

        logger.info("Running ML simulation...")
        ml_sim = self.run_simulation('ml')
        ml_kpis = self.calculate_kpis(ml_sim)

        # Create comparison
        pd.DataFrame([baseline_kpis, ml_kpis])

        # Calculate improvements
        improvements = {
            'metric': [],
            'baseline': [],
            'ml_model': [],
            'improvement': [],
            'improvement_pct': []
        }

        key_metrics = ['spoilage_rate_pct', 'stockout_rate_pct', 'profit_margin_pct']
        for metric in key_metrics:
            baseline_val = baseline_kpis[metric]
            ml_val = ml_kpis[metric]

            # For spoilage and stockout, lower is better
            if 'spoilage' in metric or 'stockout' in metric:
                improvement = baseline_val - ml_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            else:  # For profit, higher is better
                improvement = ml_val - baseline_val
                improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0

            improvements['metric'].append(metric)
            improvements['baseline'].append(baseline_val)
            improvements['ml_model'].append(ml_val)
            improvements['improvement'].append(improvement)
            improvements['improvement_pct'].append(improvement_pct)

        return pd.DataFrame(improvements)


if __name__ == "__main__":
    # Test with synthetic data
    logging.basicConfig(level=logging.INFO)

    # Create sample historical data
    dates = pd.date_range('2025-10-01', '2025-11-14', freq='D')
    np.random.seed(42)

    historical = pd.DataFrame({
        'date': dates,
        'product_id': ['P001'] * len(dates),
        'actual_demand': np.random.poisson(100, len(dates))
    })

    # Create sample forecasts
    forecasts = pd.DataFrame({
        'date': dates,
        'product_id': ['P001'] * len(dates),
        'forecast_q50': historical['actual_demand'] * 0.95,  # 95% accuracy
        'forecast_q95': historical['actual_demand'] * 1.3
    })

    # Run backtest
    backtester = InventoryBacktester(historical, forecasts)
    comparison = backtester.compare_strategies()

    print("\n" + "="*70)
    print("INVENTORY BACKTESTING RESULTS")
    print("="*70)
    print(comparison.to_string(index=False))
    print("\n✅ Inventory Backtesting Framework Ready")
