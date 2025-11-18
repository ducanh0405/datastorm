#!/usr/bin/env python3
"""
Module 3: Dynamic Pricing Engine
=================================

Intelligent pricing system for e-grocery inventory optimization.

Pricing Strategy:
- HIGH inventory + LOW demand → Apply discount (reduce waste)
- HIGH inventory + NORMAL/HIGH demand → Small discount (accelerate)
- NORMAL/LOW inventory → Maintain price (preserve margin)

Features:
- Inventory ratio-based discounting
- Demand forecast integration
- Profit margin protection
- Batch optimization

Author: SmartGrocy Team
Date: 2025-11-15
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PricingConfig:
    """Configuration for dynamic pricing."""
    inventory_high_threshold: float = 2.0  # 200% of average = high
    inventory_critical_threshold: float = 3.0  # 300% = critical
    demand_low_threshold: float = 0.8  # 80% of forecast = low
    demand_high_threshold: float = 1.2  # 120% of forecast = high
    max_discount: float = 0.50  # Maximum 50% discount
    min_discount: float = 0.05  # Minimum 5% when discounting
    min_profit_margin: float = 0.10  # Maintain at least 10% margin


class DynamicPricingEngine:
    """
    Dynamic pricing engine for inventory optimization.

    Uses inventory levels and demand forecasts to recommend optimal prices.
    """

    def __init__(self, config: PricingConfig | None = None):
        self.config = config or PricingConfig()
        logger.info("Dynamic Pricing Engine initialized")

    def calculate_discount(
        self,
        inventory_ratio: float,
        demand_ratio: float
    ) -> tuple[float, str, str]:
        """
        Calculate optimal discount based on inventory and demand ratios.

        Args:
            inventory_ratio: current_inventory / avg_inventory
            demand_ratio: forecast_demand / historical_avg_demand

        Returns:
            (discount_pct, action, reasoning)
        """
        # Critical overstock - urgent clearance
        if inventory_ratio >= self.config.inventory_critical_threshold:
            discount = np.random.uniform(0.40, self.config.max_discount)
            return discount, "clearance", "Critical overstock - urgent clearance needed"

        # High inventory scenarios
        if inventory_ratio >= self.config.inventory_high_threshold:
            if demand_ratio <= self.config.demand_low_threshold:
                # Worst case: High inventory + Low demand
                discount = np.random.uniform(0.25, 0.40)
                return discount, "large_discount", "High inventory with low demand - aggressive markdown"
            elif demand_ratio < self.config.demand_high_threshold:
                # Medium case: High inventory + Normal demand
                discount = np.random.uniform(0.15, 0.25)
                return discount, "medium_discount", "High inventory - moderate markdown"
            else:
                # High inventory + High demand
                discount = np.random.uniform(self.config.min_discount, 0.15)
                return discount, "small_discount", "High inventory but strong demand - minimal discount"

        # Normal inventory
        if inventory_ratio >= self.config.inventory_high_threshold * 0.5:
            if demand_ratio <= self.config.demand_low_threshold:
                # Normal inventory + Low demand
                discount = np.random.uniform(self.config.min_discount, 0.10)
                return discount, "small_discount", "Weak demand - small promotional discount"

        # Low inventory or balanced scenarios - maintain price
        return 0.0, "maintain", "Balanced or low inventory - maintain current price"

    def recommend_price(
        self,
        current_price: float,
        inventory_ratio: float,
        demand_ratio: float,
        cost: float,
        min_margin: float | None = None
    ) -> dict[str, any]:
        """
        Recommend optimal price for a product.

        Args:
            current_price: Current selling price
            inventory_ratio: current_inventory / avg_inventory
            demand_ratio: forecast_demand / historical_demand
            cost: Unit cost
            min_margin: Minimum profit margin (uses config if None)

        Returns:
            Dictionary with pricing recommendation
        """
        margin = min_margin or self.config.min_profit_margin

        # Calculate discount
        discount_pct, action, reasoning = self.calculate_discount(
            inventory_ratio, demand_ratio
        )

        # Apply discount
        discount_amount = current_price * discount_pct
        new_price = current_price - discount_amount

        # Enforce minimum margin
        min_price = cost * (1 + margin)
        if new_price < min_price:
            new_price = min_price
            discount_amount = current_price - new_price
            discount_pct = discount_amount / current_price if current_price > 0 else 0
            reasoning += f" (Adjusted to maintain {margin:.0%} margin)"

        # Calculate profit
        profit_per_unit = new_price - cost
        profit_margin = profit_per_unit / new_price if new_price > 0 else 0

        return {
            'recommended_price': new_price,
            'current_price': current_price,
            'discount_pct': discount_pct,
            'discount_amount': discount_amount,
            'action': action,
            'reasoning': reasoning,
            'profit_margin': profit_margin,
            'profit_per_unit': profit_per_unit,
            'should_apply': discount_pct > 0
        }

    def batch_optimize(
        self,
        pricing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Optimize pricing for multiple products.

        Args:
            pricing_df: DataFrame with columns:
                - product_id
                - current_price
                - inventory_ratio (current/avg)
                - demand_ratio (forecast/historical)
                - cost

        Returns:
            DataFrame with pricing recommendations
        """
        results = []

        for _, row in pricing_df.iterrows():
            try:
                rec = self.recommend_price(
                    current_price=row['current_price'],
                    inventory_ratio=row['inventory_ratio'],
                    demand_ratio=row['demand_ratio'],
                    cost=row['cost']
                )
                rec['product_id'] = row['product_id']
                results.append(rec)
            except Exception as e:
                logger.warning(f"Failed to optimize {row.get('product_id', 'unknown')}: {e}")

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Test with sample data
    logging.basicConfig(level=logging.INFO)

    test_data = pd.DataFrame([
        {'product_id': 'P001', 'current_price': 10.0, 'inventory_ratio': 2.5, 'demand_ratio': 0.7, 'cost': 5.0},
        {'product_id': 'P002', 'current_price': 12.0, 'inventory_ratio': 1.7, 'demand_ratio': 0.9, 'cost': 7.0},
        {'product_id': 'P003', 'current_price': 9.0, 'inventory_ratio': 1.3, 'demand_ratio': 1.3, 'cost': 4.0},
        {'product_id': 'P004', 'current_price': 8.0, 'inventory_ratio': 0.9, 'demand_ratio': 1.0, 'cost': 3.0},
    ])

    engine = DynamicPricingEngine()
    results = engine.batch_optimize(test_data)

    print("\n" + "="*70)
    print("DYNAMIC PRICING TEST RESULTS")
    print("="*70)
    print(results[['product_id', 'current_price', 'recommended_price', 'discount_pct', 'action']].to_string(index=False))
    print("\n✅ Module 3 - Dynamic Pricing Engine Ready")
