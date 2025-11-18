#!/usr/bin/env python3
"""
Module 3 Enhanced: Advanced Dynamic Pricing
==========================================
Enhanced with revenue impact and comprehensive metrics.

New Features:
- Revenue impact calculation
- Elasticity consideration
- Competitive positioning
- Profit optimization
- Risk-adjusted pricing

Author: SmartGrocy Team
Date: 2025-11-18
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PricingMetrics:
    """Enhanced pricing metrics."""

    # Price metrics
    current_price: float
    recommended_price: float
    price_change: float
    price_change_pct: float

    # Discount metrics
    discount_pct: float
    discount_amount: float

    # Margin metrics
    unit_cost: float
    current_margin: float
    new_margin: float
    margin_change: float

    # Action & reasoning
    action: str
    reasoning: str
    priority: str  # LOW, MEDIUM, HIGH, CRITICAL

    # Impact metrics (NEW)
    expected_demand_impact_pct: float
    expected_revenue_change: float
    expected_profit_change: float
    competitive_position: str

    def to_dict(self) -> dict:
        """Convert to dict for Module 4."""
        return {
            'current_price': self.current_price,
            'recommended_price': self.recommended_price,
            'price_change': self.price_change,
            'price_change_pct': self.price_change_pct,
            'discount_pct': self.discount_pct,
            'unit_cost': self.unit_cost,
            'current_margin': self.current_margin,
            'new_margin': self.new_margin,
            'action': self.action,
            'reasoning': self.reasoning,
            'priority': self.priority,
            'expected_revenue_impact': self.expected_revenue_change,
            'expected_profit_impact': self.expected_profit_change,
            'competitive_position': self.competitive_position
        }


class EnhancedPricingEngine:
    """Enhanced pricing engine with impact analysis."""

    # Price elasticity estimates
    ELASTICITY_MAP = {
        'fresh_produce': -1.5,
        'dairy': -1.2,
        'packaged': -0.8,
        'default': -1.0
    }

    def __init__(self, **config):
        self.config = config

    def optimize_with_impact(
        self,
        current_price: float,
        unit_cost: float,
        current_demand: float,
        inventory_ratio: float,
        demand_ratio: float,
        category: str = 'default'
    ) -> PricingMetrics:
        """Calculate pricing with comprehensive impact analysis."""

        # Calculate discount using existing logic
        discount_pct, action = self._calculate_discount_logic(
            inventory_ratio, demand_ratio
        )

        # New price
        recommended_price = current_price * (1 - discount_pct)

        # Enforce minimum margin (10%)
        min_price = unit_cost * 1.1
        if recommended_price < min_price:
            recommended_price = min_price
            discount_pct = (current_price - recommended_price) / current_price

        # Margins
        current_margin = (current_price - unit_cost) / current_price
        new_margin = (recommended_price - unit_cost) / recommended_price

        # Price elasticity impact
        elasticity = self.ELASTICITY_MAP.get(category, -1.0)
        price_change_pct = ((recommended_price - current_price) / current_price) * 100

        # Demand impact: %ΔQ = elasticity × %ΔP
        expected_demand_impact_pct = elasticity * (price_change_pct / 100) * 100
        new_demand = current_demand * (1 + expected_demand_impact_pct / 100)

        # Revenue impact
        current_revenue = current_price * current_demand
        new_revenue = recommended_price * new_demand
        revenue_change = new_revenue - current_revenue

        # Profit impact
        current_profit = (current_price - unit_cost) * current_demand
        new_profit = (recommended_price - unit_cost) * new_demand
        profit_change = new_profit - current_profit

        # Priority
        if inventory_ratio > 3.0:
            priority = "CRITICAL"
        elif inventory_ratio > 2.0 and demand_ratio < 0.8:
            priority = "HIGH"
        elif discount_pct > 0.15:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        # Competitive position
        if discount_pct > 0.3:
            competitive = "AGGRESSIVE"
        elif discount_pct > 0.1:
            competitive = "COMPETITIVE"
        else:
            competitive = "PREMIUM"

        # Reasoning
        reasoning = self._generate_pricing_reasoning(
            inventory_ratio, demand_ratio, discount_pct, profit_change
        )

        return PricingMetrics(
            current_price=current_price,
            recommended_price=recommended_price,
            price_change=recommended_price - current_price,
            price_change_pct=price_change_pct,
            discount_pct=discount_pct,
            discount_amount=current_price - recommended_price,
            unit_cost=unit_cost,
            current_margin=current_margin,
            new_margin=new_margin,
            margin_change=new_margin - current_margin,
            action=action,
            reasoning=reasoning,
            priority=priority,
            expected_demand_impact_pct=expected_demand_impact_pct,
            expected_revenue_change=revenue_change,
            expected_profit_change=profit_change,
            competitive_position=competitive
        )

    def _calculate_discount_logic(self, inv_ratio: float, dem_ratio: float) -> tuple:
        """Calculate discount using business rules."""
        if inv_ratio >= 3.0:
            return 0.45, "clearance"
        elif inv_ratio >= 2.0:
            if dem_ratio <= 0.8:
                return 0.30, "large_discount"
            elif dem_ratio < 1.2:
                return 0.18, "medium_discount"
            else:
                return 0.08, "small_discount"
        elif inv_ratio >= 1.5 and dem_ratio <= 0.8:
            return 0.08, "small_discount"
        else:
            return 0.0, "maintain"

    def _generate_pricing_reasoning(self, inv_ratio, dem_ratio, discount, profit_change) -> str:
        """Generate comprehensive reasoning."""
        parts = []

        if inv_ratio > 2.5:
            parts.append(f"Critical overstock ({inv_ratio:.1f}x normal)")
        elif inv_ratio > 1.8:
            parts.append(f"High inventory ({inv_ratio:.1f}x normal)")

        if dem_ratio < 0.8:
            parts.append("weak demand")
        elif dem_ratio > 1.2:
            parts.append("strong demand")

        if profit_change < 0:
            parts.append(f"accepting ${abs(profit_change):.0f} profit reduction to clear inventory")
        elif profit_change > 0:
            parts.append(f"expecting ${profit_change:.0f} profit increase")

        return " + ".join(parts) if parts else "Balanced market conditions"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = EnhancedPricingEngine()

    metrics = engine.optimize_with_impact(
        current_price=50000,
        unit_cost=30000,
        current_demand=100,
        inventory_ratio=2.3,
        demand_ratio=0.75,
        category='fresh_produce'
    )

    print("\nEnhanced Pricing Metrics:")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v}")
