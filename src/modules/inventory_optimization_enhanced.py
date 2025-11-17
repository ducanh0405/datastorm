#!/usr/bin/env python3
"""
Module 2 Enhanced: Advanced Inventory Optimization
==================================================
Enhanced version with comprehensive metrics for Module 4 integration.

New Features:
- Overstock risk calculation
- Inventory turnover metrics
- Fill rate prediction
- Service level tracking
- Comprehensive risk categorization

Author: SmartGrocy Team
Date: 2025-11-18
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class InventoryMetrics:
    """Enhanced metrics for inventory analysis."""
    
    # Core metrics
    reorder_point: float
    safety_stock: float
    economic_order_quantity: float
    current_inventory: float
    
    # Risk metrics
    stockout_risk_pct: float
    overstock_risk_pct: float
    service_level_actual: float
    
    # Performance metrics
    inventory_turnover: float
    days_of_stock: float
    fill_rate_expected: float
    
    # Cost metrics
    holding_cost_daily: float
    ordering_cost_per_cycle: float
    total_cost_daily: float
    
    # Actionable insights
    should_reorder: bool
    reorder_urgency: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_category: str  # BALANCED, STOCKOUT_RISK, OVERSTOCK_RISK, CRITICAL
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Module 4."""
        return {
            'current_inventory': self.current_inventory,
            'safety_stock': self.safety_stock,
            'reorder_point': self.reorder_point,
            'eoq': self.economic_order_quantity,
            'should_reorder': self.should_reorder,
            'reorder_urgency': self.reorder_urgency,
            'stockout_risk_pct': self.stockout_risk_pct,
            'overstock_risk_pct': self.overstock_risk_pct,
            'risk_category': self.risk_category,
            'service_level': self.service_level_actual,
            'fill_rate': self.fill_rate_expected,
            'inventory_turnover': self.inventory_turnover,
            'days_of_stock': self.days_of_stock,
            'holding_cost_daily': self.holding_cost_daily,
            'total_cost_daily': self.total_cost_daily,
            'reorder_reasoning': self._generate_reasoning()
        }
    
    def _generate_reasoning(self) -> str:
        """Generate human-readable reasoning."""
        if self.stockout_risk_pct > 70:
            return f"URGENT: {self.stockout_risk_pct:.0f}% stockout risk - immediate reorder required"
        elif self.should_reorder:
            return f"Inventory below ROP ({self.current_inventory:.0f} < {self.reorder_point:.0f}) - reorder recommended"
        elif self.overstock_risk_pct > 60:
            return f"WARNING: {self.overstock_risk_pct:.0f}% overstock risk - consider markdowns"
        else:
            return "Inventory levels optimal - continue monitoring"


class EnhancedInventoryOptimizer:
    """Enhanced inventory optimizer with comprehensive metrics."""
    
    def __init__(self, service_level: float = 0.95, **kwargs):
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)
        self.config = kwargs
        
    def optimize_with_metrics(
        self,
        avg_daily_demand: float,
        demand_std: float,
        current_inventory: float,
        unit_cost: float,
        lead_time_days: int = 7,
        ordering_cost: float = 50.0,
        holding_cost_rate: float = 0.20
    ) -> InventoryMetrics:
        """Calculate comprehensive inventory metrics."""
        
        # Safety stock
        safety_stock = self.z_score * demand_std * np.sqrt(lead_time_days)
        safety_stock = max(0, safety_stock)
        
        # Reorder point
        lead_time_demand = avg_daily_demand * lead_time_days
        reorder_point = lead_time_demand + safety_stock
        
        # EOQ
        annual_demand = avg_daily_demand * 365
        holding_cost_per_unit = unit_cost * holding_cost_rate
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        
        # Stockout risk
        if demand_std > 0:
            z_current = (current_inventory - lead_time_demand) / (demand_std * np.sqrt(lead_time_days))
            stockout_risk_pct = (1 - stats.norm.cdf(z_current)) * 100
        else:
            stockout_risk_pct = 0 if current_inventory >= lead_time_demand else 100
        
        # Overstock risk
        inventory_ratio = current_inventory / avg_daily_demand if avg_daily_demand > 0 else 0
        overstock_risk_pct = min(100, max(0, (inventory_ratio - 10) * 10))  # >10 days = risk
        
        # Service level actual
        service_level_actual = 1 - (stockout_risk_pct / 100)
        
        # Inventory turnover
        avg_inventory = eoq / 2 + safety_stock
        inventory_turnover = annual_demand / avg_inventory if avg_inventory > 0 else 0
        
        # Days of stock
        days_of_stock = current_inventory / avg_daily_demand if avg_daily_demand > 0 else 999
        
        # Fill rate
        fill_rate_expected = min(0.99, service_level_actual * 1.02)
        
        # Costs
        holding_cost_daily = current_inventory * holding_cost_per_unit / 365
        ordering_cost_per_cycle = ordering_cost / (365 / (annual_demand / eoq))
        total_cost_daily = holding_cost_daily + ordering_cost_per_cycle
        
        # Reorder decision
        should_reorder = current_inventory <= reorder_point
        
        # Urgency
        if stockout_risk_pct > 70:
            urgency = "CRITICAL"
        elif stockout_risk_pct > 50:
            urgency = "HIGH"
        elif should_reorder:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"
        
        # Risk category
        if stockout_risk_pct > 50 and overstock_risk_pct > 50:
            risk_category = "CRITICAL"
        elif stockout_risk_pct > 50:
            risk_category = "STOCKOUT_RISK"
        elif overstock_risk_pct > 60:
            risk_category = "OVERSTOCK_RISK"
        else:
            risk_category = "BALANCED"
        
        return InventoryMetrics(
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            economic_order_quantity=eoq,
            current_inventory=current_inventory,
            stockout_risk_pct=stockout_risk_pct,
            overstock_risk_pct=overstock_risk_pct,
            service_level_actual=service_level_actual,
            inventory_turnover=inventory_turnover,
            days_of_stock=days_of_stock,
            fill_rate_expected=fill_rate_expected,
            holding_cost_daily=holding_cost_daily,
            ordering_cost_per_cycle=ordering_cost_per_cycle,
            total_cost_daily=total_cost_daily,
            should_reorder=should_reorder,
            reorder_urgency=urgency,
            risk_category=risk_category
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    optimizer = EnhancedInventoryOptimizer(service_level=0.95)
    
    metrics = optimizer.optimize_with_metrics(
        avg_daily_demand=100,
        demand_std=15,
        current_inventory=120,
        unit_cost=30000,
        lead_time_days=7
    )
    
    print("\nEnhanced Inventory Metrics:")
    for k, v in metrics.to_dict().items():
        print(f"  {k}: {v}")
