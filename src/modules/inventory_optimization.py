#!/usr/bin/env python3
"""
Module 2: Inventory Optimization & Reorder Point Calculator
===========================================================

Advanced inventory management system that uses demand forecasts from Module 1
to calculate optimal reorder points, safety stock, and order quantities.

Features:
- Reorder Point (ROP) calculation
- Economic Order Quantity (EOQ)
- Safety Stock optimization
- Service Level maintenance
- Stockout prevention
- Fill Rate tracking

Integration:
- Uses quantile forecasts (Q50, Q95) from Module 1
- Optimized for FreshRetail50k dataset
- Supports continuous and periodic review policies

Author: SmartGrocy Team
Date: 2025-11-15
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class InventoryConfig:
    """
    Configuration for inventory optimization.
    
    Attributes:
        service_level: Target service level (e.g., 0.95 for 95%)
        ordering_cost: Fixed cost per order ($)
        holding_cost_rate: Annual holding cost as % of item value
        lead_time_days: Average lead time in days
        lead_time_std: Standard deviation of lead time
        review_period_days: Days between inventory reviews (0 for continuous)
        unit_cost: Cost per unit ($)
    """
    service_level: float = 0.95  # 95% service level
    ordering_cost: float = 50.0  # $50 per order
    holding_cost_rate: float = 0.20  # 20% annual holding cost
    lead_time_days: int = 7  # 1 week lead time
    lead_time_std: float = 1.0  # 1 day std dev
    review_period_days: int = 0  # Continuous review
    unit_cost: float = 10.0  # $10 per unit
    

class InventoryOptimizer:
    """
    Comprehensive inventory optimization engine.
    
    Uses demand forecasts to calculate:
    - Optimal reorder points
    - Economic order quantities
    - Safety stock levels
    - Fill rates and service levels
    """
    
    def __init__(self, config: Optional[InventoryConfig] = None):
        """
        Initialize optimizer with configuration.
        
        Args:
            config: InventoryConfig object (uses defaults if None)
        """
        self.config = config or InventoryConfig()
        self.z_score = stats.norm.ppf(self.config.service_level)
        logger.info(f"Inventory Optimizer initialized with service level: {self.config.service_level:.1%}")
    
    def calculate_safety_stock(
        self,
        demand_std: float,
        lead_time_days: Optional[int] = None
    ) -> float:
        """
        Calculate safety stock using demand uncertainty.
        
        Formula: SS = Z * σ_demand * √(LT + RP)
        
        Args:
            demand_std: Standard deviation of daily demand
            lead_time_days: Lead time in days (uses config if None)
        
        Returns:
            Safety stock quantity
        """
        lt = lead_time_days or self.config.lead_time_days
        rp = self.config.review_period_days
        
        # Account for lead time and review period uncertainty
        time_factor = np.sqrt(lt + rp)
        safety_stock = self.z_score * demand_std * time_factor
        
        return max(0, safety_stock)  # Cannot be negative
    
    def calculate_reorder_point(
        self,
        avg_daily_demand: float,
        demand_std: float,
        lead_time_days: Optional[int] = None,
        include_safety_stock: bool = True
    ) -> Dict[str, float]:
        """
        Calculate optimal reorder point.
        
        Formula: ROP = (Average Daily Demand × Lead Time) + Safety Stock
        
        Args:
            avg_daily_demand: Average daily demand (units/day)
            demand_std: Standard deviation of daily demand
            lead_time_days: Lead time in days
            include_safety_stock: Whether to include safety stock
        
        Returns:
            Dictionary with ROP components
        """
        lt = lead_time_days or self.config.lead_time_days
        
        # Lead time demand
        lead_time_demand = avg_daily_demand * lt
        
        # Safety stock
        if include_safety_stock:
            safety_stock = self.calculate_safety_stock(demand_std, lt)
        else:
            safety_stock = 0
        
        # Reorder point
        reorder_point = lead_time_demand + safety_stock
        
        return {
            'reorder_point': reorder_point,
            'lead_time_demand': lead_time_demand,
            'safety_stock': safety_stock,
            'service_level': self.config.service_level,
            'z_score': self.z_score
        }
    
    def calculate_economic_order_quantity(
        self,
        annual_demand: float,
        ordering_cost: Optional[float] = None,
        holding_cost_per_unit: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Economic Order Quantity (EOQ).
        
        Formula: EOQ = √(2DS/H)
        where:
            D = Annual demand
            S = Ordering cost per order
            H = Holding cost per unit per year
        
        Args:
            annual_demand: Annual demand in units
            ordering_cost: Cost per order ($)
            holding_cost_per_unit: Holding cost per unit per year ($)
        
        Returns:
            Dictionary with EOQ and related metrics
        """
        S = ordering_cost or self.config.ordering_cost
        H = holding_cost_per_unit or (self.config.unit_cost * self.config.holding_cost_rate)
        
        # EOQ calculation
        eoq = np.sqrt((2 * annual_demand * S) / H)
        
        # Number of orders per year
        num_orders = annual_demand / eoq
        
        # Total annual cost
        annual_ordering_cost = num_orders * S
        annual_holding_cost = (eoq / 2) * H
        total_cost = annual_ordering_cost + annual_holding_cost
        
        return {
            'eoq': eoq,
            'num_orders_per_year': num_orders,
            'annual_ordering_cost': annual_ordering_cost,
            'annual_holding_cost': annual_holding_cost,
            'total_annual_cost': total_cost,
            'order_frequency_days': 365 / num_orders
        }
    
    def optimize_inventory_from_forecast(
        self,
        forecast_df: pd.DataFrame,
        product_id: str,
        store_id: Optional[str] = None,
        forecast_horizon_days: int = 30
    ) -> Dict[str, any]:
        """
        Complete inventory optimization using forecast data from Module 1.
        
        Args:
            forecast_df: DataFrame with columns:
                - product_id, store_id (optional)
                - forecast_q50: Median forecast
                - forecast_q05, forecast_q95: Prediction intervals
                - timestamp: Forecast date
            product_id: Product identifier
            store_id: Store identifier (optional)
            forecast_horizon_days: Days to use for calculations
        
        Returns:
            Complete inventory optimization results
        """
        # Filter forecast for specific product/store
        mask = forecast_df['product_id'] == product_id
        if store_id:
            mask &= forecast_df['store_id'] == store_id
        
        forecast_subset = forecast_df[mask].head(forecast_horizon_days)
        
        if len(forecast_subset) == 0:
            raise ValueError(f"No forecast data for product {product_id}")
        
        # Calculate demand statistics from forecasts
        q50_forecast = forecast_subset['forecast_q50'].values
        q05_forecast = forecast_subset['forecast_q05'].values
        q95_forecast = forecast_subset['forecast_q95'].values
        
        # Average daily demand (median forecast)
        avg_daily_demand = np.mean(q50_forecast)
        
        # Demand std (estimated from quantile forecasts)
        # Using Q95-Q05 as proxy for ~3 sigma range
        demand_std = np.mean(q95_forecast - q05_forecast) / (2 * 1.645)  # 90% interval
        
        # Annual demand projection
        annual_demand = avg_daily_demand * 365
        
        # Calculate ROP
        rop_results = self.calculate_reorder_point(
            avg_daily_demand=avg_daily_demand,
            demand_std=demand_std
        )
        
        # Calculate EOQ
        eoq_results = self.calculate_economic_order_quantity(
            annual_demand=annual_demand
        )
        
        # Current inventory position (would come from real data)
        # For now, simulate
        current_inventory = rop_results['reorder_point'] * 1.5  # 150% of ROP
        
        # Calculate days until reorder
        days_until_reorder = (current_inventory - rop_results['reorder_point']) / avg_daily_demand
        
        # Stockout risk (probability demand exceeds current + pipeline)
        pipeline_inventory = 0  # Assuming no orders in transit
        total_available = current_inventory + pipeline_inventory
        stockout_risk = self._calculate_stockout_probability(
            total_available,
            avg_daily_demand,
            demand_std,
            self.config.lead_time_days
        )
        
        return {
            'product_id': product_id,
            'store_id': store_id,
            'timestamp': datetime.now().isoformat(),
            
            # Demand metrics
            'avg_daily_demand': avg_daily_demand,
            'demand_std': demand_std,
            'annual_demand': annual_demand,
            
            # Reorder Point
            'reorder_point': rop_results['reorder_point'],
            'lead_time_demand': rop_results['lead_time_demand'],
            'safety_stock': rop_results['safety_stock'],
            
            # Order Quantity
            'economic_order_quantity': eoq_results['eoq'],
            'order_frequency_days': eoq_results['order_frequency_days'],
            
            # Current Status
            'current_inventory': current_inventory,
            'days_until_reorder': max(0, days_until_reorder),
            'should_reorder': current_inventory <= rop_results['reorder_point'],
            'stockout_risk': stockout_risk,
            
            # Costs
            'total_annual_cost': eoq_results['total_annual_cost'],
            'service_level': self.config.service_level
        }
    
    def _calculate_stockout_probability(
        self,
        current_inventory: float,
        avg_demand: float,
        demand_std: float,
        time_horizon: int
    ) -> float:
        """
        Calculate probability of stockout within time horizon.
        
        Args:
            current_inventory: Current inventory level
            avg_demand: Average daily demand
            demand_std: Standard deviation of demand
            time_horizon: Days to consider
        
        Returns:
            Probability of stockout (0-1)
        """
        expected_demand = avg_demand * time_horizon
        demand_std_horizon = demand_std * np.sqrt(time_horizon)
        
        if demand_std_horizon == 0:
            return 1.0 if current_inventory < expected_demand else 0.0
        
        z = (current_inventory - expected_demand) / demand_std_horizon
        stockout_prob = 1 - stats.norm.cdf(z)
        
        return stockout_prob
    
    def batch_optimize(
        self,
        forecast_df: pd.DataFrame,
        product_ids: Optional[List[str]] = None,
        store_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Optimize inventory for multiple products/stores.
        
        Args:
            forecast_df: Complete forecast DataFrame
            product_ids: List of product IDs (all if None)
            store_ids: List of store IDs (all if None)
        
        Returns:
            DataFrame with optimization results for all combinations
        """
        if product_ids is None:
            product_ids = forecast_df['product_id'].unique().tolist()
        
        if store_ids is None and 'store_id' in forecast_df.columns:
            store_ids = forecast_df['store_id'].unique().tolist()
        
        results = []
        
        for pid in product_ids:
            if store_ids:
                for sid in store_ids:
                    try:
                        result = self.optimize_inventory_from_forecast(
                            forecast_df, pid, sid
                        )
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to optimize {pid}/{sid}: {e}")
            else:
                try:
                    result = self.optimize_inventory_from_forecast(
                        forecast_df, pid, None
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to optimize {pid}: {e}")
        
        return pd.DataFrame(results)


def generate_inventory_report(
    optimization_results: pd.DataFrame,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate comprehensive inventory management report.
    
    Args:
        optimization_results: DataFrame from batch_optimize()
        output_path: Path to save report (optional)
    
    Returns:
        Summary report DataFrame
    """
    # Aggregate statistics
    summary = {
        'total_products': len(optimization_results),
        'products_needing_reorder': optimization_results['should_reorder'].sum(),
        'avg_days_until_reorder': optimization_results['days_until_reorder'].mean(),
        'high_stockout_risk': (optimization_results['stockout_risk'] > 0.1).sum(),
        'total_annual_cost': optimization_results['total_annual_cost'].sum(),
        'avg_service_level': optimization_results['service_level'].mean()
    }
    
    # Priority reorders (high risk or already below ROP)
    priority_reorders = optimization_results[
        (optimization_results['should_reorder']) |
        (optimization_results['stockout_risk'] > 0.1)
    ].sort_values('stockout_risk', ascending=False)
    
    logger.info(f"Inventory Report Generated:")
    logger.info(f"  Products analyzed: {summary['total_products']}")
    logger.info(f"  Need reorder: {summary['products_needing_reorder']}")
    logger.info(f"  High risk: {summary['high_stockout_risk']}")
    
    if output_path:
        optimization_results.to_csv(output_path, index=False)
        logger.info(f"  Report saved: {output_path}")
    
    return pd.DataFrame([summary])


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample forecast data
    np.random.seed(42)
    dates = pd.date_range('2025-11-15', periods=30, freq='D')
    
    sample_forecast = pd.DataFrame({
        'product_id': ['P001'] * 30,
        'store_id': ['S001'] * 30,
        'timestamp': dates,
        'forecast_q50': np.random.normal(100, 10, 30),  # Avg 100 units/day
        'forecast_q05': np.random.normal(80, 10, 30),
        'forecast_q95': np.random.normal(120, 10, 30),
    })
    
    # Initialize optimizer
    config = InventoryConfig(
        service_level=0.95,
        ordering_cost=50,
        holding_cost_rate=0.20,
        lead_time_days=7,
        unit_cost=10
    )
    
    optimizer = InventoryOptimizer(config)
    
    # Optimize single product
    result = optimizer.optimize_inventory_from_forecast(
        sample_forecast,
        'P001',
        'S001'
    )
    
    print("\n" + "="*70)
    print("INVENTORY OPTIMIZATION RESULTS")
    print("="*70)
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:10.2f}")
        else:
            print(f"{key:30s}: {value}")
    
    print("\n✅ Module 2 - Inventory Optimization Complete")
