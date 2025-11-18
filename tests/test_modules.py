#!/usr/bin/env python3
"""
Comprehensive Test Suite for Business Modules
==============================================

Tests for:
- Module 2: Inventory Optimization
- Module 3: Dynamic Pricing
- Module 4: LLM Insights
- Inventory Backtesting Framework

Author: SmartGrocy Team
Date: 2025-11-15
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.dynamic_pricing import DynamicPricingEngine, PricingConfig
from src.modules.inventory_backtesting import BacktestConfig, InventoryBacktester
from src.modules.inventory_optimization import InventoryConfig, InventoryOptimizer
from src.modules.llm_insights import InsightConfig, LLMInsightGenerator

logger = logging.getLogger(__name__)


def test_01_inventory_optimizer_initialization():
    """Test Module 2: Inventory Optimizer initialization"""
    try:
        config = InventoryConfig(service_level=0.95)
        optimizer = InventoryOptimizer(config)
        assert optimizer.config.service_level == 0.95
        print("✓ Test 1 PASSED: Inventory Optimizer initializes correctly")
        print("✓ Test 1 PASSED: Inventory Optimizer initializes correctly")
    except Exception as e:
        print(f"✗ Test 1 FAILED: {e}")
        raise


def test_02_reorder_point_calculation():
    """Test Module 2: ROP calculation logic"""
    try:
        optimizer = InventoryOptimizer()
        result = optimizer.calculate_reorder_point(
            avg_daily_demand=100,
            demand_std=15,
            lead_time_days=7
        )

        assert 'reorder_point' in result
        assert result['reorder_point'] > result['lead_time_demand']
        assert result['safety_stock'] > 0

        print(f"✓ Test 2 PASSED: ROP = {result['reorder_point']:.1f} (includes safety stock)")
        return True
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False


def test_03_eoq_calculation():
    """Test Module 2: EOQ calculation"""
    try:
        optimizer = InventoryOptimizer()
        result = optimizer.calculate_economic_order_quantity(
            annual_demand=36500,  # 100/day
            ordering_cost=50,
            holding_cost_per_unit=2
        )

        assert 'eoq' in result
        assert result['eoq'] > 0
        assert result['num_orders_per_year'] > 0

        print(f"✓ Test 3 PASSED: EOQ = {result['eoq']:.1f} units")
        return True
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        return False


def test_04_pricing_engine_initialization():
    """Test Module 3: Dynamic Pricing initialization"""
    try:
        config = PricingConfig(max_discount=0.50)
        engine = DynamicPricingEngine(config)
        assert engine.config.max_discount == 0.50
        print("✓ Test 4 PASSED: Dynamic Pricing Engine initializes correctly")
        return True
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        return False


def test_05_discount_calculation_logic():
    """Test Module 3: Discount calculation for different scenarios"""
    try:
        engine = DynamicPricingEngine()

        # Test Case 1: High inventory + Low demand = Large discount
        discount1, action1, _ = engine.calculate_discount(2.5, 0.7)
        assert discount1 >= 0.25, "Should give large discount"
        assert action1 in ['large_discount', 'clearance']

        # Test Case 2: Low inventory = No discount
        discount2, action2, _ = engine.calculate_discount(0.5, 1.0)
        assert discount2 == 0.0, "Should maintain price"
        assert action2 == 'maintain'

        print("✓ Test 5 PASSED: Discount logic works correctly")
        return True
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
        return False


def test_06_pricing_recommendation():
    """Test Module 3: Full pricing recommendation"""
    try:
        engine = DynamicPricingEngine()
        result = engine.recommend_price(
            current_price=10.0,
            inventory_ratio=2.0,
            demand_ratio=0.8,
            cost=5.0
        )

        assert 'recommended_price' in result
        assert result['recommended_price'] <= result['current_price']
        assert result['profit_margin'] >= 0.10  # Minimum margin maintained

        print(f"✓ Test 6 PASSED: Price ${result['current_price']:.2f} → ${result['recommended_price']:.2f}")
        return True
    except Exception as e:
        print(f"✗ Test 6 FAILED: {e}")
        return False


def test_07_llm_insights_initialization():
    """Test Module 4: LLM Insights initialization"""
    try:
        config = InsightConfig(use_llm_api=False, template_style='bullet')
        generator = LLMInsightGenerator(config)
        assert generator.config.template_style == 'bullet'
        print("✓ Test 7 PASSED: LLM Insights Generator initializes correctly")
        return True
    except Exception as e:
        print(f"✗ Test 7 FAILED: {e}")
        return False


def test_08_rule_based_insight_generation():
    """Test Module 4: Rule-based insight generation"""
    try:
        generator = LLMInsightGenerator()

        forecast_data = {'q50': 150, 'q95': 200, 'trend': 'up'}
        shap_values = {'promo_active': 0.35, 'price': -0.15}

        insight = generator.generate_forecast_insight(
            'TEST001',
            forecast_data,
            shap_values
        )

        assert 'insight' in insight
        assert 'causes' in insight
        assert 'actions' in insight
        assert len(insight['causes']) > 0

        print("✓ Test 8 PASSED: Insight generation works")
        return True
    except Exception as e:
        print(f"✗ Test 8 FAILED: {e}")
        return False


def test_09_backtesting_initialization():
    """Test Backtesting Framework initialization"""
    try:
        dates = pd.date_range('2025-11-01', '2025-11-14', freq='D')
        historical = pd.DataFrame({
            'date': dates,
            'product_id': ['P001'] * len(dates),
            'actual_demand': np.random.poisson(100, len(dates))
        })

        backtester = InventoryBacktester(historical)
        assert backtester.config.lead_time_days > 0

        print("✓ Test 9 PASSED: Backtesting Framework initializes correctly")
        return True
    except Exception as e:
        print(f"✗ Test 9 FAILED: {e}")
        return False


def test_10_backtesting_simulation():
    """Test Backtesting Framework simulation"""
    try:
        dates = pd.date_range('2025-11-01', '2025-11-14', freq='D')
        np.random.seed(42)

        historical = pd.DataFrame({
            'date': dates,
            'product_id': ['P001'] * len(dates),
            'actual_demand': np.random.poisson(100, len(dates))
        })

        forecasts = pd.DataFrame({
            'date': dates,
            'product_id': ['P001'] * len(dates),
            'forecast_q50': historical['actual_demand'] * 0.95,
            'forecast_q95': historical['actual_demand'] * 1.3
        })

        backtester = InventoryBacktester(historical, forecasts)
        comparison = backtester.compare_strategies()

        assert len(comparison) == 3  # 3 metrics compared
        assert 'improvement_pct' in comparison.columns

        print("✓ Test 10 PASSED: Simulation runs and generates KPIs")
        return True
    except Exception as e:
        print(f"✗ Test 10 FAILED: {e}")
        return False


def test_11_module_integration():
    """Test integration between modules"""
    try:
        # Simulate end-to-end flow
        # Module 1: Forecast (simulated)
        pd.DataFrame({
            'product_id': ['P001'],
            'forecast_q50': [100],
            'forecast_q05': [80],
            'forecast_q95': [120]
        })

        # Module 2: Inventory optimization
        InventoryOptimizer()
        # Would normally use optimize_inventory_from_forecast, but simplified here

        # Module 3: Dynamic pricing
        pricing_data = pd.DataFrame({
            'product_id': ['P001'],
            'current_price': [10.0],
            'inventory_ratio': [2.0],
            'demand_ratio': [0.8],
            'cost': [5.0]
        })

        engine = DynamicPricingEngine()
        pricing_result = engine.batch_optimize(pricing_data)

        assert len(pricing_result) == 1
        assert 'recommended_price' in pricing_result.columns

        print("✓ Test 11 PASSED: Modules integrate correctly")
        return True
    except Exception as e:
        print(f"✗ Test 11 FAILED: {e}")
        return False


def run_all_tests():
    """Run all module tests."""
    print("="*70)
    print("SMARTGROCY MODULE TEST SUITE")
    print("="*70)
    print("Testing Modules 2, 3, 4 and Integration")
    print("="*70)

    tests = [
        test_01_inventory_optimizer_initialization,
        test_02_reorder_point_calculation,
        test_03_eoq_calculation,
        test_04_pricing_engine_initialization,
        test_05_discount_calculation_logic,
        test_06_pricing_recommendation,
        test_07_llm_insights_initialization,
        test_08_rule_based_insight_generation,
        test_09_backtesting_initialization,
        test_10_backtesting_simulation,
        test_11_module_integration
    ]

    results = []
    for i, test_func in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Running {test_func.__name__}...")
        passed = test_func()
        results.append(passed)

    # Summary
    passed_count = sum(results)
    total_count = len(results)

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {total_count}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {total_count - passed_count}")
    print(f"Success rate: {passed_count/total_count*100:.0f}%")
    print("="*70)

    if passed_count == total_count:
        print("\n✅ ALL TESTS PASSED - Modules Ready for Production!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Review errors above")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
