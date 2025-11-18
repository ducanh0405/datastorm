#!/usr/bin/env python3
"""
Integrated Insights Generator
==============================
Combines all 4 modules with comprehensive validation.

Flow: Module 1 -> Module 2 -> Module 3 -> Module 4 (validated)

Author: SmartGrocy Team
Date: 2025-11-18
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.modules.llm_insights import LLMInsightGenerator
from src.modules.metrics_validator import MetricsValidator

try:
    from src.modules.inventory_optimization_enhanced import (
        EnhancedInventoryOptimizer,
        InventoryMetrics,
    )
    ENHANCED_INV_AVAILABLE = True
except ImportError:
    from src.modules.inventory_optimization import InventoryOptimizer
    ENHANCED_INV_AVAILABLE = False

try:
    from src.modules.dynamic_pricing_enhanced import EnhancedPricingEngine, PricingMetrics
    ENHANCED_PRICE_AVAILABLE = True
except ImportError:
    from src.modules.dynamic_pricing import DynamicPricingEngine
    ENHANCED_PRICE_AVAILABLE = False

logger = logging.getLogger(__name__)


class IntegratedInsightsGenerator:
    """Generate validated insights from all 4 modules."""

    def __init__(self, use_llm: bool = False, api_key: str | None = None):
        self.use_llm = use_llm
        self.api_key = api_key

        # Initialize modules
        if ENHANCED_INV_AVAILABLE:
            self.inventory_optimizer = EnhancedInventoryOptimizer()
            logger.info("Using Enhanced Inventory Optimizer")
        else:
            self.inventory_optimizer = InventoryOptimizer()
            logger.info("Using Standard Inventory Optimizer")

        if ENHANCED_PRICE_AVAILABLE:
            self.pricing_engine = EnhancedPricingEngine()
            logger.info("Using Enhanced Pricing Engine")
        else:
            self.pricing_engine = DynamicPricingEngine()
            logger.info("Using Standard Pricing Engine")

        self.insight_generator = LLMInsightGenerator(
            use_llm_api=use_llm,
            api_key=api_key
        )

        logger.info("Integrated Insights Generator initialized")

    def generate_validated_insight(
        self,
        product_id: str,
        forecast_data: dict,
        current_inventory: float,
        unit_cost: float,
        current_price: float,
        shap_values: dict | None = None,
        max_retries: int = 3
    ) -> dict:
        """
        Generate comprehensive validated insight.

        Steps:
        1. Validate forecast data (Module 1 output)
        2. Calculate inventory metrics (Module 2)
        3. Calculate pricing metrics (Module 3)
        4. Generate insight (Module 4) with retry

        Args:
            product_id: Product ID
            forecast_data: From Module 1 (q05, q50, q95)
            current_inventory: Current stock level
            unit_cost: Cost per unit
            current_price: Current selling price
            shap_values: Feature importance (optional)
            max_retries: Max retries for LLM API

        Returns:
            Comprehensive insight with all metrics
        """

        # STEP 1: Validate forecast data
        logger.debug(f"Validating forecast for {product_id}...")
        forecast_result = MetricsValidator.validate_forecast_metrics(forecast_data)

        if not forecast_result.is_valid:
            logger.error(f"Forecast validation failed: {forecast_result.errors}")
            return self._generate_error_insight(
                product_id,
                "Invalid forecast data",
                forecast_result.errors
            )

        validated_forecast = forecast_result.validated_data
        logger.debug(f"  [OK] Forecast validated (confidence: {validated_forecast['confidence_level']})")

        # STEP 2: Calculate inventory metrics (Module 2)
        logger.debug("Calculating inventory metrics...")

        try:
            q50 = validated_forecast['q50']
            uncertainty = validated_forecast.get('uncertainty', q50 * 0.2)
            demand_std = uncertainty / 1.645  # Approximate std from 90% interval

            if ENHANCED_INV_AVAILABLE:
                inventory_metrics = self.inventory_optimizer.optimize_with_metrics(
                    avg_daily_demand=q50,
                    demand_std=demand_std,
                    current_inventory=current_inventory,
                    unit_cost=unit_cost
                )
                inventory_data = inventory_metrics.to_dict()
            else:
                inventory_data = {
                    'current_inventory': current_inventory,
                    'safety_stock': 1.645 * demand_std * np.sqrt(7),
                    'reorder_point': q50 * 7 + 1.645 * demand_std * np.sqrt(7),
                    'should_reorder': current_inventory <= (q50 * 7 + 1.645 * demand_std * np.sqrt(7)),
                    'stockout_risk_pct': 5.0,
                    'overstock_risk_pct': 10.0
                }

            logger.debug("  [OK] Inventory metrics calculated")

        except Exception as e:
            logger.error(f"Inventory calculation failed: {e}")
            inventory_data = {'current_inventory': current_inventory, 'safety_stock': 0, 'reorder_point': 0}

        # Validate inventory metrics
        inventory_result = MetricsValidator.validate_inventory_metrics(inventory_data)
        if inventory_result.warnings:
            logger.warning(f"Inventory warnings: {inventory_result.warnings}")
        inventory_data = inventory_result.validated_data or inventory_data

        # STEP 3: Calculate pricing metrics (Module 3)
        logger.debug("Calculating pricing metrics...")

        try:
            inv_ratio = current_inventory / q50 if q50 > 0 else 1.0
            dem_ratio = validated_forecast.get('demand_ratio', 1.0)

            if ENHANCED_PRICE_AVAILABLE:
                pricing_metrics = self.pricing_engine.optimize_with_impact(
                    current_price=current_price,
                    unit_cost=unit_cost,
                    current_demand=q50,
                    inventory_ratio=inv_ratio,
                    demand_ratio=dem_ratio
                )
                pricing_data = pricing_metrics.to_dict()
            else:
                pricing_rec = self.pricing_engine.recommend_price(
                    current_price=current_price,
                    inventory_ratio=inv_ratio,
                    demand_ratio=dem_ratio,
                    cost=unit_cost
                )
                pricing_data = pricing_rec

            logger.debug("  [OK] Pricing metrics calculated")

        except Exception as e:
            logger.error(f"Pricing calculation failed: {e}")
            pricing_data = {'current_price': current_price, 'recommended_price': current_price, 'action': 'maintain'}

        # Validate pricing metrics
        pricing_result = MetricsValidator.validate_pricing_metrics(pricing_data)
        if pricing_result.warnings:
            logger.warning(f"Pricing warnings: {pricing_result.warnings}")
        pricing_data = pricing_result.validated_data or pricing_data

        # Validate SHAP
        shap_result = MetricsValidator.validate_shap_values(shap_values or {})
        shap_data = shap_result.validated_data

        # STEP 4: Generate insight with retry (Module 4)
        logger.debug("Generating insight...")

        insight = self._generate_with_retry(
            product_id,
            validated_forecast,
            inventory_data,
            pricing_data,
            shap_data,
            max_retries
        )

        # Add validation summary
        insight['validation_summary'] = {
            'forecast_valid': forecast_result.is_valid,
            'inventory_valid': inventory_result.is_valid,
            'pricing_valid': pricing_result.is_valid,
            'warnings': (
                forecast_result.warnings +
                inventory_result.warnings +
                pricing_result.warnings
            )
        }

        logger.info(f"[SUCCESS] Insight generated for {product_id} (method: {insight['method']})")

        return insight

    def _generate_with_retry(
        self,
        product_id: str,
        forecast_data: dict,
        inventory_data: dict,
        pricing_data: dict,
        shap_values: dict,
        max_retries: int
    ) -> dict:
        """Generate insight with retry logic."""

        if not self.use_llm or not self.api_key:
            return self.insight_generator._generate_rule_based_insight_comprehensive(
                product_id, forecast_data, inventory_data, pricing_data, shap_values
            )


        for attempt in range(max_retries):
            try:
                logger.debug(f"LLM API attempt {attempt + 1}/{max_retries}")

                return self.insight_generator.generate_comprehensive_insight(
                    product_id, forecast_data, inventory_data, pricing_data, shap_values
                )

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.debug(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

        logger.warning(f"All {max_retries} LLM attempts failed, using rule-based")
        return self.insight_generator._generate_rule_based_insight_comprehensive(
            product_id, forecast_data, inventory_data, pricing_data, shap_values
        )

    def _generate_error_insight(self, product_id: str, error: str, details: list) -> dict:
        """Generate error insight."""
        return {
            'product_id': product_id,
            'insight_text': f"ERROR: {error}\nDetails: {details}",
            'method': 'error',
            'confidence': 0.0,
            'errors': details
        }

    def batch_generate(
        self,
        forecasts_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        unit_costs: dict[str, float],
        current_prices: dict[str, float],
        shap_dict: dict[str, dict] | None = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """Generate insights for top N products."""

        results = []

        # Get top products by forecast
        top_products = forecasts_df.nlargest(top_n, 'forecast_q50')

        for _, row in top_products.iterrows():
            pid = row['product_id']

            try:
                forecast_data = {
                    'q50': row['forecast_q50'],
                    'q05': row.get('forecast_q05', row['forecast_q50'] * 0.8),
                    'q95': row.get('forecast_q95', row['forecast_q50'] * 1.2)
                }

                inv_row = inventory_df[inventory_df['product_id'] == pid].iloc[0]
                current_inv = inv_row['current_inventory']

                insight = self.generate_validated_insight(
                    pid,
                    forecast_data,
                    current_inv,
                    unit_costs.get(pid, 5000),
                    current_prices.get(pid, 10000),
                    shap_dict.get(pid) if shap_dict else None
                )

                results.append(insight)

            except Exception as e:
                logger.error(f"Failed to generate insight for {pid}: {e}")

        return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = IntegratedInsightsGenerator(use_llm=False)

    forecast = {'q50': 150, 'q05': 100, 'q95': 200}

    insight = generator.generate_validated_insight(
        'TEST001',
        forecast,
        current_inventory=120,
        unit_cost=30000,
        current_price=50000
    )

    print("\n" + "="*70)
    print("INTEGRATED INSIGHT")
    print("="*70)
    try:
        print(insight['insight_text'])
    except UnicodeEncodeError:
        # Handle Windows console encoding issues
        clean_text = insight['insight_text'].encode('ascii', 'ignore').decode('ascii')
        print(clean_text)
    print(f"\nMethod: {insight['method']}")
    print(f"Confidence: {insight['confidence']:.1%}")
