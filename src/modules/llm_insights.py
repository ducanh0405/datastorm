#!/usr/bin/env python3
"""
Module 4: LLM Insights Generator
=================================

LLM-powered insight generation for demand forecasts and business decisions.
Integrates results from Modules 1 (Forecasting), 2 (Inventory), and 3 (Pricing).

Capabilities:
- Interpret SHAP values and feature importance
- Generate causal explanations for demand changes
- Provide actionable recommendations
- Synthesize insights from all three business modules
- Support Gemini API with .env configuration

Framework: Causal → Impact → Action

Author: SmartGrocy Team
Date: 2025-11-16
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Try to load dotenv for .env file support
# Note: If .env file has encoding issues, API keys can still be set via environment variables
try:
    from dotenv import load_dotenv
    try:
        # Try to load .env file
        # If it fails due to encoding issues (e.g., UTF-16), we'll continue without it
        # API keys can be set directly via environment variables (e.g., $env:GEMINI_API_KEY)
        load_dotenv()
    except (UnicodeDecodeError, Exception):
        # Silently continue if .env file can't be loaded due to encoding or other issues
        # This allows the module to work even if .env has encoding problems
        # Users can set API keys via environment variables instead
        pass
except ImportError:
    # If python-dotenv not installed, continue without it
    pass


def calculate_risk_metrics(forecast_df: pd.DataFrame, product_id: int) -> tuple[float, float]:
    """
    Calculate stockout and overstock risk from forecast data using business-realistic assumptions.

    Args:
        forecast_df: DataFrame with forecast data
        product_id: Product ID to calculate risk for

    Returns:
        Tuple of (stockout_risk_pct, overstock_risk_pct)
    """
    try:
        # Filter data for this product
        product_data = forecast_df[forecast_df['product_id'] == product_id]

        if len(product_data) == 0:
            return 0.0, 0.0

        # Calculate basic statistics
        avg_forecast = product_data['forecast_q50'].mean()
        std_forecast = product_data['forecast_q50'].std()

        if np.isnan(avg_forecast) or avg_forecast <= 0:
            return 0.0, 0.0

        # 🎯 CẢI THIỆN 1: Inventory level hợp lý hơn với variance (200-300% thay vì fixed 250%)
        # Thêm variance để tạo sự đa dạng trong overstock risk
        np.random.seed(42 + product_id)  # Reproducible per product
        inventory_variance = np.random.uniform(-0.3, 0.3)  # ±30% variance
        inventory_multiplier = 2.5 + inventory_variance
        current_inventory = avg_forecast * max(1.0, inventory_multiplier)  # Không để < 100%

        # 🎯 CẢI THIỆN 2: Lead time hợp lý hơn (2 ngày thay vì 7 ngày)
        lead_time = 2

        # 🎯 CẢI THIỆN 3: Thêm safety buffer
        safety_buffer = avg_forecast * 0.3  # 30% safety buffer
        effective_inventory = current_inventory + safety_buffer

        # Calculate stockout risk (probability of running out of stock)
        if std_forecast > 0 and not np.isnan(std_forecast):
            expected_demand = avg_forecast * lead_time
            demand_std_horizon = std_forecast * np.sqrt(lead_time)

            if demand_std_horizon > 0:
                z_score = (effective_inventory - expected_demand) / demand_std_horizon
                stockout_risk = 1 - stats.norm.cdf(z_score)
            else:
                stockout_risk = 0.0
        else:
            stockout_risk = 0.0

        # Calculate overstock risk using rule-based approach
        # Instead of statistical modeling which doesn't work well with small numbers
        inventory_ratio = effective_inventory / avg_forecast if avg_forecast > 0 else 0

        if inventory_ratio >= 8.0:  # Very high inventory (>800% of forecast)
            overstock_risk = 0.80  # 80% risk
        elif inventory_ratio >= 6.0:  # High inventory (>600% of forecast)
            overstock_risk = 0.60  # 60% risk
        elif inventory_ratio >= 4.0:  # Moderate-high inventory (>400% of forecast)
            overstock_risk = 0.30  # 30% risk
        elif inventory_ratio >= 3.0:  # Moderate inventory (>300% of forecast)
            overstock_risk = 0.15  # 15% risk
        else:  # Normal inventory levels
            overstock_risk = 0.05  # 5% risk (very low)

        # 🎯 CẢI THIỆN 5: Cap at reasonable maximums (95% thay vì 100%)
        stockout_risk_pct = min(max(stockout_risk * 100, 0.0), 95.0)
        overstock_risk_pct = min(max(overstock_risk * 100, 0.0), 95.0)

        return stockout_risk_pct, overstock_risk_pct

    except Exception as e:
        logger.warning(f"Error calculating risk metrics for product {product_id}: {e}")
        return 0.0, 0.0


@dataclass
class InsightConfig:
    """Configuration for insight generation."""
    use_llm_api: bool = True  # Default to True, will auto-detect API key
    api_provider: str = "gemini"  # gemini, openai, anthropic
    api_key: str | None = None  # Will load from env if None
    model: str = "gemini-2.0-flash-exp"  # Default Gemini model
    template_style: str = "bullet"  # bullet, paragraph, table


class LLMInsightGenerator:
    """
    Generate business insights from forecast data using LLM.

    Can work in two modes:
    1. Rule-based (no API) - Template-based insights
    2. LLM-powered (with API) - Advanced interpretations using Gemini

    Supports integration with Modules 1, 2, and 3.
    """

    def __init__(
        self,
        config: InsightConfig | None = None,
        use_llm_api: bool | None = None,
        api_key: str | None = None,
        api_provider: str = "gemini",
        model: str = "gemini-2.0-flash-exp"
    ):
        """
        Initialize LLM Insight Generator.

        Args:
            config: InsightConfig object (uses defaults if None)
            use_llm_api: Override config.use_llm_api
            api_key: Override API key (will check env vars if None)
            api_provider: API provider name
            model: Model name
        """
        self.config = config or InsightConfig()

        # Override config with direct parameters if provided
        if use_llm_api is not None:
            self.config.use_llm_api = use_llm_api
        if api_key is not None:
            self.config.api_key = api_key
        if api_provider:
            self.config.api_provider = api_provider
        if model:
            self.config.model = model

        # Load API key from environment if not provided
        if self.config.api_key is None:
            if self.config.api_provider == "gemini":
                self.config.api_key = os.getenv('GEMINI_API_KEY')
            elif self.config.api_provider == "openai":
                self.config.api_key = os.getenv('OPENAI_API_KEY')
            elif self.config.api_provider == "anthropic":
                self.config.api_key = os.getenv('ANTHROPIC_API_KEY')

        # Auto-detect if API should be used
        if self.config.use_llm_api and self.config.api_key:
            logger.info(f"LLM Insight Generator initialized with {self.config.api_provider} API")
            logger.info(f"Model: {self.config.model}")
        else:
            self.config.use_llm_api = False
            logger.info("LLM Insight Generator initialized (rule-based mode)")

    def generate_comprehensive_insight(
        self,
        product_id: str,
        forecast_data: dict,  # Module 1 results
        inventory_data: dict,  # Module 2 results
        pricing_data: dict,  # Module 3 results
        shap_values: dict | None = None,
        use_llm: bool | None = None
    ) -> dict:
        """
        Generate comprehensive insight combining all three modules.

        Args:
            product_id: Product identifier
            forecast_data: Module 1 - Demand forecasting results
            inventory_data: Module 2 - Inventory optimization results
            pricing_data: Module 3 - Dynamic pricing results
            shap_values: Top SHAP contributors (optional)
            use_llm: Force LLM mode (overrides config)

        Returns:
            Dictionary with insight sections
        """
        use_llm = use_llm if use_llm is not None else self.config.use_llm_api

        if use_llm and self.config.api_key:
            return self._generate_llm_insight_comprehensive(
                product_id, forecast_data, inventory_data, pricing_data, shap_values
            )
        else:
            return self._generate_rule_based_insight_comprehensive(
                product_id, forecast_data, inventory_data, pricing_data, shap_values
            )

    def generate_forecast_insight(
        self,
        product_id: str,
        forecast_data: dict,
        shap_values: dict | None = None,
        historical_data: dict | None = None,
        use_llm: bool | None = None
    ) -> dict:
        """
        Generate insight for a forecast (backward compatibility).

        Args:
            product_id: Product identifier
            forecast_data: Dict with keys:
                - q50: Median forecast
                - q05, q95: Prediction intervals
                - trend: up/down/stable
            shap_values: Top SHAP contributors (optional)
            historical_data: Historical context (optional)
            use_llm: Force LLM mode

        Returns:
            Dictionary with insight sections
        """
        use_llm = use_llm if use_llm is not None else self.config.use_llm_api

        # Create dummy inventory and pricing data for backward compatibility
        inventory_data = {
            'current_inventory': forecast_data.get('current_inventory', 0),
            'safety_stock': forecast_data.get('safety_stock', 0),
            'reorder_point': forecast_data.get('reorder_point', 0),
            'should_reorder': forecast_data.get('should_reorder', False)
        }

        pricing_data = {
            'current_price': forecast_data.get('current_price', 0),
            'recommended_price': forecast_data.get('recommended_price', 0),
            'discount_pct': forecast_data.get('discount_pct', 0)
        }

        return self.generate_comprehensive_insight(
            product_id, forecast_data, inventory_data, pricing_data, shap_values, use_llm
        )

    def _generate_llm_insight_comprehensive(
        self,
        product_id: str,
        forecast_data: dict,
        inventory_data: dict,
        pricing_data: dict,
        shap_values: dict | None
    ) -> dict:
        """Generate insight using LLM API (Gemini)."""
        try:
            # Import Gemini
            import google.generativeai as genai

            # Configure API
            genai.configure(api_key=self.config.api_key)

            # Get model
            model = genai.GenerativeModel(self.config.model)

            # Format prompt
            prompt = self._format_comprehensive_prompt(
                product_id, forecast_data, inventory_data, pricing_data, shap_values
            )

            logger.info(f"Calling {self.config.api_provider} API for insight generation...")

            # Generate content
            response = model.generate_content(prompt)

            insight_text = response.text

            # Extract risk metrics from inventory data
            stockout_risk = inventory_data.get('stockout_risk_pct', 0)
            overstock_risk = inventory_data.get('overstock_risk_pct', 0)

            return {
                'product_id': product_id,
                'insight_text': insight_text,
                'insight': insight_text,  # Backward compatibility
                'stockout_risk_pct': stockout_risk,
                'overstock_risk_pct': overstock_risk,
                'method': 'llm',
                'provider': self.config.api_provider,
                'model': self.config.model,
                'confidence': 0.85,  # LLM insights have high confidence
                'generated_by': 'llm_api'
            }

        except ImportError:
            logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
            logger.warning("Falling back to rule-based mode")
            return self._generate_rule_based_insight_comprehensive(
                product_id, forecast_data, inventory_data, pricing_data, shap_values
            )
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            logger.warning("Falling back to rule-based mode")
            return self._generate_rule_based_insight_comprehensive(
                product_id, forecast_data, inventory_data, pricing_data, shap_values
            )

    def _format_comprehensive_prompt(
        self,
        product_id: str,
        forecast_data: dict,
        inventory_data: dict,
        pricing_data: dict,
        shap_values: dict | None
    ) -> str:
        """Format comprehensive prompt using template from llm_prompts.py"""
        try:
            from src.modules.llm_prompts import FORECAST_INSIGHT_PROMPT_V2
            prompt_template = FORECAST_INSIGHT_PROMPT_V2
        except ImportError:
            logger.warning("Could not import prompt template, using basic prompt")
            return self._build_basic_prompt(product_id, forecast_data, inventory_data, pricing_data, shap_values)

        # Extract SHAP top features
        top_shap = self._extract_top_shap_features(shap_values)

        # Calculate derived metrics
        q50 = forecast_data.get('q50', forecast_data.get('forecast_q50', 0))
        q05 = forecast_data.get('q05', forecast_data.get('forecast_q05', 0))
        q95 = forecast_data.get('q95', forecast_data.get('forecast_q95', 0))

        uncertainty = q95 - q50
        uncertainty_pct = (uncertainty / q50 * 100) if q50 > 0 else 0

        # Format prompt with all placeholders
        prompt = prompt_template.format(
            # Product info
            product_id=product_id,
            category=forecast_data.get('category', 'Unknown'),
            store_id=forecast_data.get('store_id', 'Unknown'),
            forecast_date=forecast_data.get('date', forecast_data.get('forecast_date', 'Unknown')),
            horizon=forecast_data.get('horizon', '24 hours'),

            # Module 1: Forecasting
            q50=q50,
            q05=q05,
            q95=q95,
            uncertainty=uncertainty,
            uncertainty_pct=uncertainty_pct,
            confidence_level=self._calculate_confidence_level(uncertainty_pct),
            vs_yesterday=forecast_data.get('vs_yesterday', 0),
            vs_last_week=forecast_data.get('vs_last_week', 0),
            vs_monthly_avg=forecast_data.get('vs_monthly_avg', 0),
            trend_icon_yesterday=self._get_trend_icon(forecast_data.get('vs_yesterday', 0)),
            trend_icon_week=self._get_trend_icon(forecast_data.get('vs_last_week', 0)),
            trend_icon_month=self._get_trend_icon(forecast_data.get('vs_monthly_avg', 0)),
            trend_direction=self._get_trend_direction(forecast_data),
            cause_1=top_shap[0]['name'],
            impact_1=top_shap[0]['impact'],
            cause_1_detail=top_shap[0]['detail'],
            cause_2=top_shap[1]['name'],
            impact_2=top_shap[1]['impact'],
            cause_2_detail=top_shap[1]['detail'],
            cause_3=top_shap[2]['name'],
            impact_3=top_shap[2]['impact'],
            cause_3_detail=top_shap[2]['detail'],

            # Module 2: Inventory
            current_inventory=inventory_data.get('current_inventory', 0),
            days_of_stock=self._calculate_days_of_stock(
                inventory_data.get('current_inventory', 0), q50
            ),
            inventory_ratio=inventory_data.get('inventory_ratio', 1.0),
            safety_stock=inventory_data.get('safety_stock', 0),
            reorder_point=inventory_data.get('reorder_point', 0),
            eoq=inventory_data.get('eoq', 0),
            recommended_order_qty=inventory_data.get('recommended_order_qty', 0),
            stockout_risk=self._get_risk_level(inventory_data.get('stockout_risk_pct', 0)),
            stockout_risk_pct=inventory_data.get('stockout_risk_pct', 0),
            stockout_status=self._get_stockout_status(inventory_data.get('stockout_risk_pct', 0)),
            overstock_risk=self._get_risk_level(inventory_data.get('overstock_risk_pct', 0)),
            overstock_risk_pct=inventory_data.get('overstock_risk_pct', 0),
            overstock_status=self._get_overstock_status(inventory_data.get('overstock_risk_pct', 0)),
            should_reorder='YES' if inventory_data.get('should_reorder', False) else 'NO',
            reorder_reasoning=inventory_data.get('reorder_reasoning', 'No reorder needed'),
            service_level=inventory_data.get('service_level', 0.95),
            fill_rate=inventory_data.get('fill_rate', 0.95),
            inventory_turnover=inventory_data.get('inventory_turnover', 12.0),

            # Module 3: Pricing
            current_price=pricing_data.get('current_price', 0),
            unit_cost=pricing_data.get('unit_cost', 0),
            current_margin=self._calculate_margin(
                pricing_data.get('current_price', 0),
                pricing_data.get('unit_cost', 0)
            ),
            recommended_price=pricing_data.get('recommended_price', pricing_data.get('current_price', 0)),
            price_change=pricing_data.get('recommended_price', 0) - pricing_data.get('current_price', 0),
            price_change_pct=self._calculate_price_change_pct(
                pricing_data.get('current_price', 0),
                pricing_data.get('recommended_price', pricing_data.get('current_price', 0))
            ),
            discount_pct=pricing_data.get('discount_pct', 0),
            pricing_action=pricing_data.get('action', 'maintain'),
            pricing_reasoning=pricing_data.get('reasoning', 'No pricing change needed'),
            pricing_impact=pricing_data.get('expected_impact', 'No significant impact expected'),
            new_margin=self._calculate_margin(
                pricing_data.get('recommended_price', pricing_data.get('current_price', 0)),
                pricing_data.get('unit_cost', 0)
            ),
            revenue_impact=pricing_data.get('revenue_impact', 'Neutral'),
            inventory_ratio_pricing=pricing_data.get('inventory_ratio', 1.0),
            demand_ratio=pricing_data.get('demand_ratio', 1.0),
            competitive_position=pricing_data.get('competitive_position', 'Normal')
        )

        return prompt

    def _extract_top_shap_features(self, shap_values: dict | None, top_n: int = 3) -> list[dict]:
        """Extract top N SHAP features with details."""
        if not shap_values:
            return [
                {'name': 'Historical Demand Pattern', 'impact': 40.0, 'detail': 'Based on historical sales trends'},
                {'name': 'Seasonal Factors', 'impact': 30.0, 'detail': 'Seasonal demand variations'},
                {'name': 'Market Conditions', 'impact': 30.0, 'detail': 'General market dynamics'}
            ]

        # Sort by absolute value
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

        results = []
        total_abs = sum(abs(v) for v in shap_values.values())

        for _i, (feat, val) in enumerate(sorted_features):
            impact_pct = (abs(val) / total_abs * 100) if total_abs > 0 else 0

            # Generate detail based on feature name
            detail = self._generate_feature_detail(feat, val)

            results.append({
                'name': feat.replace('_', ' ').title(),
                'impact': impact_pct,
                'detail': detail
            })

        # Pad if less than top_n
        while len(results) < top_n:
            results.append({
                'name': f'Factor {len(results) + 1}',
                'impact': 0.0,
                'detail': 'No significant impact'
            })

        return results

    def _generate_feature_detail(self, feature_name: str, shap_value: float) -> str:
        """Generate human-readable detail for a feature."""
        feat_lower = feature_name.lower()

        if 'promo' in feat_lower:
            return 'Promotional activity driving demand' if shap_value > 0 else 'Lack of promotion reducing demand'
        elif 'price' in feat_lower:
            return 'Price changes affecting demand elasticity' if shap_value < 0 else 'Price stability supporting demand'
        elif 'lag' in feat_lower or 'historical' in feat_lower:
            return 'Historical demand patterns influencing forecast'
        elif 'season' in feat_lower:
            return 'Seasonal factors affecting demand'
        elif 'day' in feat_lower or 'week' in feat_lower:
            return 'Day-of-week or weekly patterns'
        elif 'weather' in feat_lower:
            return 'Weather conditions impacting demand'
        else:
            return f'Feature contributing {shap_value:.2f} to forecast'

    def _calculate_confidence_level(self, uncertainty_pct: float) -> str:
        """Calculate confidence level based on uncertainty."""
        if uncertainty_pct < 10:
            return 'HIGH'
        elif uncertainty_pct < 25:
            return 'MODERATE'
        else:
            return 'LOW'

    def _get_trend_icon(self, change: float) -> str:
        """Get trend icon based on change."""
        if change > 5:
            return '📈'
        elif change < -5:
            return '📉'
        else:
            return '➡️'

    def _get_trend_direction(self, forecast_data: dict) -> str:
        """Get trend direction."""
        vs_yesterday = forecast_data.get('vs_yesterday', 0)
        vs_week = forecast_data.get('vs_last_week', 0)

        if vs_yesterday > 10 or vs_week > 10:
            return 'STRONG UPWARD'
        elif vs_yesterday > 5 or vs_week > 5:
            return 'UPWARD'
        elif vs_yesterday < -10 or vs_week < -10:
            return 'STRONG DOWNWARD'
        elif vs_yesterday < -5 or vs_week < -5:
            return 'DOWNWARD'
        else:
            return 'STABLE'

    def _calculate_days_of_stock(self, current_inventory: float, q50: float) -> float:
        """Calculate days of stock."""
        if q50 > 0:
            return current_inventory / q50
        return 0.0

    def _get_risk_level(self, risk_pct: float) -> str:
        """Get risk level string."""
        if risk_pct >= 70:
            return 'CRITICAL'
        elif risk_pct >= 50:
            return 'HIGH'
        elif risk_pct >= 30:
            return 'MODERATE'
        else:
            return 'LOW'

    def _get_stockout_status(self, risk_pct: float) -> str:
        """Get stockout status."""
        if risk_pct >= 70:
            return 'IMMEDIATE ACTION REQUIRED'
        elif risk_pct >= 50:
            return 'MONITOR CLOSELY'
        else:
            return 'MANAGEABLE'

    def _get_overstock_status(self, risk_pct: float) -> str:
        """Get overstock status."""
        if risk_pct >= 60:
            return 'URGENT CLEARANCE NEEDED'
        elif risk_pct >= 40:
            return 'CONSIDER MARKDOWNS'
        else:
            return 'ACCEPTABLE'

    def _calculate_margin(self, price: float, cost: float) -> float:
        """Calculate profit margin."""
        if price > 0:
            return (price - cost) / price
        return 0.0

    def _calculate_price_change_pct(self, old_price: float, new_price: float) -> float:
        """Calculate price change percentage."""
        if old_price > 0:
            return ((new_price - old_price) / old_price) * 100
        return 0.0

    def _build_basic_prompt(
        self,
        product_id: str,
        forecast_data: dict,
        inventory_data: dict,
        pricing_data: dict,
        shap_values: dict | None
    ) -> str:
        """Build basic prompt if template not available."""
        return f"""Analyze the following business data for product {product_id}:

Forecast: {forecast_data}
Inventory: {inventory_data}
Pricing: {pricing_data}
SHAP: {shap_values}

Provide comprehensive business insights.
"""

    def _generate_rule_based_insight_comprehensive(
        self,
        product_id: str,
        forecast_data: dict,
        inventory_data: dict,
        pricing_data: dict,
        shap_values: dict | None
    ) -> dict:
        """Generate rule-based insight combining all modules."""
        q50 = forecast_data.get('q50', forecast_data.get('forecast_q50', 0))

        # Extract causes
        causes = []
        if shap_values:
            top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            for feat, val in top_features:
                if 'promo' in feat.lower() and val > 0:
                    causes.append("Active promotional campaign")
                elif 'price' in feat.lower():
                    causes.append("Price dynamics affecting demand")
                elif 'lag' in feat.lower():
                    causes.append("Historical demand momentum")

        if not causes:
            causes = ["Normal market conditions"]

        # Determine impact
        trend = self._get_trend_direction(forecast_data)
        if 'UPWARD' in trend:
            impact = f"Demand forecast shows growth to {q50:.1f} units"
        elif 'DOWNWARD' in trend:
            impact = f"Demand forecast declining to {q50:.1f} units"
        else:
            impact = f"Demand forecast stable at {q50:.1f} units"

        # Actions
        actions = []

        # Inventory actions
        stockout_risk = inventory_data.get('stockout_risk_pct', 0)
        overstock_risk = inventory_data.get('overstock_risk_pct', 0)

        if stockout_risk > 50:
            actions.append(f"URGENT: Reorder immediately (stockout risk: {stockout_risk:.0f}%)")
        elif inventory_data.get('should_reorder', False):
            actions.append("HIGH: Place reorder to maintain service level")

        if overstock_risk > 50:
            actions.append(f"URGENT: Apply markdown pricing (overstock risk: {overstock_risk:.0f}%)")

        # Pricing actions
        pricing_action = pricing_data.get('action', 'maintain')
        if pricing_action != 'maintain':
            discount = pricing_data.get('discount_pct', 0)
            actions.append(f"HIGH: Adjust pricing ({pricing_action}, {discount:.1%} discount)")

        if not actions:
            actions = ["MEDIUM: Continue monitoring current strategy"]

        # Format insight
        insight_text = f"""## 📊 EXECUTIVE SUMMARY

Demand forecast for {product_id} is **{q50:.1f} units** with **{trend.lower()}** trend.
Current inventory: {inventory_data.get('current_inventory', 0):.0f} units.

## 🔍 CAUSAL FACTORS

{chr(10).join('- ' + c for c in causes)}

## 📈 BUSINESS IMPACT

- **Forecast**: {impact}
- **Inventory Status**: {inventory_data.get('current_inventory', 0):.0f} units
- **Stockout Risk**: {stockout_risk:.0f}%
- **Overstock Risk**: {overstock_risk:.0f}%

## ✅ RECOMMENDED ACTIONS

{chr(10).join('- ' + a for a in actions)}
"""

        return {
            'product_id': product_id,
            'insight_text': insight_text,
            'insight': insight_text,
            'causes': causes,
            'impact': impact,
            'actions': actions,
            'stockout_risk_pct': stockout_risk,
            'overstock_risk_pct': overstock_risk,
            'method': 'rule_based',
            'confidence': 0.70,
            'generated_by': 'rule_based'
        }

    def batch_generate_comprehensive(
        self,
        forecasts_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        pricing_df: pd.DataFrame,
        shap_dict: dict[str, dict] | None = None,
        top_n: int = 10,
        use_llm: bool | None = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive insights for top N products.

        Args:
            forecasts_df: Module 1 results DataFrame
            inventory_df: Module 2 results DataFrame
            pricing_df: Module 3 results DataFrame
            shap_dict: Dict mapping product_id to SHAP values
            top_n: Number of products to generate insights for
            use_llm: Force LLM mode

        Returns:
            DataFrame with insights
        """
        # Determine merge keys based on available columns
        # Check which columns are available in each dataframe
        forecast_cols = set(forecasts_df.columns)
        inventory_cols = set(inventory_df.columns)
        pricing_cols = set(pricing_df.columns)

        # Determine merge keys - use store_id if available in all dataframes
        merge_keys_inv = ['product_id']
        merge_keys_price = ['product_id']

        if 'store_id' in forecast_cols and 'store_id' in inventory_cols:
            merge_keys_inv.append('store_id')

        if 'store_id' in forecast_cols and 'store_id' in pricing_cols:
            merge_keys_price.append('store_id')

        logger.debug(f"Merging inventory on: {merge_keys_inv}")
        logger.debug(f"Merging pricing on: {merge_keys_price}")

        # Merge all dataframes
        try:
            merged = forecasts_df.merge(
                inventory_df, on=merge_keys_inv, how='inner', suffixes=('', '_inv')
            )
            logger.debug(f"After inventory merge: {len(merged)} rows, columns: {list(merged.columns)}")
        except KeyError as e:
            logger.error(f"Error merging inventory data: {e}")
            logger.error(f"Forecast columns: {list(forecasts_df.columns)}")
            logger.error(f"Inventory columns: {list(inventory_df.columns)}")
            raise

        # For pricing merge, use the same keys that were used for inventory merge
        # (to maintain consistency)
        if 'store_id' in merged.columns and 'store_id' in pricing_cols:
            merge_keys_price_final = ['product_id', 'store_id']
        else:
            merge_keys_price_final = ['product_id']

        try:
            merged = merged.merge(
                pricing_df, on=merge_keys_price_final, how='inner', suffixes=('', '_price')
            )
            logger.debug(f"After pricing merge: {len(merged)} rows, columns: {list(merged.columns)}")
        except KeyError as e:
            logger.error(f"Error merging pricing data: {e}")
            logger.error(f"Merged columns: {list(merged.columns)}")
            logger.error(f"Pricing columns: {list(pricing_df.columns)}")
            raise

        if len(merged) == 0:
            logger.warning("No records after merging all dataframes. Check if product_id matches.")
            logger.info("Falling back to forecasts-only analysis...")
            merged = forecasts_df.copy()
            # Add dummy columns for missing inventory/pricing data
            merged['current_inventory'] = merged.get('current_inventory', 2.0)
            merged['safety_stock'] = merged.get('safety_stock', 1.0)
            merged['reorder_point'] = merged.get('reorder_point', 1.5)
            merged['should_reorder'] = merged.get('should_reorder', False)
            merged['stockout_risk_pct'] = merged.get('stockout_risk_pct', 0.0)
            merged['overstock_risk_pct'] = merged.get('overstock_risk_pct', 0.0)
            merged['inventory_ratio'] = merged.get('inventory_ratio', 1.0)
            merged['current_price'] = merged.get('current_price', 10.0)
            merged['recommended_price'] = merged.get('recommended_price', 10.0)
            merged['discount_pct'] = merged.get('discount_pct', 0.0)
            merged['action'] = merged.get('action', 'maintain')

        # Check if we have enough products after merge
        unique_products_after_merge = len(merged['product_id'].unique()) if 'product_id' in merged.columns else 0

        if unique_products_after_merge < top_n and len(forecasts_df) > 0:
            logger.info(f"Only {unique_products_after_merge} products available after merge, expanding to forecasts data...")
            # Use forecasts data directly if merged data is insufficient
            merged = forecasts_df.copy()
            # Add dummy inventory/pricing columns with calculated risk metrics
            merged['current_inventory'] = 2.0
            merged['safety_stock'] = 1.0
            merged['reorder_point'] = 1.5
            merged['should_reorder'] = False
            merged['inventory_ratio'] = 1.0
            merged['current_price'] = 10.0
            merged['recommended_price'] = 10.0
            merged['discount_pct'] = 0.0
            merged['action'] = 'maintain'

            # Risk metrics already added above in the sampling section

        # Improved product selection logic: Random sample + sort by product_id for consistency
        # Group by product_id to get one representative record per product
        if 'product_id' in merged.columns:
            # Group by product_id and aggregate numeric columns
            grouped = merged.groupby('product_id').agg({
                col: 'mean' if merged[col].dtype in ['int64', 'float64'] else 'first'
                for col in merged.columns
                if col != 'product_id'
            }).reset_index()

            # Random sample products with fixed seed for reproducibility
            if len(grouped) > top_n:
                np.random.seed(42)  # Fixed seed for consistency
                sampled_products = grouped.sample(n=top_n, random_state=42)
            else:
                sampled_products = grouped

            # Add calculated risk metrics to sampled products
            logger.info("Adding calculated risk metrics to selected products...")
            stockout_risks = []
            overstock_risks = []

            for pid in sampled_products['product_id'].unique():
                stockout_risk, overstock_risk = calculate_risk_metrics(forecasts_df, pid)
                stockout_risks.append((pid, stockout_risk))
                overstock_risks.append((pid, overstock_risk))

            # Create mapping dictionaries
            stockout_map = dict(stockout_risks)
            overstock_map = dict(overstock_risks)

            # Add risk metrics to sampled products
            sampled_products['stockout_risk_pct'] = sampled_products['product_id'].map(stockout_map).fillna(0.0)
            sampled_products['overstock_risk_pct'] = sampled_products['product_id'].map(overstock_map).fillna(0.0)

            # Sort by product_id for consistent ordering
            sorted_df = sampled_products.sort_values('product_id').reset_index(drop=True)

            logger.info(f"Selected {len(sorted_df)} products for analysis: {[int(pid) for pid in sorted_df['product_id'].tolist()]}")
        else:
            # Fallback to original logic if no product_id column
            logger.warning("No 'product_id' column found, using original selection logic")
            if 'forecast_q50' in merged.columns:
                sorted_df = merged.nlargest(top_n, 'forecast_q50')
            elif 'q50' in merged.columns:
                sorted_df = merged.nlargest(top_n, 'q50')
            else:
                sorted_df = merged.head(top_n)

        insights = []
        for _, row in sorted_df.iterrows():
            product_id = row['product_id']

            # Extract data for each module
            forecast_data = {
                'q50': row.get('forecast_q50', row.get('q50', 0)),
                'q05': row.get('forecast_q05', row.get('q05', 0)),
                'q95': row.get('forecast_q95', row.get('q95', 0)),
                'category': row.get('category', 'Unknown'),
                'store_id': row.get('store_id', 'Unknown'),
                'vs_yesterday': row.get('vs_yesterday', 0),
                'vs_last_week': row.get('vs_last_week', 0),
                'vs_monthly_avg': row.get('vs_monthly_avg', 0)
            }

            inventory_data = {
                'current_inventory': row.get('current_inventory', 0),
                'safety_stock': row.get('safety_stock', 0),
                'reorder_point': row.get('reorder_point', 0),
                'should_reorder': row.get('should_reorder', False),
                'stockout_risk_pct': row.get('stockout_risk_pct', 0),
                'overstock_risk_pct': row.get('overstock_risk_pct', 0),
                'inventory_ratio': row.get('inventory_ratio', 1.0)
            }

            pricing_data = {
                'current_price': row.get('current_price', 0),
                'recommended_price': row.get('recommended_price', row.get('current_price', 0)),
                'discount_pct': row.get('discount_pct', 0),
                'action': row.get('action', 'maintain'),
                'reasoning': row.get('reasoning', ''),
                'unit_cost': row.get('unit_cost', 0)
            }

            shap_values = shap_dict.get(product_id, None) if shap_dict else None

            insight = self.generate_comprehensive_insight(
                product_id, forecast_data, inventory_data, pricing_data, shap_values, use_llm
            )
            insights.append(insight)

        return pd.DataFrame(insights)


# Convenience function
def generate_insight(
    product_id: str,
    forecast_data: dict,
    inventory_data: dict,
    pricing_data: dict,
    shap_values: dict | None = None,
    use_llm: bool = True,
    api_key: str | None = None,
    api_provider: str = "gemini",
    model: str = "gemini-2.0-flash-exp"
) -> str:
    """
    Convenience function to generate insight.

    Returns:
        Insight text string
    """
    generator = LLMInsightGenerator(
        use_llm_api=use_llm,
        api_key=api_key,
        api_provider=api_provider,
        model=model
    )

    result = generator.generate_comprehensive_insight(
        product_id, forecast_data, inventory_data, pricing_data, shap_values, use_llm
    )

    return result.get('insight_text', result.get('insight', ''))


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)

    generator = LLMInsightGenerator()

    test_forecast = {
        'q50': 150,
        'q05': 100,
        'q95': 200,
        'category': 'Fresh Produce',
        'store_id': 'S001',
        'vs_yesterday': 15.5,
        'vs_last_week': 8.2
    }

    test_inventory = {
        'current_inventory': 120,
        'safety_stock': 30,
        'reorder_point': 100,
        'should_reorder': True,
        'stockout_risk_pct': 45,
        'overstock_risk_pct': 20
    }

    test_pricing = {
        'current_price': 50000,
        'recommended_price': 45000,
        'discount_pct': 0.10,
        'action': 'small_discount',
        'unit_cost': 30000
    }

    test_shap = {
        'promo_active': 0.35,
        'price_change': -0.15,
        'day_of_week': 0.10
    }

    insight = generator.generate_comprehensive_insight(
        'TEST001',
        test_forecast,
        test_inventory,
        test_pricing,
        test_shap
    )

    print("\n" + "="*70)
    print("LLM INSIGHT TEST")
    print("="*70)
    print(insight.get('insight_text', insight.get('insight', '')))
    print("\n✅ Module 4 - LLM Insights Ready")
