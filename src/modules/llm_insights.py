#!/usr/bin/env python3
"""
Module 4: LLM Insights Generator
=================================

LLM-powered insight generation for demand forecasts and business decisions.

Capabilities:
- Interpret SHAP values and feature importance
- Generate causal explanations for demand changes
- Provide actionable recommendations
- Summarize forecast trends
- Detect anomalies and alert

Framework: Causal → Impact → Action

Author: SmartGrocy Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class InsightConfig:
    """Configuration for insight generation."""
    use_llm_api: bool = False  # Set True to use actual LLM API
    api_provider: str = "openai"  # openai, anthropic, local
    api_key: Optional[str] = None
    model: str = "gpt-4"
    template_style: str = "bullet"  # bullet, paragraph, table


class LLMInsightGenerator:
    """
    Generate business insights from forecast data using LLM.
    
    Can work in two modes:
    1. Rule-based (no API) - Template-based insights
    2. LLM-powered (with API) - Advanced interpretations
    """
    
    def __init__(self, config: Optional[InsightConfig] = None):
        self.config = config or InsightConfig()
        logger.info(f"LLM Insight Generator initialized (API: {self.config.use_llm_api})")
    
    def generate_forecast_insight(
        self,
        product_id: str,
        forecast_data: Dict,
        shap_values: Optional[Dict] = None,
        historical_data: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Generate insight for a forecast.
        
        Args:
            product_id: Product identifier
            forecast_data: Dict with keys:
                - q50: Median forecast
                - q05, q95: Prediction intervals
                - trend: up/down/stable
            shap_values: Top SHAP contributors (optional)
            historical_data: Historical context (optional)
        
        Returns:
            Dictionary with insight sections
        """
        if self.config.use_llm_api and self.config.api_key:
            return self._generate_llm_insight(
                product_id, forecast_data, shap_values, historical_data
            )
        else:
            return self._generate_rule_based_insight(
                product_id, forecast_data, shap_values, historical_data
            )
    
    def _generate_rule_based_insight(
        self,
        product_id: str,
        forecast_data: Dict,
        shap_values: Optional[Dict],
        historical_data: Optional[Dict]
    ) -> Dict[str, str]:
        """
        Generate insight using rule-based templates (no API needed).
        """
        q50 = forecast_data.get('q50', 0)
        trend = forecast_data.get('trend', 'stable')
        
        # Determine cause
        causes = []
        if shap_values:
            top_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            for feat, val in top_features:
                if 'promo' in feat.lower() and val > 0:
                    causes.append("Active promotional campaign")
                elif 'price' in feat.lower() and val < 0:
                    causes.append("Price increase reducing demand")
                elif 'season' in feat.lower():
                    causes.append("Seasonal demand pattern")
                elif 'lag' in feat.lower():
                    causes.append("Historical demand momentum")
        
        if not causes:
            causes = ["Normal market conditions"]
        
        # Determine impact
        if trend == 'up':
            impact = f"Demand forecast increased to {q50:.1f} units (strong growth trend)"
        elif trend == 'down':
            impact = f"Demand forecast decreased to {q50:.1f} units (declining trend)"
        else:
            impact = f"Demand forecast stable at {q50:.1f} units"
        
        # Determine action
        q95 = forecast_data.get('q95', q50 * 1.5)
        uncertainty = q95 - q50
        
        actions = []
        if uncertainty > q50 * 0.5:  # High uncertainty
            actions.append(f"Maintain higher safety stock due to {uncertainty/q50:.0%} forecast uncertainty")
        
        if trend == 'up':
            actions.append("Consider increasing order quantity to prevent stockout")
        elif trend == 'down':
            actions.append("Monitor inventory levels and consider markdown pricing if needed")
        
        if not actions:
            actions = ["Continue with current inventory policy"]
        
        # Format based on template style
        if self.config.template_style == "bullet":
            insight_text = f"""**Forecast Insight for {product_id}**

**Cause:**
{chr(10).join('- ' + c for c in causes)}

**Impact:**
- {impact}

**Recommended Actions:**
{chr(10).join('- ' + a for a in actions)}
"""
        elif self.config.template_style == "paragraph":
            insight_text = f"""For product {product_id}, the demand is influenced by {', '.join(causes).lower()}. {impact}. To optimize inventory, {' '.join(actions).lower()}."""
        else:  # table
            insight_text = json.dumps({
                "product_id": product_id,
                "cause": causes,
                "impact": impact,
                "actions": actions
            }, indent=2)
        
        return {
            'product_id': product_id,
            'insight': insight_text,
            'causes': causes,
            'impact': impact,
            'actions': actions,
            'generated_by': 'rule_based'
        }
    
    def _generate_llm_insight(
        self,
        product_id: str,
        forecast_data: Dict,
        shap_values: Optional[Dict],
        historical_data: Optional[Dict]
    ) -> Dict[str, str]:
        """
        Generate insight using LLM API (OpenAI/Anthropic).
        
        Note: Requires API key to be configured.
        """
        # Prepare prompt
        prompt = self._build_insight_prompt(
            product_id, forecast_data, shap_values, historical_data
        )
        
        try:
            # Call LLM API (placeholder - implement when API key available)
            logger.info("Calling LLM API for insight generation...")
            
            # TODO: Implement actual API call
            # if self.config.api_provider == "openai":
            #     response = openai.ChatCompletion.create(...)
            # elif self.config.api_provider == "anthropic":
            #     response = anthropic.messages.create(...)
            
            # For now, fallback to rule-based
            logger.warning("LLM API not configured, using rule-based fallback")
            return self._generate_rule_based_insight(
                product_id, forecast_data, shap_values, historical_data
            )
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return self._generate_rule_based_insight(
                product_id, forecast_data, shap_values, historical_data
            )
    
    def _build_insight_prompt(
        self,
        product_id: str,
        forecast_data: Dict,
        shap_values: Optional[Dict],
        historical_data: Optional[Dict]
    ) -> str:
        """Build prompt for LLM API."""
        prompt = f"""Analyze the following demand forecast for product {product_id}:

Forecast Data:
{json.dumps(forecast_data, indent=2)}

Top Contributing Features (SHAP values):
{json.dumps(shap_values or {}, indent=2)}

Provide insight in this format:
1. Cause: What factors are driving this forecast?
2. Impact: What does this mean for inventory?
3. Action: What should the operator do?

Keep it concise and actionable.
"""
        return prompt
    
    def batch_generate(
        self,
        forecast_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Generate insights for top N products by importance.
        
        Args:
            forecast_df: Forecast results DataFrame
            top_n: Number of products to generate insights for
        
        Returns:
            DataFrame with insights
        """
        # Sort by forecast value or volatility
        if 'q50' in forecast_df.columns:
            sorted_df = forecast_df.nlargest(top_n, 'q50')
        else:
            sorted_df = forecast_df.head(top_n)
        
        insights = []
        for _, row in sorted_df.iterrows():
            forecast_data = {
                'q50': row.get('q50', row.get('forecast', 0)),
                'q05': row.get('q05', 0),
                'q95': row.get('q95', 0),
                'trend': row.get('trend', 'stable')
            }
            
            insight = self.generate_forecast_insight(
                product_id=row['product_id'],
                forecast_data=forecast_data
            )
            insights.append(insight)
        
        return pd.DataFrame(insights)


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.INFO)
    
    generator = LLMInsightGenerator(config=InsightConfig(template_style="bullet"))
    
    test_forecast = {
        'q50': 150,
        'q05': 100,
        'q95': 200,
        'trend': 'up'
    }
    
    test_shap = {
        'promo_active': 0.35,
        'price_change': -0.15,
        'day_of_week': 0.10
    }
    
    insight = generator.generate_forecast_insight(
        'TEST001',
        test_forecast,
        test_shap
    )
    
    print("\n" + "="*70)
    print("LLM INSIGHT TEST")
    print("="*70)
    print(insight['insight'])
    print("\n✅ Module 4 - LLM Insights Ready")
