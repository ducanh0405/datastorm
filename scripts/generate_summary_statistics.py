#!/usr/bin/env python3
"""
Generate Summary Statistics for Report
======================================
Calculate comprehensive statistics from all modules.

Outputs:
- Model performance summary
- Business impact summary  
- Inventory optimization summary
- Pricing optimization summary

Author: SmartGrocy Team
Date: 2025-11-18
"""

import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)


def to_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def convert_dict_values_to_serializable(data):
    """Recursively convert all values in a dict to JSON serializable types."""
    if isinstance(data, dict):
        return {k: convert_dict_values_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_dict_values_to_serializable(item) for item in data]
    else:
        return to_serializable(data)


def calculate_model_summary() -> Dict:
    """Calculate model performance summary."""
    
    metrics_file = Path('reports/metrics/model_metrics.json')
    
    if not metrics_file.exists():
        logger.warning("Model metrics not found")
        return {}
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract key metrics
    q50_metrics = metrics.get('q50', {})
    
    summary = {
        'mae_q50': q50_metrics.get('mae', 0),
        'rmse_q50': q50_metrics.get('rmse', 0),
        'r2_score': q50_metrics.get('r2_score', 0),
        'forecast_accuracy_pct': (1 - q50_metrics.get('mae', 1)) * 100
    }
    
    return summary


def calculate_inventory_summary() -> Dict:
    """Calculate inventory optimization summary."""
    
    inv_file = Path('reports/inventory_recommendations.csv')
    
    if not inv_file.exists():
        logger.warning("Inventory recommendations not found")
        return {}
    
    df = pd.read_csv(inv_file)
    
    summary = {
        'total_products': len(df),
        'products_need_reorder': df['should_reorder'].sum() if 'should_reorder' in df.columns else 0,
        'avg_stockout_risk': df['stockout_risk_pct'].mean() if 'stockout_risk_pct' in df.columns else 0,
        'avg_safety_stock': df['safety_stock'].mean() if 'safety_stock' in df.columns else 0,
        'high_risk_products': (df['stockout_risk_pct'] > 50).sum() if 'stockout_risk_pct' in df.columns else 0
    }
    
    return summary


def calculate_pricing_summary() -> Dict:
    """Calculate pricing optimization summary."""
    
    price_file = Path('reports/pricing_recommendations.csv')
    
    if not price_file.exists():
        logger.warning("Pricing recommendations not found")
        return {}
    
    df = pd.read_csv(price_file)
    
    summary = {
        'total_products': len(df),
        'products_with_discount': (df['discount_pct'] > 0).sum() if 'discount_pct' in df.columns else 0,
        'avg_discount_pct': df['discount_pct'].mean() * 100 if 'discount_pct' in df.columns else 0,
        'total_revenue_impact': df['expected_revenue_impact'].sum() if 'expected_revenue_impact' in df.columns else 0
    }
    
    return summary


def main():
    """Generate all summary statistics."""
    
    logger.info("\n" + "="*70)
    logger.info("GENERATING SUMMARY STATISTICS")
    logger.info("="*70 + "\n")
    
    summaries = {
        'model_performance': calculate_model_summary(),
        'inventory_optimization': calculate_inventory_summary(),
        'pricing_optimization': calculate_pricing_summary(),
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    # Calculate aggregate metrics
    summaries['aggregate'] = {
        'total_products_analyzed': summaries['inventory_optimization'].get('total_products', 0),
        'forecast_accuracy': summaries['model_performance'].get('forecast_accuracy_pct', 0),
        'reorder_recommendations': summaries['inventory_optimization'].get('products_need_reorder', 0),
        'pricing_adjustments': summaries['pricing_optimization'].get('products_with_discount', 0)
    }
    
    # Save
    output_file = Path('reports/summary_statistics.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON serializable types
    serializable_summaries = convert_dict_values_to_serializable(summaries)

    with open(output_file, 'w') as f:
        json.dump(serializable_summaries, f, indent=2)
    
    logger.info(f"\n[SUCCESS] Summary statistics saved: {output_file}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    for key, value in summaries['aggregate'].items():
        logger.info(f"  {key}: {value}")
    
    return summaries


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
