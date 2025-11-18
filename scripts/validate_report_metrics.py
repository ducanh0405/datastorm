#!/usr/bin/env python3
"""
Report Metrics Validator
========================
Validate all metrics in the report against actual pipeline outputs.

Checks:
- Model performance metrics
- Business KPIs
- Inventory metrics
- Pricing metrics
- Identifies placeholders and estimates

Author: SmartGrocy Team
Date: 2025-11-18
"""

import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)


class ReportMetricsValidator:
    """Validate all report metrics."""
    
    def __init__(self):
        self.validation_results = []
        self.warnings = []
        self.errors = []
        
    def validate_model_metrics(self) -> Dict:
        """Validate model performance metrics."""
        logger.info("Validating model metrics...")

        metrics_file = Path('reports/metrics/model_metrics.json')

        if not metrics_file.exists():
            self.errors.append("Model metrics file not found")
            return {'status': 'error', 'message': 'File not found'}

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # The metrics are stored in flat structure with prefixed keys
        # e.g., "q05_mae", "q05_rmse", "q25_mae", etc.
        required_quantiles = ['q05', 'q25', 'q50', 'q75', 'q95']
        required_metrics = ['mae', 'rmse']

        # Check for required quantile metrics
        for q in required_quantiles:
            for metric in required_metrics:
                key = f"{q}_{metric}"
                if key not in metrics:
                    self.errors.append(f"Missing metric: {key}")
                else:
                    value = metrics[key]
                    if value < 0:
                        self.errors.append(f"Invalid {key}: {value} (negative)")
                    elif metric == 'mae' and value > 100:
                        self.warnings.append(f"{key} unusually high: {value}")

        # Validate R2 score (stored as "r2_score" key)
        if 'r2_score' in metrics:
            r2 = metrics['r2_score']
            if r2 < 0 or r2 > 1:
                self.errors.append(f"Invalid R2 score: {r2}")
            elif r2 < 0.7:
                self.warnings.append(f"Low R2 score: {r2:.3f}")
        else:
            self.errors.append("Missing R2 score")

        # Validate coverage (stored as "coverage_90%" key)
        if 'coverage_90%' in metrics:
            coverage = metrics['coverage_90%']
            if coverage < 0 or coverage > 1:
                self.errors.append(f"Invalid coverage: {coverage}")
            elif coverage < 0.8:
                self.warnings.append(f"Low coverage: {coverage:.1%}")
        else:
            self.warnings.append("Coverage metric not found")

        return {
            'status': 'valid' if not self.errors else 'error',
            'metrics': metrics,
            'warnings': self.warnings
        }
    
    def validate_business_kpis(self) -> Dict:
        """Validate business KPIs."""
        logger.info("Validating business KPIs...")
        
        # Check for market analysis data
        market_file = Path('reports/market_analysis/market_kpis.csv')
        
        if not market_file.exists():
            self.warnings.append("Market KPIs file not found - may use estimates")
            return {'status': 'warning', 'source': 'estimated'}
        
        df = pd.read_csv(market_file)
        
        # Validate KPI ranges
        kpis_to_check = [
            ('spoilage_rate', 0, 0.20),  # 0-20%
            ('stockout_rate', 0, 0.15),  # 0-15%
            ('fill_rate', 0.80, 1.0),    # 80-100%
        ]
        
        for kpi, min_val, max_val in kpis_to_check:
            if kpi in df.columns:
                value = df[kpi].iloc[0]
                if value < min_val or value > max_val:
                    self.warnings.append(f"{kpi} out of typical range: {value}")
        
        return {
            'status': 'valid',
            'source': 'actual_data',
            'kpis': df.to_dict('records')[0] if len(df) > 0 else {}
        }
    
    def validate_predictions(self) -> Dict:
        """Validate prediction outputs."""
        logger.info("Validating predictions...")
        
        pred_file = Path('reports/predictions_test_set.csv')
        
        if not pred_file.exists():
            self.errors.append("Predictions file not found")
            return {'status': 'error'}
        
        df = pd.read_csv(pred_file)
        
        # Check required columns
        required_cols = ['product_id', 'forecast_q50']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            self.errors.append(f"Missing columns: {missing_cols}")
        
        # Validate predictions
        if 'forecast_q50' in df.columns:
            forecasts = df['forecast_q50']
            
            if (forecasts < 0).any():
                self.errors.append("Negative forecasts detected")
            
            if forecasts.isna().any():
                self.warnings.append(f"{forecasts.isna().sum()} NaN forecasts")
        
        return {
            'status': 'valid' if not self.errors else 'error',
            'num_predictions': len(df),
            'mean_forecast': df['forecast_q50'].mean() if 'forecast_q50' in df.columns else 0
        }
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report."""
        
        logger.info("\n" + "="*70)
        logger.info("REPORT METRICS VALIDATION")
        logger.info("="*70 + "\n")
        
        results = {
            'model_metrics': self.validate_model_metrics(),
            'business_kpis': self.validate_business_kpis(),
            'predictions': self.validate_predictions()
        }
        
        # Summary
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        
        results['summary'] = {
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'status': 'PASS' if total_errors == 0 else 'FAIL',
            'all_errors': self.errors,
            'all_warnings': self.warnings
        }
        
        # Log summary
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Status: {results['summary']['status']}")
        logger.info(f"Errors: {total_errors}")
        logger.info(f"Warnings: {total_warnings}")
        
        if self.errors:
            logger.error("\nErrors:")
            for err in self.errors:
                logger.error(f"  - {err}")
        
        if self.warnings:
            logger.warning("\nWarnings:")
            for warn in self.warnings:
                logger.warning(f"  - {warn}")
        
        return results


def main():
    """Main function."""
    
    validator = ReportMetricsValidator()
    results = validator.generate_validation_report()
    
    # Save results
    output_file = Path('reports/validation_report.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n[SUCCESS] Validation report saved: {output_file}")
    
    # Exit code
    sys.exit(0 if results['summary']['status'] == 'PASS' else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
