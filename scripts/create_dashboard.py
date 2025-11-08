#!/usr/bin/env python3
"""
Create Dashboard for Forecasting Results
========================================
Generates visualizations and summary reports for dashboard display.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import logging
from src.config import OUTPUT_FILES
from src.pipelines._05_prediction import QuantileForecaster, predict_on_test_set
from src.utils.visualization import create_dashboard_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate dashboard files."""
    logger.info("=" * 70)
    logger.info("CREATING DASHBOARD")
    logger.info("=" * 70)

    # 1. Generate predictions on test set
    logger.info("Step 1: Generating predictions...")
    predictions, metrics = predict_on_test_set()

    # 2. Create visualizations
    logger.info("Step 2: Creating visualizations...")
    dashboard_dir = OUTPUT_FILES['dashboard_dir']
    create_dashboard_summary(predictions, metrics, dashboard_dir)

    # 3. Create summary report
    logger.info("Step 3: Creating summary report...")
    summary = {
        'total_predictions': len(predictions),
        'metrics': metrics,
        'sample_predictions': predictions.head(100).to_dict('records')
    }

    import json
    with open(dashboard_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("DASHBOARD CREATED SUCCESSFULLY")
    logger.info(f"Files saved to: {dashboard_dir}")
    logger.info("=" * 70)
    logger.info("To view dashboard, open:")
    logger.info(f"  {dashboard_dir / 'index.html'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
