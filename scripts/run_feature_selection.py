#!/usr/bin/env python3
"""
Feature Selection Script
========================
Automatically select optimal features for model training.

Usage:
    python scripts/run_feature_selection.py
    python scripts/run_feature_selection.py --importance-threshold 0.005 --correlation-threshold 0.90
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config import DATA_DIRS, setup_logging
from src.features.feature_selection import get_optimal_features

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main(
    importance_threshold: float = 0.005,
    correlation_threshold: float = 0.95,
    max_features: Optional[int] = None,
    sample_size: Optional[int] = None
):
    """Run feature selection on master table data."""

    logger.info("=" * 60)
    logger.info("FEATURE SELECTION PIPELINE")
    logger.info("=" * 60)

    # Load master table
    master_table_path = DATA_DIRS['processed_data'] / 'master_feature_table.parquet'

    if not master_table_path.exists():
        logger.error(f"Master table not found: {master_table_path}")
        logger.info("Please run feature enrichment pipeline first")
        sys.exit(1)

    logger.info(f"Loading master table: {master_table_path}")

    try:
        df = pd.read_parquet(master_table_path)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to load master table: {e}")
        sys.exit(1)

    # Sample data if requested (for faster processing)
    if sample_size and len(df) > sample_size:
        logger.info(f"Sampling {sample_size:,} rows from {len(df):,} total rows")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Filter to rows with sales > 0 for training
    original_len = len(df)
    df = df[df['SALES_VALUE'] > 0].reset_index(drop=True)
    logger.info(f"Filtered to {len(df):,} rows with SALES_VALUE > 0 (from {original_len:,})")

    # Run feature selection
    logger.info("Starting feature selection...")
    result = get_optimal_features(
        df=df,
        target_col='SALES_VALUE',
        importance_threshold=importance_threshold,
        correlation_threshold=correlation_threshold,
        max_features=max_features,
        save_report=True
    )

    # Update model features configuration
    update_model_features_config(result['selected_features'])

    logger.info("=" * 60)
    logger.info("FEATURE SELECTION COMPLETED")
    logger.info(f"Selected {len(result['selected_features'])} features")
    logger.info("=" * 60)


def update_model_features_config(selected_features: list):
    """Update the model features configuration file."""

    try:
        import json
        from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

        # Separate numeric and categorical features
        all_config_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        selected_numeric = [f for f in selected_features if f in NUMERIC_FEATURES]
        selected_categorical = [f for f in selected_features if f in CATEGORICAL_FEATURES]

        # Create updated config
        updated_config = {
            "all_features": selected_features,
            "categorical_features": selected_categorical,
            "quantiles": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            "model_type": "LightGBM_Quantile_Regression",
            "hyperparameter_tuning": False,
            "optuna_available": True,
            "feature_selection": {
                "method": "importance_correlation",
                "n_selected": len(selected_features),
                "selection_timestamp": pd.Timestamp.now().isoformat()
            }
        }

        # Save updated config
        config_path = DATA_DIRS['models'] / 'model_features.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f, indent=2)

        logger.info(f"Updated model features config: {config_path}")
        logger.info(f"Selected {len(selected_features)} features ({len(selected_numeric)} numeric, {len(selected_categorical)} categorical)")

    except Exception as e:
        logger.warning(f"Failed to update model features config: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automatic feature selection")
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=0.005,
        help="Minimum importance score for feature selection (default: 0.005)"
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Correlation threshold for removing multicollinear features (default: 0.95)"
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to select (default: no limit)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for faster processing (default: use all data)"
    )

    args = parser.parse_args()

    main(
        importance_threshold=args.importance_threshold,
        correlation_threshold=args.correlation_threshold,
        max_features=args.max_features,
        sample_size=args.sample_size
    )
