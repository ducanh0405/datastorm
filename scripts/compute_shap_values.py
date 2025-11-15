#!/usr/bin/env python3
"""
SHAP Values Computation Script
===============================
Tính toán và phân tích SHAP values cho model interpretability.

Usage:
    python scripts/compute_shap_values.py
    python scripts/compute_shap_values.py --sample-size 2000
    python scripts/compute_shap_values.py --quantile 0.50 --model-type lightgbm
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    OUTPUT_FILES, SHAP_CONFIG, TRAINING_CONFIG,
    get_dataset_config, setup_project_path, setup_logging
)
from src.pipelines._05_prediction import QuantileForecaster

# Setup
setup_project_path()
setup_logging()
logger = logging.getLogger(__name__)

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.error("SHAP or matplotlib not available. Install with: pip install shap matplotlib")
    sys.exit(1)


def compute_shap_values(
    sample_size: int = None,
    quantile: float = 0.50,
    model_type: str = 'lightgbm'
) -> tuple:
    """
    Tính toán SHAP values cho model.
    
    Args:
        sample_size: Số lượng samples để tính SHAP
        quantile: Quantile của model (0.50 = median)
        model_type: Loại model ('lightgbm', 'catboost', 'random_forest')
    
    Returns:
        Tuple của (X_sample DataFrame, shap_summary dict)
    """
    logger.info("="*70)
    logger.info("COMPUTING SHAP VALUES")
    logger.info("="*70)
    
    # Load data
    logger.info("Loading master feature table...")
    df = pd.read_parquet(OUTPUT_FILES['master_feature_table'])
    config = get_dataset_config()
    time_col = config['time_column']
    
    # Get test set (same as prediction pipeline)
    time_col_data = pd.to_datetime(df[time_col])
    cutoff_percentile = TRAINING_CONFIG['train_test_split']['cutoff_percentile']
    cutoff_time = time_col_data.quantile(cutoff_percentile)
    test_mask = time_col_data >= cutoff_time
    df_test = df[test_mask].copy()
    
    logger.info(f"Test set size: {len(df_test):,} rows")
    
    # Initialize forecaster
    logger.info("Loading models...")
    forecaster = QuantileForecaster()
    forecaster.load_models()
    
    # Compute SHAP values
    logger.info(f"Computing SHAP values for {model_type} Q{int(quantile*100):02d}...")
    sample_size = sample_size or SHAP_CONFIG.get('sample_size', 1000)
    
    try:
        X_sample, shap_summary = forecaster.predict_shap(
            df_test,
            model_type=model_type,
            sample_size=sample_size,
            quantile=quantile
        )
        
        # Add X_sample to shap_summary for visualizations
        shap_summary['X_sample'] = X_sample
        
        logger.info("✓ SHAP values computed successfully")
        logger.info(f"  - Samples: {len(shap_summary['shap_values']):,}")
        logger.info(f"  - Features: {len(shap_summary['feature_names'])}")
        logger.info(f"  - Base value: {shap_summary['base_value']:.4f}")
        
        return X_sample, shap_summary
        
    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}", exc_info=True)
        raise


def analyze_feature_importance(shap_summary: dict) -> pd.DataFrame:
    """
    Phân tích feature importance từ SHAP values.
    
    Args:
        shap_summary: Dict chứa SHAP values từ compute_shap_values()
    
    Returns:
        DataFrame với feature importance metrics
    """
    shap_df = shap_summary['shap_values']
    
    # Tính mean absolute SHAP values (feature importance)
    importance = shap_df.abs().mean().sort_values(ascending=False)
    
    # Tính các metrics khác
    results = []
    for feature in importance.index:
        shap_values = shap_df[feature]
        results.append({
            'feature': feature,
            'mean_abs_shap': importance[feature],
            'mean_shap': shap_values.mean(),
            'std_shap': shap_values.std(),
            'min_shap': shap_values.min(),
            'max_shap': shap_values.max(),
            'positive_impact_pct': (shap_values > 0).sum() / len(shap_values) * 100,
            'negative_impact_pct': (shap_values < 0).sum() / len(shap_values) * 100,
        })
    
    importance_df = pd.DataFrame(results)
    return importance_df


def create_shap_visualizations(
    shap_summary: dict,
    forecaster: QuantileForecaster,
    output_dir: Path,
    model_type: str = 'lightgbm',
    quantile: float = 0.50
):
    """
    Tạo các visualization cho SHAP values.
    
    Args:
        shap_summary: Dict chứa SHAP values
        forecaster: QuantileForecaster instance với models đã load
        output_dir: Thư mục để lưu plots
        model_type: Model type
        quantile: Quantile level
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shap_df = shap_summary['shap_values']
    X_sample = shap_summary.get('X_sample')
    
    try:
        # Get model
        model = forecaster.models[model_type][quantile]
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values_array = shap_df.values
        
        # Summary plot (beeswarm)
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values_array,
            X_sample if X_sample is not None else shap_df,
            max_display=SHAP_CONFIG.get('max_display_features', 20),
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_summary_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved SHAP summary plot (beeswarm)")
        
        # Bar plot (mean absolute SHAP)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_array,
            X_sample if X_sample is not None else shap_df,
            plot_type="bar",
            max_display=SHAP_CONFIG.get('max_display_features', 20),
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_bar_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("✓ Saved SHAP bar plot")
        
    except Exception as e:
        logger.warning(f"Could not create SHAP plots: {e}")
        logger.debug(f"Plot error details: {e}", exc_info=True)


def save_shap_results(shap_summary: dict, importance_df: pd.DataFrame):
    """
    Lưu SHAP values và analysis results.
    
    Args:
        shap_summary: Dict chứa SHAP values
        importance_df: DataFrame với feature importance
    """
    shap_dir = OUTPUT_FILES['shap_values_dir']
    shap_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SHAP values DataFrame
    shap_df_path = shap_dir / 'shap_values.csv'
    shap_summary['shap_values'].to_csv(shap_df_path)
    logger.info(f"✓ SHAP values saved to: {shap_df_path}")
    
    # Save feature importance
    importance_path = shap_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"✓ Feature importance saved to: {importance_path}")
    
    # Save summary JSON
    summary_path = shap_dir / 'shap_summary.json'
    summary_to_save = {
        'base_value': float(shap_summary['base_value']),
        'feature_names': shap_summary['feature_names'],
        'model_type': shap_summary['model_type'],
        'quantile': shap_summary['quantile'],
        'sample_size': len(shap_summary['shap_values']),
        'top_features': importance_df.head(10).to_dict('records'),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_to_save, f, indent=4)
    logger.info(f"✓ SHAP summary saved to: {summary_path}")


def print_summary(importance_df: pd.DataFrame, shap_summary: dict):
    """In summary của SHAP analysis."""
    print("\n" + "="*70)
    print("SHAP VALUES ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nModel: {shap_summary['model_type']} Q{int(shap_summary['quantile']*100):02d}")
    print(f"Samples analyzed: {len(shap_summary['shap_values']):,}")
    print(f"Base value: {shap_summary['base_value']:.4f}")
    print(f"\nTop 15 Most Important Features:")
    print("-"*70)
    print(f"{'Rank':<6} {'Feature':<35} {'Mean |SHAP|':<15} {'Impact':<15}")
    print("-"*70)
    
    for idx, row in importance_df.head(15).iterrows():
        impact = "Positive" if row['mean_shap'] > 0 else "Negative"
        print(f"{idx+1:<6} {row['feature']:<35} {row['mean_abs_shap']:<15.4f} {impact:<15}")
    
    print("\n" + "="*70)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute and analyze SHAP values')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of samples for SHAP calculation (default: from config)')
    parser.add_argument('--quantile', type=float, default=0.50,
                       help='Quantile to analyze (default: 0.50 = median)')
    parser.add_argument('--model-type', type=str, default='lightgbm',
                       choices=['lightgbm', 'catboost', 'random_forest'],
                       help='Model type to analyze (default: lightgbm)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating visualization plots')
    
    args = parser.parse_args()
    
    if not SHAP_AVAILABLE:
        logger.error("SHAP library not available. Install with: pip install shap matplotlib")
        sys.exit(1)
    
    try:
        # Initialize forecaster (needed for visualizations)
        forecaster = QuantileForecaster()
        forecaster.load_models()
        
        # Compute SHAP values
        X_sample, shap_summary = compute_shap_values(
            sample_size=args.sample_size,
            quantile=args.quantile,
            model_type=args.model_type
        )
        
        # Analyze feature importance
        logger.info("\nAnalyzing feature importance...")
        importance_df = analyze_feature_importance(shap_summary)
        
        # Save results
        save_shap_results(shap_summary, importance_df)
        
        # Create visualizations
        if not args.no_plots:
            logger.info("\nCreating visualizations...")
            # Re-initialize forecaster for visualizations
            forecaster_viz = QuantileForecaster()
            forecaster_viz.load_models()
            create_shap_visualizations(
                shap_summary,
                forecaster_viz,
                OUTPUT_FILES['shap_values_dir'],
                model_type=args.model_type,
                quantile=args.quantile
            )
        
        # Print summary
        print_summary(importance_df, shap_summary)
        
        logger.info("\n✅ SHAP analysis complete!")
        logger.info(f"Results saved to: {OUTPUT_FILES['shap_values_dir']}")
        
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

