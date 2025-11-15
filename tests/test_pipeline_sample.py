"""
Test Pipeline với Sample Data
==============================
Script test để chạy pipeline với sample data nhỏ.
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    setup_project_path, setup_logging, ensure_directories,
    OUTPUT_FILES, TRAINING_CONFIG, get_dataset_config, MODEL_TYPES
)
setup_project_path()
setup_logging(level='INFO')
ensure_directories()

logger = logging.getLogger(__name__)

def check_data_availability():
    """Kiểm tra xem data có sẵn không."""
    logger.info("=" * 70)
    logger.info("CHECKING DATA AVAILABILITY")
    logger.info("=" * 70)
    
    master_table_path = OUTPUT_FILES['master_feature_table']
    
    if not master_table_path.exists():
        logger.error(f"Master feature table not found: {master_table_path}")
        logger.info("Please run _02_feature_enrichment.py first to create master feature table.")
        return False
    
    logger.info(f"✓ Master feature table found: {master_table_path}")
    
    # Load và kiểm tra data
    try:
        df = pd.read_parquet(master_table_path)
        logger.info(f"✓ Data loaded. Shape: {df.shape}")
        logger.info(f"  Columns: {len(df.columns)}")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Kiểm tra target column
        config = get_dataset_config()
        target_col = config['target_column']
        time_col = config['time_column']
        
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in data")
            return False
        logger.info(f"✓ Target column '{target_col}' found")
        
        if time_col not in df.columns:
            logger.error(f"Time column '{time_col}' not found in data")
            return False
        logger.info(f"✓ Time column '{time_col}' found")
        
        # Kiểm tra số lượng samples
        logger.info(f"✓ Total samples: {len(df):,}")
        logger.info(f"✓ Non-null target values: {df[target_col].notna().sum():,}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return False


def test_training_with_sample():
    """Test training với sample data nhỏ."""
    logger.info("=" * 70)
    logger.info("TESTING TRAINING WITH SAMPLE DATA")
    logger.info("=" * 70)
    
    try:
        # Import training functions
        from src.pipelines._03_model_training import (
            load_data, prepare_data, train_quantile_models, evaluate_quantile_models, save_artifacts
        )
        
        # Load data
        logger.info("Loading data...")
        df = load_data(OUTPUT_FILES['master_feature_table'])
        
        # Sample data để test nhanh hơn (lấy 10% data)
        sample_size = min(10000, len(df) // 10)
        if len(df) > sample_size:
            logger.info(f"Sampling {sample_size} rows for quick test...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sample shape: {df.shape}")
        
        # Get config
        config = get_dataset_config()
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test, features, cat_features = prepare_data(df, config)
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        logger.info(f"Features: {len(features)}")
        logger.info(f"Categorical features: {len(cat_features)}")
        
        # Train models (chỉ LightGBM để test nhanh)
        logger.info(f"Training models: {MODEL_TYPES}")
        all_models = train_quantile_models(
            X_train, y_train, cat_features, model_types=MODEL_TYPES
        )
        
        # Evaluate
        logger.info("Evaluating models...")
        all_metrics = evaluate_quantile_models(all_models, X_test, y_test, cat_features)
        
        # Save artifacts
        logger.info("Saving artifacts...")
        final_features_config = {
            "all_features": features,
            "categorical_features": cat_features,
            "quantiles": TRAINING_CONFIG['quantiles'],
            "model_types": list(all_models.keys()),
            "dataset_trained_on": config['name']
        }
        save_artifacts(all_models, final_features_config, all_metrics)
        
        logger.info("=" * 70)
        logger.info("TRAINING TEST COMPLETE")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in training test: {e}", exc_info=True)
        return False


def test_prediction():
    """Test prediction với models đã train."""
    logger.info("=" * 70)
    logger.info("TESTING PREDICTION")
    logger.info("=" * 70)
    
    try:
        from src.pipelines._05_prediction import QuantileForecaster
        
        # Load data
        df = pd.read_parquet(OUTPUT_FILES['master_feature_table'])
        config = get_dataset_config()
        target_col = config['target_column']
        time_col = config['time_column']
        
        # Sample data
        sample_size = min(1000, len(df) // 20)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Split test set
        time_col_data = pd.to_datetime(df[time_col])
        cutoff_percentile = TRAINING_CONFIG['train_test_split']['cutoff_percentile']
        cutoff_time = time_col_data.quantile(cutoff_percentile)
        test_mask = time_col_data >= cutoff_time
        df_test = df[test_mask].copy()
        
        logger.info(f"Test set size: {len(df_test)}")
        
        # Initialize forecaster
        forecaster = QuantileForecaster()
        forecaster.load_models()
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = forecaster.predict(df_test)
        
        logger.info(f"✓ Predictions generated. Shape: {predictions.shape}")
        logger.info(f"  Prediction columns: {[col for col in predictions.columns if col.startswith('forecast_q')]}")
        
        # Test SHAP (optional, có thể bỏ qua nếu chậm)
        try:
            logger.info("Testing SHAP values...")
            results = forecaster.predict_with_shap(df_test, sample_size=100)
            logger.info("✓ SHAP values calculated")
        except Exception as e:
            logger.warning(f"SHAP test skipped: {e}")
        
        logger.info("=" * 70)
        logger.info("PREDICTION TEST COMPLETE")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in prediction test: {e}", exc_info=True)
        return False


def main():
    """Main test function."""
    logger.info("=" * 70)
    logger.info("STARTING PIPELINE TEST WITH SAMPLE DATA")
    logger.info("=" * 70)
    
    # 1. Check data availability
    if not check_data_availability():
        logger.error("Data not available. Please run feature enrichment pipeline first.")
        sys.exit(1)
    
    # 2. Test training
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: TEST TRAINING")
    logger.info("=" * 70)
    if not test_training_with_sample():
        logger.error("Training test failed.")
        sys.exit(1)
    
    # 3. Test prediction
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: TEST PREDICTION")
    logger.info("=" * 70)
    if not test_prediction():
        logger.error("Prediction test failed.")
        sys.exit(1)
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL TESTS PASSED")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

