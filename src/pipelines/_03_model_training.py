"""
Unified Model Training Pipeline (Config-Driven)
=================================================
Trains quantile models based on the active dataset config.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_pinball_loss
import warnings
import time
import sys
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any
import argparse

# Setup project path for imports
try:
    from src.config import (
        setup_project_path, setup_logging, ensure_directories,
        OUTPUT_FILES, TRAINING_CONFIG, ALL_FEATURES_CONFIG,
        get_model_config, get_dataset_config
    )
    setup_project_path()
    setup_logging()
    ensure_directories()
    logger = logging.getLogger(__name__)
except ImportError:
    print("Error: config.py not found. Please ensure src/config.py exists.")
    sys.exit(1)

warnings.filterwarnings('ignore')

def load_data(filepath: Path) -> pd.DataFrame:
    """Loads the clean feature table."""
    logger.info(f"Loading data from: {filepath}...")
    try:
        df = pd.read_parquet(filepath)
        logger.info(f"Load complete. Shape: {df.shape}.")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}. Run _02_feature_enrichment.py first.")
        sys.exit(1)

def get_features_from_config(config: Dict) -> Tuple[List[str], List[str]]:
    """
    Builds the feature lists dynamically based on dataset config toggles.
    """
    logger.info("Building feature list from config toggles...")
    all_features = []
    
    # Base timeseries features (luôn có)
    all_features.extend(ALL_FEATURES_CONFIG['timeseries_base'])
    
    # Các feature tùy chọn
    if config['has_relational']:
        all_features.extend(ALL_FEATURES_CONFIG['relational'])
    if config['has_intraday_patterns']:
        all_features.extend(ALL_FEATURES_CONFIG['intraday_patterns'])
    if config['has_behavior']:
        all_features.extend(ALL_FEATURES_CONFIG['behavior'])
    if config['has_price_promo']:
        all_features.extend(ALL_FEATURES_CONFIG['price_promo'])
    if config['has_stockout']:
        all_features.extend(ALL_FEATURES_CONFIG['stockout'])
    if config['has_weather']:
        all_features.extend(ALL_FEATURES_CONFIG['weather'])

    # Lấy danh sách categorical features từ config
    all_categorical = []
    for ws_features in ALL_FEATURES_CONFIG.values():
        all_categorical.extend(ws_features) # Tạm lấy tất cả
        
    # Lọc features thực sự tồn tại trong DataFrame
    logger.info("Finding categorical features...")
    
    # Tạo một DataFrame mẫu để kiểm tra kiểu dữ liệu (để tìm categorical)
    # Đây là một cách đơn giản, chúng ta sẽ cải thiện nó trong prepare_data
    temp_numeric = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Giả định: Bất kỳ feature nào KHÔNG phải là số đều là categorical
    # Chúng ta sẽ làm sạch danh sách này trong prepare_data
    
    return all_features, all_categorical


def prepare_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Prepares data for modeling: selects features and performs time-based split.
    """
    logger.info("Preparing data for modeling (config-driven)...")

    # 1. Get Target and Time columns from config
    target_col = config['target_column']
    time_col_name = config['time_column']
    
    logger.info(f"Target (Y): {target_col}")
    logger.info(f"Time Column: {time_col_name}")

    if target_col not in df.columns or time_col_name not in df.columns:
        logger.error(f"FATAL: Target '{target_col}' or Time '{time_col_name}' not in DataFrame.")
        sys.exit(1)

    df_model = df.dropna(subset=[target_col]).copy()

    # 2. Build Feature List
    all_features_config, all_categorical_config = get_features_from_config(config)
    
    # Lọc: chỉ giữ lại các features có trong df
    all_features = [col for col in all_features_config if col in df.columns]
    
    # Xác định chính xác categorical features
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    categorical_features = []
    for col in all_features:
        if df_model[col].dtype.name not in numeric_dtypes:
             categorical_features.append(col)
             
    # Thêm các cột số nhưng thực ra là categorical (ví dụ: hour_of_day)
    manual_cats = ['hour_of_day', 'day_of_week', 'is_morning_peak', 'is_evening_peak', 
                   'is_weekend', 'temp_category', 'is_rainy', 'rain_intensity', 
                   'is_extreme_heat', 'is_extreme_cold', 'is_high_humidity',
                   'is_on_display', 'is_on_mailer', 'is_on_retail_promo', 'is_on_coupon_promo']
                   
    for col in manual_cats:
        if col in all_features and col not in categorical_features:
            categorical_features.append(col)

    logger.info(f"Found {len(all_features)} total features.")
    logger.info(f"Found {len(categorical_features)} categorical features: {categorical_features}")

    X = df_model[all_features]
    y = df_model[target_col]
    
    # CHỐT CHẶN CUỐI: Fill tất cả NaN còn sót lại
    # --------------------------------------------------
    numeric_features = X.select_dtypes(include=[np.number]).columns
    X[numeric_features] = X[numeric_features].fillna(0)
    
    logger.info(f"Final safeguard: Filled NaNs in {len(numeric_features)} numeric columns with 0.")
    # --------------------------------------------------
    
    # Chuyển đổi categorical features sang 'category' dtype
    for col in categorical_features:
        X[col] = X[col].astype('category')
        # Fill 'Unknown' cho categorical
        if X[col].isnull().any():
            X[col] = X[col].cat.add_categories(['Unknown']).fillna('Unknown')
    
    logger.info("Final safeguard: Filled NaNs in categorical columns with 'Unknown'.")

    # 3. Time-based split (NO SHUFFLE!)
    logger.info("=" * 70)
    logger.info("PERFORMING TIME-BASED SPLIT (Leak-Safe)")
    
    time_col_data = pd.to_datetime(df_model[time_col_name])
    cutoff_percentile = TRAINING_CONFIG['train_test_split']['cutoff_percentile']
    cutoff_time = time_col_data.quantile(cutoff_percentile)
    
    logger.info(f"Time cutoff: < {cutoff_time} = TRAIN, >= {cutoff_time} = TEST")
    
    train_mask = time_col_data < cutoff_time
    test_mask = time_col_data >= cutoff_time
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    logger.info(f"Train set: {len(X_train):,} samples (up to {time_col_data[train_mask].max()})")
    logger.info(f"Test set:  {len(X_test):,} samples (from {time_col_data[test_mask].min()})")
    logger.info(f"Split ratio: {len(X_train)/len(X)*100:.1f}% train / {len(X_test)/len(X)*100:.1f}% test")
    logger.info("=" * 70)

    return X_train, X_test, y_train, y_test, all_features, categorical_features

def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str]
) -> Dict[float, lgb.LGBMRegressor]:
    """
    Trains separate LightGBM models for each quantile.
    """
    quantiles = TRAINING_CONFIG['quantiles']
    logger.info("=" * 70)
    logger.info(f"TRAINING {len(quantiles)} QUANTILE MODELS: {quantiles}")
    logger.info(f"Using {TRAINING_CONFIG['hyperparameters']['n_jobs']} threads.")
    logger.info("=" * 70)
    start_train = time.time()
    
    models = {}
    
    for alpha in quantiles:
        logger.info(f"Training Q{int(alpha*100):02d} model (alpha={alpha})...")
        
        model_params = get_model_config(alpha)
        model = lgb.LGBMRegressor(**model_params)
        
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_features
        )
        models[alpha] = model
    
    logger.info(f"All quantile models trained (Took {time.time() - start_train:.2f}s)")
    return models

def evaluate_quantile_models(
    models: Dict[float, lgb.LGBMRegressor],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluates models using pinball loss."""
    logger.info("=" * 70)
    logger.info("EVALUATING QUANTILE MODELS (Pinball Loss)")
    logger.info("=" * 70)
    
    metrics = {}
    predictions = {}
    quantiles = TRAINING_CONFIG['quantiles']
    
    for alpha in quantiles:
        model = models[alpha]
        y_pred = model.predict(X_test)
        y_pred[y_pred < 0] = 0  # Clip negative predictions
        predictions[alpha] = y_pred
        
        pinball = mean_pinball_loss(y_test, y_pred, alpha=alpha)
        metrics[f'q{int(alpha*100):02d}_pinball_loss'] = pinball
        logger.info(f"  Q{int(alpha*100):02d} Pinball Loss: {pinball:.4f}")

    # Calculate coverage
    lower_q = min(quantiles)
    upper_q = max(quantiles)
    lower_pred = predictions[lower_q]
    upper_pred = predictions[upper_q]
    coverage = ((y_test >= lower_pred) & (y_test <= upper_pred)).mean()
    metrics[f'coverage_{(upper_q-lower_q)*100:.0f}%'] = coverage
    logger.info(f"Coverage ({ (upper_q-lower_q)*100:.0f}%): {coverage*100:.2f}%")
    
    return metrics

def save_artifacts(
    models: Dict[float, lgb.LGBMRegressor],
    features_config: Dict,
    metrics: Dict
) -> None:
    """Saves models, features, and metrics."""
    logger.info("Saving model artifacts...")

    # Save models
    for alpha, model in models.items():
        model_key = f'model_q{int(alpha*100):02d}'
        # Tạo tên tệp động nếu không có trong config
        model_path = OUTPUT_FILES.get(model_key, 
            OUTPUT_FILES['models_dir'] / f'q{int(alpha*100):02d}_forecaster.joblib')
        joblib.dump(model, model_path)
        logger.info(f"  {model_key} saved to: {model_path}")

    # Save feature config
    features_path = OUTPUT_FILES['model_features']
    with open(features_path, 'w') as f:
        json.dump(features_config, f, indent=4)
    logger.info(f"  Feature config saved to: {features_path}")

    # Save metrics
    metrics_path = OUTPUT_FILES['model_metrics']
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"  Metrics saved to: {metrics_path}")

def main(args) -> None:
    """Orchestrates the entire training pipeline."""
    logger.info("=" * 70)
    logger.info("STARTING MODEL TRAINING PIPELINE (Config-Driven)")
    logger.info("=" * 70)
    total_start_time = time.time()
    
    # 1. Lấy config
    config = get_dataset_config()
    logger.info(f"Active Dataset: {config['name']}")

    # 2. Load Data
    logger.info("STEP 1: LOAD DATA")
    df = load_data(OUTPUT_FILES['master_feature_table'])

    # 3. Prepare Data
    logger.info("STEP 2: PREPARE DATA & TIME-BASED SPLIT")
    X_train, X_test, y_train, y_test, features, cat_features = prepare_data(df, config)

    # 4. Train Models (Tuning-aware)
    logger.info("STEP 3: TRAIN QUANTILE MODELS")
    # (Logic tuning của bạn từ file gốc có thể được thêm vào đây nếu muốn)
    if args.tune:
        logger.warning("Hyperparameter tuning not implemented in this refactor. Using standard params.")
        # Thêm logic gọi hàm train_quantile_models_tuned(....) ở đây
        
    quantile_models = train_quantile_models(X_train, y_train, cat_features)

    # 5. Evaluate
    logger.info("STEP 4: EVALUATE QUANTILE MODELS")
    metrics = evaluate_quantile_models(quantile_models, X_test, y_test)

    # 6. Save
    logger.info("STEP 5: SAVE ARTIFACTS")
    final_features_config = {
        "all_features": features,
        "categorical_features": cat_features,
        "quantiles": TRAINING_CONFIG['quantiles'],
        "model_type": "LightGBM_Quantile_Regression",
        "dataset_trained_on": config['name']
    }
    save_artifacts(quantile_models, final_features_config, metrics)

    logger.info("=" * 70)
    logger.info(f"COMPLETE! Total runtime: {time.time() - total_start_time:.2f} seconds.")
    logger.info(f"Artifacts saved to: {OUTPUT_FILES['models_dir']}")
    logger.info("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train quantile regression models')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning (nếu được implement)')
    args = parser.parse_args()
    
    main(args)