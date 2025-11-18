"""
Unified Model Training Pipeline (Config-Driven)
=================================================
Trains quantile models based on the active dataset config.
"""
import logging
import sys
import warnings
from pathlib import Path

# Setup project path FIRST before any other imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import config
try:
    from src.config import (
        ALL_FEATURES_CONFIG,
        MODEL_CONFIGS,
        OUTPUT_FILES,
        TRAINING_CONFIG,
        ensure_directories,
        get_dataset_config,
        get_model_config,
        setup_logging,
        setup_project_path,
    )
    setup_project_path()
    setup_logging()
    ensure_directories()
    logger = logging.getLogger(__name__)
except ImportError as e:
    print("Error: Cannot import config. Please ensure src/config.py exists.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Import error: {e}")
    print(f"Python path: {sys.path[:3]}")
    sys.exit(1)

# Import other dependencies after config is loaded
import argparse
import json
import time
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error

# Import feature selection utilities for optimal feature subset selection
from src.features.feature_selection import get_optimal_features

# Import optional models
# Note: Logger is initialized above, so it's safe to use here
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    logger.debug("CatBoost is available")
except ImportError as e:
    CATBOOST_AVAILABLE = False
    # CatBoost installation on Windows may require Rust compiler
    # Users can install via: pip install catboost --no-build-isolation
    # Or use conda: conda install -c conda-forge catboost
    # Only log warning if logger is initialized
    if logger:
        logger.warning(
            f"CatBoost not available (ImportError: {e}). CatBoost will be skipped. "
            "To install on Windows, try: pip install catboost --no-build-isolation "
            "or use conda: conda install -c conda-forge catboost"
        )
    cb = None  # Set to None to prevent NameError

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

def get_features_from_config(config: dict) -> tuple[list[str], list[str]]:
    """
    Builds the feature lists dynamically based on dataset config toggles.

    Uses dict-based ALL_FEATURES_CONFIG with type metadata to automatically
    identify categorical vs numeric features.
    """
    logger.info("Building feature list from config toggles...")
    all_features = []
    categorical_features = []

    # Helper function to add features from workstream
    def add_workstream_features(ws_name: str):
        if ws_name in ALL_FEATURES_CONFIG:
            for feat in ALL_FEATURES_CONFIG[ws_name]:
                all_features.append(feat['name'])
                if feat['type'] == 'cat':
                    categorical_features.append(feat['name'])

    # Base timeseries features (luôn có)
    add_workstream_features('timeseries_base')

    # Các feature tùy chọn dựa trên config toggles
    if config.get('has_relational', False):
        add_workstream_features('relational')
    if config.get('has_intraday_patterns', False):
        add_workstream_features('intraday_patterns')
    if config.get('has_behavior', False):
        add_workstream_features('behavior')
    if config.get('has_price_promo', False):
        add_workstream_features('price_promo')
    if config.get('has_stockout', False):
        add_workstream_features('stockout')
    if config.get('has_weather', False):
        add_workstream_features('weather')

    logger.info(f"Built feature list: {len(all_features)} total, {len(categorical_features)} categorical")
    logger.info(f"Categorical features: {categorical_features}")

    return all_features, categorical_features


def prepare_data(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
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

    # Filter categorical features to only those present in df
    # Categorical features are automatically identified from config metadata
    categorical_features = [col for col in all_categorical_config if col in df.columns]

    logger.info(f"Found {len(all_features)} total features.")
    logger.info(f"Found {len(categorical_features)} categorical features (from config): {categorical_features}")

    X = df_model[all_features]
    y = df_model[target_col]

    # Intelligent Missing Value Handling
    # -----------------------------------
    # Different strategies for different feature types:
    # 1. Lag/rolling features: fillna(0) is appropriate (missing = no historical data)
    # 2. Weather features: should NOT be 0 (already handled in WS6 with ffill/bfill/mean)
    # 3. Price/promotion features: should NOT be 0 (0 price is invalid)
    # --------------------------------------------------

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    # Features safe to fill with 0 (lag, rolling, sales-related)
    safe_to_zero = [
        col for col in numeric_features
        if any(keyword in col.lower() for keyword in ['lag', 'rolling', 'sales', 'quantity', 'frequency', 'duration'])
    ]

    # Weather, price, promotion features - use forward/backward fill then mean
    sensitive_features = [
        col for col in numeric_features
        if any(keyword in col.lower() for keyword in ['temperature', 'precipitation', 'humidity', 'weather', 'price', 'discount'])
    ]

    # Other numeric features - use median
    other_features = [col for col in numeric_features if col not in safe_to_zero and col not in sensitive_features]

    if safe_to_zero:
        X.loc[:, safe_to_zero] = X[safe_to_zero].fillna(0)
        logger.info(f"Filled {len(safe_to_zero)} lag/rolling features with 0")

    if sensitive_features:
        for col in sensitive_features:
            if X[col].isnull().any():
                # Forward fill -> backward fill -> mean (pandas 2.x compatible)
                X.loc[:, col] = X[col].ffill().bfill().fillna(X[col].mean())
        logger.info(f"Filled {len(sensitive_features)} weather/price features with ffill/bfill/mean")

    if other_features:
        for col in other_features:
            if X[col].isnull().any():
                X.loc[:, col] = X[col].fillna(X[col].median())
        logger.info(f"Filled {len(other_features)} other features with median")

    # Final check: any remaining NaNs fill with 0 (safety net)
    remaining_nulls = X.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"Final safety net: Filling {remaining_nulls} remaining NaNs with 0")
        X = X.fillna(0).copy()

    logger.info("✅ Intelligent missing value handling complete")
    # --------------------------------------------------

    # Chuyển đổi categorical features sang 'category' dtype
    for col in categorical_features:
        X.loc[:, col] = X[col].astype('category')
        # Fill 'Unknown' cho categorical
        if X[col].isnull().any():
            X.loc[:, col] = X[col].cat.add_categories(['Unknown']).fillna('Unknown')

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

def create_model(model_type: str, quantile: float, categorical_features: list[str]) -> Any:
    """
    Tạo model instance dựa trên model type và quantile.

    Args:
        model_type: Model type (lightgbm, catboost, random_forest)
        quantile: Quantile level
        categorical_features: List categorical features

    Returns:
        Model instance
    """
    model_config = MODEL_CONFIGS[model_type]
    params = get_model_config(quantile, model_type=model_type)

    if model_type == 'lightgbm':
        model = lgb.LGBMRegressor(**params)
    elif model_type == 'catboost':
        if not CATBOOST_AVAILABLE or cb is None:
            error_msg = (
                "CatBoost not available. Install with: "
                "pip install catboost --no-build-isolation "
                "or use conda: conda install -c conda-forge catboost"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        try:
            # CatBoost cũng cần wrapper
            model = cb.CatBoostRegressor(**{k: v for k, v in params.items() if k != 'quantile'})
        except Exception as e:
            logger.error(f"Failed to create CatBoost model: {e}")
            raise RuntimeError(f"CatBoost model creation failed: {e}") from e
    elif model_type == 'random_forest':
        # Random Forest cần wrapper
        model = RandomForestRegressor(**{k: v for k, v in params.items() if k != 'quantile'})
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, model_config['quantile_support']


def train_quantile_model(
    model_type: str,
    quantile: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str]
) -> Any:
    """
    Train một quantile model.

    Args:
        model_type: Model type
        quantile: Quantile level
        X_train: Training features
        y_train: Training target
        categorical_features: List categorical features

    Returns:
        Trained model
    """
    model, quantile_support = create_model(model_type, quantile, categorical_features)

    # Encode categorical features cho các model không hỗ trợ categorical trực tiếp
    def encode_categorical(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
        """Encode categorical features thành numeric codes."""
        X_encoded = X.copy()
        for col in cat_features:
            if col in X_encoded.columns:
                if X_encoded[col].dtype.name == 'category':
                    X_encoded[col] = X_encoded[col].cat.codes
                elif X_encoded[col].dtype.name == 'object':
                    X_encoded[col] = pd.Categorical(X_encoded[col]).codes
        return X_encoded

    # Train model dựa trên model type
    if model_type == 'lightgbm' and quantile_support:
        # LightGBM hỗ trợ quantile regression trực tiếp
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_features
        )
    elif model_type == 'catboost':
        if not CATBOOST_AVAILABLE or cb is None:
            raise ImportError("CatBoost not available")
        try:
            # CatBoost hỗ trợ categorical features trực tiếp
            cat_indices = [i for i, col in enumerate(X_train.columns) if col in categorical_features]
            model.fit(
                X_train, y_train,
                cat_features=cat_indices,
                verbose=False
            )
        except Exception as e:
            logger.error(f"CatBoost training failed: {e}")
            raise RuntimeError(f"CatBoost training error: {e}") from e
    else:
        # Random Forest và các model khác
        X_train_encoded = encode_categorical(X_train, categorical_features)
        model.fit(X_train_encoded, y_train)

    return model


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
    model_types: list[str] | None = None
) -> dict[str, dict[float, Any]]:
    """
    Trains multiple models for each quantile.

    Args:
        X_train: Training features
        y_train: Training target
        categorical_features: List categorical features
        model_types: List model types to train. If None, use TRAINING_CONFIG['model_types']

    Returns:
        Dict of {model_type: {quantile: model}}
    """
    quantiles = TRAINING_CONFIG['quantiles']
    model_types = model_types or TRAINING_CONFIG.get('model_types', ['lightgbm'])

    # Validate: CHỈ TRAIN 5 QUANTILES (Q05, Q25, Q50, Q75, Q95)
    if len(quantiles) != 5:
        logger.warning(f"Expected 5 quantiles, but found {len(quantiles)}: {quantiles}")
    expected_quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    if set(quantiles) != set(expected_quantiles):
        logger.warning(f"Quantiles do not match expected values. Expected: {expected_quantiles}, Got: {quantiles}")

    logger.info("=" * 70)
    logger.info("TRAINING MODELS - 5 QUANTILES ONLY")
    logger.info(f"Model types: {model_types}")
    logger.info(f"Quantiles: {quantiles} (Total: {len(quantiles)} quantiles)")
    logger.info("=" * 70)

    # Filter available models
    available_models = []
    for model_type in model_types:
        if model_type == 'catboost' and not CATBOOST_AVAILABLE:
            logger.warning(f"Skipping {model_type}: not available")
            continue
        if model_type not in MODEL_CONFIGS:
            logger.warning(f"Skipping {model_type}: not in MODEL_CONFIGS")
            continue
        available_models.append(model_type)

    if not available_models:
        raise ValueError("No available models to train")

    logger.info(f"Training {len(available_models)} model types: {available_models}")

    all_models = {}
    total_start_time = time.time()

    for model_type in available_models:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training {model_type.upper()} models")
        logger.info(f"{'='*70}")
        all_models[model_type] = {}
        model_start_time = time.time()

        for quantile in quantiles:
            logger.info(f"Training {model_type} Q{int(quantile*100):02d} model (alpha={quantile})...")

            try:
                model = train_quantile_model(
                    model_type, quantile, X_train, y_train, categorical_features
                )
                all_models[model_type][quantile] = model
                logger.info(f"  ✓ {model_type} Q{int(quantile*100):02d} trained")
            except Exception as e:
                logger.error(f"  ✗ Error training {model_type} Q{int(quantile*100):02d}: {e}")
                raise

        logger.info(f"{model_type.upper()} models trained in {time.time() - model_start_time:.2f}s")

    total_time = time.time() - total_start_time
    total_models = sum(len(models) for models in all_models.values())
    logger.info(f"\n{'='*70}")
    logger.info(f"All {total_models} models trained in {total_time:.2f}s")
    logger.info(f"{'='*70}")

    return all_models

def encode_categorical_for_prediction(X: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    """Encode categorical features thành numeric codes cho prediction."""
    X_encoded = X.copy()
    for col in categorical_features:
        if col in X_encoded.columns:
            if X_encoded[col].dtype.name == 'category':
                X_encoded[col] = X_encoded[col].cat.codes
            elif X_encoded[col].dtype.name == 'object':
                X_encoded[col] = pd.Categorical(X_encoded[col]).codes
    return X_encoded


def predict_with_model(model: Any, model_type: str, X: pd.DataFrame,
                      categorical_features: list[str]) -> np.ndarray:
    """
    Predict với model, xử lý categorical features cho từng model type.

    Args:
        model: Trained model
        model_type: Model type
        X: Features
        categorical_features: List categorical features

    Returns:
        Predictions
    """
    if model_type == 'lightgbm':
        return model.predict(X)
    elif model_type == 'catboost':
        if not CATBOOST_AVAILABLE or cb is None:
            raise ImportError("CatBoost not available")
        try:
            return model.predict(X)
        except Exception as e:
            logger.error(f"CatBoost prediction failed: {e}")
            raise RuntimeError(f"CatBoost prediction error: {e}") from e
    else:
        # Random Forest và các model khác cần encode categorical
        X_encoded = encode_categorical_for_prediction(X, categorical_features)
        return model.predict(X_encoded)


def evaluate_quantile_models(
    all_models: dict[str, dict[float, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    categorical_features: list[str]
) -> dict[str, dict[str, float]]:
    """
    Evaluates all models using pinball loss.

    Args:
        all_models: Dict of {model_type: {quantile: model}}
        X_test: Test features
        y_test: Test target
        categorical_features: List categorical features

    Returns:
        Dict of {model_type: {metric: value}}
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    logger.info("=" * 70)
    logger.info("EVALUATING QUANTILE MODELS")
    logger.info("=" * 70)

    quantiles = TRAINING_CONFIG['quantiles']
    all_metrics = {}

    for model_type, models in all_models.items():
        logger.info(f"\nEvaluating {model_type.upper()} models...")
        metrics = {}
        predictions = {}

        for quantile in quantiles:
            if quantile not in models:
                continue

            model = models[quantile]
            y_pred = predict_with_model(model, model_type, X_test, categorical_features)
            y_pred = np.maximum(y_pred, 0)  # Clip negative predictions
            predictions[quantile] = y_pred

            # Pinball loss
            pinball = mean_pinball_loss(y_test, y_pred, alpha=quantile)
            metrics[f'q{int(quantile*100):02d}_pinball_loss'] = pinball

            # MAE
            mae = mean_absolute_error(y_test, y_pred)
            metrics[f'q{int(quantile*100):02d}_mae'] = mae

            # RMSE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics[f'q{int(quantile*100):02d}_rmse'] = rmse

            logger.info(f"  Q{int(quantile*100):02d} - Pinball: {pinball:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # Coverage
        if len(predictions) >= 2:
            lower_q = min(quantiles)
            upper_q = max(quantiles)
            if lower_q in predictions and upper_q in predictions:
                lower_pred = predictions[lower_q]
                upper_pred = predictions[upper_q]
                coverage = ((y_test >= lower_pred) & (y_test <= upper_pred)).mean()
                metrics[f'coverage_{(upper_q-lower_q)*100:.0f}%'] = coverage
                logger.info(f"  Coverage ({ (upper_q-lower_q)*100:.0f}%): {coverage*100:.2f}%")

        # R2 score (median quantile)
        median_q = 0.50
        if median_q in predictions:
            r2 = r2_score(y_test, predictions[median_q])
            metrics['r2_score'] = r2
            logger.info(f"  R2 Score: {r2:.4f}")

        all_metrics[model_type] = metrics

    return all_metrics

def save_artifacts(
    all_models: dict[str, dict[float, Any]],
    features_config: dict,
    all_metrics: dict[str, dict[str, float]]
) -> None:
    """Saves models, features, and metrics."""
    logger.info("Saving model artifacts...")

    # Save models
    for model_type, models in all_models.items():
        for quantile, model in models.items():
            model_filename = f"{model_type}_q{int(quantile*100):02d}_forecaster.joblib"
            model_path = OUTPUT_FILES['models_dir'] / model_filename
            joblib.dump(model, model_path)
            logger.info(f"  {model_filename} saved to: {model_path}")

    # Save feature config
    features_path = OUTPUT_FILES['model_features']
    with open(features_path, 'w') as f:
        json.dump(features_config, f, indent=4)
    logger.info(f"  Feature config saved to: {features_path}")

    # Save metrics
    metrics_path = OUTPUT_FILES['model_metrics']
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logger.info(f"  Metrics saved to: {metrics_path}")

def main(args=None) -> None:
    """Orchestrates the entire training pipeline."""
    # Create default args if none provided (for orchestrator compatibility)
    if args is None:
        args = argparse.Namespace(
            tune=False,
            model_types=None,
            skip_feature_selection=False
        )

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

    # Feature Selection (Step 2.5) - Select optimal feature subset
    logger.info("=" * 70)
    logger.info("STEP 2.5: FEATURE SELECTION")
    logger.info("=" * 70)

    if args.skip_feature_selection:
        logger.info("Feature selection SKIPPED (--skip-feature-selection flag)")
        selected_features = features
    else:
        # Reconstruct full dataframe for feature selection
        df_for_selection = pd.concat([X_train, X_test], axis=0)
        target_col = config['target_column']
        df_for_selection[target_col] = pd.concat([y_train, y_test], axis=0)

        # Run feature selection
        selection_result = get_optimal_features(
            df=df_for_selection,
            target_col=target_col,
            importance_threshold=0.005,  # Keep features with >0.5% importance
            correlation_threshold=0.95,   # Remove highly correlated features
            max_features=None,  # No hard limit
            save_report=True
        )

        selected_features = selection_result['selected_features']
        logger.info("Feature Selection Results:")
        logger.info(f"  Initial features: {selection_result['n_initial']}")
        logger.info(f"  Selected features: {selection_result['n_selected']}")
        logger.info(f"  Reduction: {(1 - selection_result['n_selected']/selection_result['n_initial'])*100:.1f}%")

        # Update X_train and X_test with selected features
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        # Update categorical features list (only keep those in selected features)
        cat_features = [f for f in cat_features if f in selected_features]
        logger.info(f"  Categorical features after selection: {len(cat_features)}")

    logger.info("=" * 70)

    # 4. Train Models
    logger.info("STEP 3: TRAIN QUANTILE MODELS")
    model_types = args.model_types if args.model_types else TRAINING_CONFIG.get('model_types', ['lightgbm'])

    if args.tune:
        logger.warning("Hyperparameter tuning not implemented. Using standard params.")

    all_models = train_quantile_models(X_train, y_train, cat_features, model_types=model_types)

    # 5. Evaluate
    logger.info("STEP 4: EVALUATE QUANTILE MODELS")
    all_metrics = evaluate_quantile_models(all_models, X_test, y_test, cat_features)

    # 6. Save
    logger.info("STEP 5: SAVE ARTIFACTS")
    # Save selected features (or all features if selection was skipped)
    final_features_config = {
        "all_features": selected_features if not args.skip_feature_selection else features,
        "categorical_features": cat_features,
        "quantiles": TRAINING_CONFIG['quantiles'],
        "model_types": list(all_models.keys()),
        "dataset_trained_on": config['name'],
        "feature_selection_enabled": not args.skip_feature_selection
    }
    save_artifacts(all_models, final_features_config, all_metrics)

    logger.info("=" * 70)
    logger.info(f"COMPLETE! Total runtime: {time.time() - total_start_time:.2f} seconds.")
    logger.info(f"Artifacts saved to: {OUTPUT_FILES['models_dir']}")
    logger.info("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train quantile regression models')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning (nếu được implement)')
    parser.add_argument('--model-types', nargs='+', type=str, default=None,
                       help='Model types to train (lightgbm, catboost, random_forest)')
    parser.add_argument('--skip-feature-selection', action='store_true',
                       help='Skip automatic feature selection and use all features')
    args = parser.parse_args()

    main(args)
