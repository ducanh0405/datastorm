"""
Unified Model Training Pipeline
===============================
Trains quantile regression models for probabilistic forecasting.

Features:
- Standard training (fast, reliable)
- Optional hyperparameter tuning with Optuna (slow, optimal)
- Time-series cross-validation for tuning
- Automatic fallback to standard training if Optuna not available

USAGE:
    # Standard training
    python _03_model_training.py

    # With hyperparameter tuning
    python _03_model_training.py --tune --trials 50
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
from typing import Dict, Tuple, List, Any, Optional
import argparse

# === PROJECT ROOT ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# ====================

# Import configuration
from src.config import (
    OUTPUT_FILES, TRAINING_CONFIG, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    get_model_config, ensure_directories
)

# Try to import Optuna (optional for tuning)
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure directories exist
ensure_directories()


# -----------------------------------------------------------------
# 2. FUNCTIONAL DEFINITIONS (All print/logging in English)
# -----------------------------------------------------------------

def load_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the clean feature table from the processing pipeline.
    
    Args:
        filepath: Path to feature table file (.parquet or .csv)
    
    Returns:
        Loaded dataframe
    
    Raises:
        SystemExit: If file not found or loading fails
    """
    logger.info(f"Loading data from: {filepath}...")
    start_time = time.time()
    try:
        if str(filepath).endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif str(filepath).endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        logger.info(f"Load complete. Shape: {df.shape}. (Took {time.time() - start_time:.2f}s)")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        logger.error("Please run the data processing pipeline (_02_feature_enrichment.py) first.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading file: {e}", exc_info=True)
        sys.exit(1)


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    Prepares data for modeling: selects features and performs time-based split.
    
    Filters, creates target variable (sales), selects features from all workstreams,
    and splits data BY TIME (leak-safe).
    
    Args:
        df: Master feature table
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, all_features, categorical_features)
    
    Raises:
        SystemExit: If required columns or features are missing
    """
    logger.info("Preparing data for modeling...")

    # Define target variable
    target_col = 'SALES_VALUE'  # Assuming Dunnhumby

    if target_col not in df.columns:
        if 'sales' in df.columns:  # Fallback for M5
            target_col = 'sales'
        else:
            logger.error(f"Target column '{target_col}' or 'sales' not found.")
            sys.exit(1)

    logger.info(f"Target variable (Y) set to: {target_col}")

    df_model = df.dropna(subset=[target_col]).copy()

    if df_model.empty:
        logger.error("No data left to train after dropping NaN target values.")
        sys.exit(1)

    # Use features from config
    all_features = [col for col in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if col in df.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in all_features]

    missing_features = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) - set(df.columns)
    if missing_features:
        logger.warning(f"Missing expected features (WS may be toggled off): {missing_features}")

    if not all_features:
        logger.error("No valid features found in the input file.")
        sys.exit(1)

    logger.info(f"Found {len(all_features)} valid features for training.")

    X = df_model[all_features]
    y = df_model[target_col]
    
    # Add WEEK_NO for time-based split
    if 'WEEK_NO' not in df_model.columns:
        logger.error("WEEK_NO column required for time-based split!")
        sys.exit(1)
    
    week_no = df_model['WEEK_NO']

    logger.info(f"Converting {len(categorical_features)} columns to 'category' dtype...")
    for col in categorical_features:
        X[col] = X[col].astype('category')

    # Time-based split (NO RANDOM SHUFFLE!)
    logger.info("=" * 70)
    logger.info("PERFORMING TIME-BASED SPLIT (Leak-Safe)")
    logger.info("=" * 70)
    
    # Get cutoff from config
    cutoff_percentile = TRAINING_CONFIG['train_test_split']['cutoff_percentile']
    cutoff_week = week_no.quantile(cutoff_percentile)
    logger.info(f"Time cutoff: WEEK_NO < {cutoff_week:.0f} = TRAIN, >= {cutoff_week:.0f} = TEST")
    
    # Split by time
    train_mask = week_no < cutoff_week
    test_mask = week_no >= cutoff_week
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    logger.info(f"Train set: {len(X_train):,} samples (weeks {week_no[train_mask].min():.0f}-{week_no[train_mask].max():.0f})")
    logger.info(f"Test set:  {len(X_test):,} samples (weeks {week_no[test_mask].min():.0f}-{week_no[test_mask].max():.0f})")
    logger.info(f"Split ratio: {len(X_train)/len(X)*100:.1f}% train / {len(X_test)/len(X)*100:.1f}% test")
    logger.info("=" * 70)
    logger.info("Data preparation complete (TIME-BASED, NO LEAKAGE).")

    return X_train, X_test, y_train, y_test, all_features, categorical_features


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str],
    quantiles: List[float] = None
) -> Dict[float, lgb.LGBMRegressor]:
    """
    Trains separate LightGBM models for each quantile (probabilistic forecasting).
    
    Uses objective='quantile' to enable prediction intervals for inventory optimization.
    
    Args:
        X_train: Training features
        y_train: Training target
        categorical_features: List of categorical column names
        quantiles: List of quantile levels to train (default: from config)
    
    Returns:
        Dict mapping quantile -> trained model
    """
    if quantiles is None:
        quantiles = TRAINING_CONFIG['quantiles']
    
    logger.info("=" * 70)
    logger.info("TRAINING QUANTILE MODELS (Probabilistic Forecasting)")
    logger.info("=" * 70)
    logger.info(f"Training {len(quantiles)} separate models for quantiles: {quantiles}")
    start_train = time.time()
    
    models = {}
    
    for alpha in quantiles:
        logger.info(f"Training Q{int(alpha*100):02d} model (alpha={alpha})...")
        
        # Get model config from centralized config
        model_params = get_model_config(alpha)
        
        model = lgb.LGBMRegressor(**model_params)
        
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_features
        )
        
        models[alpha] = model
        logger.info(f"Q{int(alpha*100):02d} model trained successfully")
    
    logger.info(f"All quantile models trained (Took {time.time() - start_train:.2f}s)")
    logger.info("=" * 70)

    return models


def train_quantile_models_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: List[str],
    feature_cols: List[str] = None,
    n_trials: int = 30,
    quantiles: List[float] = None
) -> Dict[float, lgb.LGBMRegressor]:
    """
    Trains separate LightGBM models for each quantile with hyperparameter tuning.

    Uses Optuna for hyperparameter optimization with time-series cross-validation.

    Args:
        X_train: Training features
        y_train: Training target
        categorical_features: List of categorical column names
        feature_cols: List of feature column names (default: X_train.columns)
        n_trials: Number of Optuna trials per quantile
        quantiles: List of quantile levels to train (default: from config)

    Returns:
        Dict mapping quantile -> trained model with tuned hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not available, falling back to standard training")
        return train_quantile_models(X_train, y_train, categorical_features, quantiles)

    if quantiles is None:
        quantiles = TRAINING_CONFIG['quantiles']

    if feature_cols is None:
        feature_cols = X_train.columns.tolist()

    logger.info("=" * 70)
    logger.info("TRAINING TUNED QUANTILE MODELS (Hyperparameter Optimization)")
    logger.info("=" * 70)
    logger.info(f"Training {len(quantiles)} separate models with {n_trials} trials each")
    start_train = time.time()

    models = {}

    for alpha in quantiles:
        logger.info(f"\n[TUNING] Q{int(alpha*100):02d} (alpha={alpha}) - {n_trials} trials")

        # Get best hyperparameters for this quantile
        # Create full training DataFrame with features and target
        train_df = X_train.copy()
        train_df['SALES_VALUE'] = y_train

        best_params = tune_quantile_hyperparameters(
            train_df, feature_cols, categorical_features, alpha, n_trials
        )

        # Set the alpha for quantile regression
        best_params['alpha'] = alpha

        # Train final model with best parameters
        logger.info(f"Training final model with tuned parameters...")
        model = lgb.LGBMRegressor(**best_params)
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_features
        )

        models[alpha] = model
        logger.info(f"Q{int(alpha*100):02d} tuned model trained successfully")

    logger.info(f"\nAll tuned quantile models trained (Took {time.time() - start_train:.2f}s)")
    logger.info("=" * 70)

    return models


def evaluate_quantile_models(
    models: Dict[float, lgb.LGBMRegressor],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluates quantile models using pinball loss (the correct metric for quantile regression).
    
    Args:
        models: Dict mapping quantile -> model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dict of evaluation metrics
    """
    logger.info("=" * 70)
    logger.info("EVALUATING QUANTILE MODELS (Pinball Loss)")
    logger.info("=" * 70)
    
    metrics = {}
    predictions = {}
    
    for alpha, model in models.items():
        logger.info(f"Evaluating Q{int(alpha*100):02d} (alpha={alpha})...")
        
        y_pred = model.predict(X_test)
        y_pred[y_pred < 0] = 0  # Clip negative predictions
        
        predictions[alpha] = y_pred
        
        # Pinball loss (primary metric for quantile regression)
        pinball = mean_pinball_loss(y_test, y_pred, alpha=alpha)
        
        # Also calculate RMSE for reference
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        metrics[f'q{int(alpha*100):02d}_pinball_loss'] = pinball
        metrics[f'q{int(alpha*100):02d}_rmse'] = rmse
        
        logger.info(f"  Pinball Loss: {pinball:.4f}")
        logger.info(f"  RMSE (reference): {rmse:.4f}")
    
    # Calculate prediction interval coverage (if we have q05 and q95)
    if 0.05 in predictions and 0.95 in predictions:
        lower = predictions[0.05]
        upper = predictions[0.95]
        coverage = ((y_test >= lower) & (y_test <= upper)).mean()
        metrics['prediction_interval_coverage'] = coverage
        logger.info(f"Prediction Interval Coverage (90%): {coverage*100:.2f}%")
        logger.info(f"  (Target: ~90%, Actual: {coverage*100:.1f}%)")
    
    logger.info("=" * 70)
    
    return metrics


# ============================================================================
# TUNING FUNCTIONS (OPTIONAL - requires Optuna)
# ============================================================================

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Pinball loss for quantile regression."""
    residual = y_true - y_pred
    return np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual))


def create_time_series_cv_splits(
    train_df: pd.DataFrame,
    n_splits: int = 3
) -> list:
    """
    Time-series cross-validation splits (expanding window).

    Example with n_splits=3:
    - Fold 1: weeks 1-54 train, 55-68 val
    - Fold 2: weeks 1-68 train, 69-81 val
    - Fold 3: weeks 1-75 train, 76-81 val
    """
    # Ensure index is reset to positional
    train_df = train_df.reset_index(drop=True)

    weeks = sorted(train_df['WEEK_NO'].unique())
    n_weeks = len(weeks)

    if n_splits < 2:
        logger.warning("n_splits < 2, using single split at 80%")
        n_splits = 2

    splits = []

    for i in range(1, n_splits + 1):
        # Expanding window
        val_end_idx = int(n_weeks * (0.6 + 0.2 * i / n_splits))
        val_start_idx = max(0, val_end_idx - int(n_weeks * 0.15))

        val_end_week = weeks[min(val_end_idx, n_weeks - 1)]
        val_start_week = weeks[val_start_idx]

        train_mask = train_df['WEEK_NO'] < val_start_week
        val_mask = (train_df['WEEK_NO'] >= val_start_week) & (train_df['WEEK_NO'] < val_end_week)

        # Use numpy where to get positional indices
        train_idx = np.where(train_mask)[0].tolist()
        val_idx = np.where(val_mask)[0].tolist()

        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
            logger.info(f"  Fold {i}: Train weeks <{val_start_week}, Val weeks {val_start_week}-{val_end_week}")

    return splits


def tune_quantile_hyperparameters(
    train_df: pd.DataFrame,
    feature_cols: list,
    categorical_features: list,
    alpha: float,
    n_trials: int = 50,
    n_cv_splits: int = 3
) -> Dict[str, Any]:
    """
    Optuna-based hyperparameter tuning for one quantile.
    """
    if not OPTUNA_AVAILABLE:
        logger.warning(f"Optuna not available, using default params for alpha={alpha}")
        return {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }

    logger.info(f"[TUNING] Starting Optuna for alpha={alpha}, {n_trials} trials, {n_cv_splits}-fold CV")

    # Reset index to ensure positional indexing works
    train_df = train_df.reset_index(drop=True)

    cv_splits = create_time_series_cv_splits(train_df, n_splits=n_cv_splits)

    def objective(trial):
        params = {
            'objective': 'quantile',
            'alpha': alpha,
            'metric': 'quantile',
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

        cv_scores = []

        for train_idx, val_idx in cv_splits:
            # Use iloc for positional indexing
            X_tr = train_df.iloc[train_idx][feature_cols]
            y_tr = train_df.iloc[train_idx]['SALES_VALUE']
            X_val = train_df.iloc[val_idx][feature_cols]
            y_val = train_df.iloc[val_idx]['SALES_VALUE']

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                categorical_feature=categorical_features,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )

            preds = model.predict(X_val)
            score = pinball_loss(y_val.values, preds, alpha)
            cv_scores.append(score)

        return np.mean(cv_scores)

    study = optuna.create_study(direction='minimize', study_name=f'quantile_{alpha}')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"[TUNING] Best score: {study.best_value:.6f}")
    logger.info(f"[TUNING] Best params: {study.best_params}")

    best_params = study.best_params
    best_params['objective'] = 'quantile'
    best_params['alpha'] = alpha
    best_params['metric'] = 'quantile'
    best_params['verbosity'] = -1

    return best_params


def save_artifacts(
    models: Dict[float, lgb.LGBMRegressor],
    features_config: Dict,
    metrics: Dict,
    hyperparameters: Optional[Dict] = None
) -> None:
    """
    Saves quantile models, features, metrics, and optionally hyperparameters to disk.

    Saves 3 separate model files (q05, q50, q95) for probabilistic forecasting.

    Args:
        models: Dict mapping quantile -> model
        features_config: Feature configuration dict
        metrics: Evaluation metrics dict
        hyperparameters: Optional hyperparameters from tuning
    """
    logger.info("Saving model artifacts...")

    # Save each quantile model separately
    for alpha, model in models.items():
        model_key = f'model_q{int(alpha*100):02d}'
        if model_key in OUTPUT_FILES:
            model_path = OUTPUT_FILES[model_key]
        else:
            # Fallback to default naming
            model_path = OUTPUT_FILES['models_dir'] / f'q{int(alpha*100):02d}_forecaster.joblib'

        try:
            joblib.dump(model, model_path)
            logger.info(f"Q{int(alpha*100):02d} model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Error saving Q{int(alpha*100):02d} model: {e}", exc_info=True)

    # Save feature config
    try:
        features_path = OUTPUT_FILES['model_features']
        with open(features_path, 'w') as f:
            json.dump(features_config, f, indent=4)
        logger.info(f"Feature config saved to: {features_path}")
    except Exception as e:
        logger.error(f"Error saving feature config: {e}", exc_info=True)

    # Save metrics
    try:
        metrics_path = OUTPUT_FILES['model_metrics']
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to: {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}", exc_info=True)

    # Save hyperparameters if provided (from tuning)
    if hyperparameters:
        try:
            hyperparams_path = OUTPUT_FILES['models_dir'] / 'best_hyperparameters.json'
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparameters, f, indent=4)
            logger.info(f"Hyperparameters saved to: {hyperparams_path}")
        except Exception as e:
            logger.error(f"Error saving hyperparameters: {e}", exc_info=True)


# -----------------------------------------------------------------
# 3. MAIN ORCHESTRATOR (All English)
# -----------------------------------------------------------------

def main(tune_hyperparameters: bool = False, n_trials: int = 30) -> None:
    """
    Orchestrates the entire training pipeline.

    Pipeline steps:
    1. Time-based split (no random shuffle)
    2. Quantile regression (3 models: q05, q50, q95)
    3. Pinball loss evaluation

    Args:
        tune_hyperparameters: Whether to use hyperparameter tuning with Optuna
        n_trials: Number of Optuna trials per quantile
    """
    logger.info("=" * 70)
    logger.info("STARTING MODEL TRAINING PIPELINE (QUANTILE REGRESSION)")
    logger.info("=" * 70)
    total_start_time = time.time()

    logger.info("STEP 1: LOAD DATA")
    df = load_data(OUTPUT_FILES['master_feature_table'])

    logger.info("STEP 2: PREPARE DATA & TIME-BASED SPLIT")
    X_train, X_test, y_train, y_test, features, cat_features = prepare_data(df)

    logger.info("STEP 3: TRAIN QUANTILE MODELS (Q05, Q50, Q95)")
    if tune_hyperparameters and OPTUNA_AVAILABLE:
        logger.info(f"  Using hyperparameter tuning with {n_trials} trials per quantile")
        quantile_models = train_quantile_models_tuned(X_train, y_train, cat_features, features, n_trials=n_trials)
    elif tune_hyperparameters and not OPTUNA_AVAILABLE:
        logger.warning("  Optuna not available, falling back to standard training")
        quantile_models = train_quantile_models(X_train, y_train, cat_features)
    else:
        logger.info("  Using standard training (no hyperparameter tuning)")
        quantile_models = train_quantile_models(X_train, y_train, cat_features)

    logger.info("STEP 4: EVALUATE QUANTILE MODELS (PINBALL LOSS)")
    metrics = evaluate_quantile_models(quantile_models, X_test, y_test)

    logger.info("STEP 5: SAVE ARTIFACTS (MODELS, FEATURES, METRICS)")
    features_config = {
        "all_features": features,
        "categorical_features": cat_features,
        "quantiles": TRAINING_CONFIG['quantiles'],
        "model_type": "LightGBM_Quantile_Regression",
        "hyperparameter_tuning": tune_hyperparameters,
        "optuna_available": OPTUNA_AVAILABLE
    }
    save_artifacts(quantile_models, features_config, metrics)

    # Save hyperparameters if tuning was used
    if tune_hyperparameters and OPTUNA_AVAILABLE:
        try:
            # Get hyperparameters from tuned models
            hyperparameters = {}
            for alpha, model in quantile_models.items():
                quantile_key = f'q{int(alpha*100):02d}'
                # Extract parameters from the trained model
                params = model.get_params()
                # Remove non-serializable parameters
                serializable_params = {k: v for k, v in params.items()
                                     if isinstance(v, (int, float, str, bool, list))}
                hyperparameters[quantile_key] = serializable_params

            hyperparams_path = OUTPUT_FILES['models_dir'] / 'best_hyperparameters.json'
            with open(hyperparams_path, 'w') as f:
                json.dump(hyperparameters, f, indent=2)
            logger.info(f"Hyperparameters saved to: {hyperparams_path}")
        except Exception as e:
            logger.warning(f"Could not save hyperparameters: {e}")

    logger.info("=" * 70)
    logger.info(f"COMPLETE! Total runtime: {time.time() - total_start_time:.2f} seconds.")
    logger.info(f"Artifacts saved to: {OUTPUT_FILES['models_dir']} and {OUTPUT_FILES['reports_dir'] / 'metrics'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train quantile regression models with optional hyperparameter tuning')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials per quantile (default: 30)')
    parser.add_argument('--quick', action='store_true', help='Quick mode (no tuning, 10 trials)')

    args = parser.parse_args()

    if args.quick:
        args.tune = False
        args.trials = 10

    # Run main pipeline with tuning option
    main(tune_hyperparameters=args.tune, n_trials=args.trials)