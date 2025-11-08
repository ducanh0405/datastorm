"""
Project Configuration
=====================
Centralized configuration management for the E-Grocery Forecaster pipeline.

All hard-coded values should be moved here for easier maintenance and deployment.
"""
from pathlib import Path
from typing import Dict, List, Any

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================================
# DATA PATHS
# ============================================================================

DATA_DIRS = {
    'poc_data': PROJECT_ROOT / 'data' / 'poc_data',  # 1% sample for testing
    'raw_data': PROJECT_ROOT / 'data' / '2_raw',  # Full raw data
    'processed_data': PROJECT_ROOT / 'data' / '3_processed',  # Processed features
    'models': PROJECT_ROOT / 'models',  # Trained models
    'reports': PROJECT_ROOT / 'reports',  # Reports and metrics
    'logs': PROJECT_ROOT / 'logs',  # Log files
}

# Output files
OUTPUT_FILES = {
    'master_feature_table': DATA_DIRS['processed_data'] / 'master_feature_table.parquet',
    'model_q05': DATA_DIRS['models'] / 'q05_forecaster.joblib',
    'model_q50': DATA_DIRS['models'] / 'q50_forecaster.joblib',
    'model_q95': DATA_DIRS['models'] / 'q95_forecaster.joblib',
    'model_features': DATA_DIRS['models'] / 'model_features.json',
    'model_metrics': DATA_DIRS['reports'] / 'metrics' / 'quantile_model_metrics.json',
    'models_dir': DATA_DIRS['models'],  # Directory for models
    'reports_dir': DATA_DIRS['reports'],  # Directory for reports
    # Dashboard outputs
    'dashboard_dir': DATA_DIRS['reports'] / 'dashboard',
    'predictions_test': DATA_DIRS['reports'] / 'predictions_test_set.csv',
}

# ============================================================================
# FEATURE ENGINEERING CONFIG
# ============================================================================

# WS0: Aggregation
AGGREGATION_CONFIG = {
    'groupby_keys': ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
    'aggregation_rules': {
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum',
        'RETAIL_DISC': 'sum',
        'COUPON_DISC': 'sum',
        'COUPON_MATCH_DISC': 'sum',
    }
}

# WS2: Time-Series Features
TIMESERIES_CONFIG = {
    'target_columns': ['SALES_VALUE', 'QUANTITY'],
    'lags': [1, 4, 8, 12],  # Weeks: 1 week, 1 month, 2 months, 3 months
    'rolling_windows': [4, 8, 12],  # Weeks
    'base_lag': 1,  # Calculate rolling stats on lag_1 (leak-safe)
}

# ============================================================================
# MODEL TRAINING CONFIG
# ============================================================================

# Quantile levels for probabilistic forecasting
QUANTILES = [0.05, 0.50, 0.95]  # Lower bound, median, upper bound

# LightGBM hyperparameters
LIGHTGBM_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,  # No limit
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1,  # L2 regularization
    'random_state': 42,
    'n_jobs': -1,  # Use all CPUs
    'verbose': -1,  # Suppress LightGBM output
}

# Training configuration
TRAINING_CONFIG = {
    'quantiles': QUANTILES,
    'hyperparameters': LIGHTGBM_PARAMS,
    'train_test_split': {
        'method': 'time_based',  # Use time-based split (leak-safe)
        'cutoff_percentile': 0.8,  # 80th percentile of WEEK_NO
    },
    'tuning_iterations': 20,  # For future hyperparameter tuning
    'cv_folds': 3,  # For future cross-validation
}

# ============================================================================
# FEATURE SELECTION
# ============================================================================

# Numeric features (from all workstreams)
NUMERIC_FEATURES = [
    # WS0: Aggregated
    'QUANTITY',

    # WS2: Time-Series Lags
    'sales_value_lag_1', 'sales_value_lag_4', 'sales_value_lag_8', 'sales_value_lag_12',
    'quantity_lag_1', 'quantity_lag_4',

    # WS2: Rolling Statistics (Complete set)
    'rolling_mean_4_lag_1', 'rolling_std_4_lag_1', 'rolling_max_4_lag_1', 'rolling_min_4_lag_1',
    'rolling_mean_8_lag_1', 'rolling_std_8_lag_1', 'rolling_max_8_lag_1', 'rolling_min_8_lag_1',
    'rolling_mean_12_lag_1', 'rolling_std_12_lag_1', 'rolling_max_12_lag_1', 'rolling_min_12_lag_1',

    # WS2: Calendar & Trend Features
    'week_of_year', 'month_proxy', 'quarter', 'week_sin', 'week_cos',
    'wow_change', 'wow_pct_change', 'momentum', 'volatility',

    # WS4: Price/Promo
    'base_price', 'total_discount', 'discount_pct',
]

# Categorical features
CATEGORICAL_FEATURES = [
    # WS1: Relational
    'DEPARTMENT', 'COMMODITY_DESC',
    
    # WS4: Price/Promo
    'is_on_display', 'is_on_mailer', 'is_on_retail_promo', 'is_on_coupon_promo',
]

# Required columns for validation
REQUIRED_COLUMNS = ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'SALES_VALUE']

# ============================================================================
# VALIDATION CONFIG
# ============================================================================

VALIDATION_CONFIG = {
    'required_columns': REQUIRED_COLUMNS,
    'data_ranges': {
        'SALES_VALUE': (0, None),  # Non-negative
        'QUANTITY': (0, None),  # Non-negative
        'WEEK_NO': (1, 104),  # Typical week range
        'discount_pct': (0, 1),  # 0-100%
    },
    'quality_thresholds': {
        'excellent': 90,
        'good': 75,
        'fair': 60,
    }
}

# ============================================================================
# PERFORMANCE CONFIG (NEW!)
# ============================================================================

PERFORMANCE_CONFIG = {
    'use_polars': True,  # Enable Polars for 2-10x speedup (if available)
    'use_duckdb': False,  # Enable DuckDB for SQL operations (if available)
    'fallback_to_pandas': True,  # Fall back to pandas if Polars/DuckDB not available
    'memory_limit_gb': 16,  # Memory limit for large datasets
    'parallel_threads': -1,  # Use all cores (-1 = all available)
    'chunk_size_mb': 100,  # Chunk size for large file processing
    'lazy_evaluation': True,  # Use lazy evaluation for very large datasets
}

# ============================================================================
# LOGGING CONFIG
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': DATA_DIRS['logs'] / 'pipeline.log',
    'log_to_file': True,
    'log_to_console': True,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_features() -> List[str]:
    """Get list of all expected features."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def get_model_config(quantile: float) -> Dict[str, Any]:
    """
    Get model configuration for a specific quantile.

    Args:
        quantile: Quantile level (e.g., 0.05, 0.50, 0.95)

    Returns:
        Dictionary with model configuration
    """
    config = LIGHTGBM_PARAMS.copy()
    config['objective'] = 'quantile'
    config['alpha'] = quantile
    return config


def get_data_directory(prefer_poc_data: bool = True) -> Path:
    """
    Get the appropriate data directory based on availability and preference.

    Args:
        prefer_poc_data: If True, prefer POC data over full data (recommended for development)

    Returns:
        Path to the data directory
    """
    if prefer_poc_data and DATA_DIRS['poc_data'].exists():
        return DATA_DIRS['poc_data']
    elif DATA_DIRS['raw_data'].exists():
        return DATA_DIRS['raw_data']
    else:
        # Fallback to poc_data directory (will be created if needed)
        return DATA_DIRS['poc_data']


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    for dir_path in DATA_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    # Also create metrics subdirectory
    (DATA_DIRS['reports'] / 'metrics').mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = None, log_to_file: bool = None) -> None:
    """
    Setup centralized logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
    """
    import logging

    level = level or LOGGING_CONFIG.get('level', 'INFO')
    log_format = LOGGING_CONFIG.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_to_file = log_to_file if log_to_file is not None else LOGGING_CONFIG.get('log_to_file', True)

    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[]
    )

    # Add console handler
    if LOGGING_CONFIG.get('log_to_console', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)

    # Add file handler
    if log_to_file:
        log_file = LOGGING_CONFIG.get('log_file', DATA_DIRS['logs'] / 'pipeline.log')
        # Ensure logs directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

