"""
Project Configuration (Phiên bản đầy đủ, modular)
=================================================
Quản lý config tập trung cho SmartGrocy.
"""
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- 1. PROJECT PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def setup_project_path() -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT

DATA_DIRS = {
    'raw_data': PROJECT_ROOT / 'data' / '2_raw',
    'processed_data': PROJECT_ROOT / 'data' / '3_processed',
    'models': PROJECT_ROOT / 'models',
    'reports': PROJECT_ROOT / 'reports',
    'logs': PROJECT_ROOT / 'logs',
}

OUTPUT_FILES = {
    'master_feature_table': DATA_DIRS['processed_data'] / 'master_feature_table.parquet',
    'master_feature_table_csv': DATA_DIRS['processed_data'] / 'master_feature_table.csv',
    'model_features': DATA_DIRS['models'] / 'model_features.json',
    'model_metrics': DATA_DIRS['reports'] / 'metrics' / 'model_metrics.json',
    'models_dir': DATA_DIRS['models'],
    'reports_dir': DATA_DIRS['reports'],
    'dashboard_dir': DATA_DIRS['reports'] / 'dashboard',
    'predictions_test': DATA_DIRS['reports'] / 'predictions_test_set.csv',
    'shap_values_dir': DATA_DIRS['reports'] / 'shap_values',
    'dashboard_html': DATA_DIRS['reports'] / 'dashboard' / 'forecast_dashboard.html',
}

# --- 2. DATASET CONFIGURATIONS (CORE) ---
# Định nghĩa các "khả năng" của mỗi dataset
DATASET_CONFIGS = {
    'freshretail': {
        'name': 'FreshRetailNet-50K',
        'file_format': 'parquet', # hoặc 'csv'
        'temporal_unit': 'hour',
        'time_column': 'hour_timestamp',
        'target_column': 'sales_quantity',
        'groupby_keys': ['product_id', 'store_id', 'hour_timestamp'],
        'required_columns': ['product_id', 'store_id', 'hour_timestamp', 'sales_quantity'],
        
        # --- Feature Workstream Toggles ---
        'has_relational': True,  # (product_info.csv)
        'has_stockout': True,    # (ws5)
        'has_weather': True,     # (ws6)
        'has_price_promo': False,  # (Không có causal_data)
        'has_behavior': False,   # (Không có clickstream)
        
        # --- WS2 Config ---
        'lag_periods': [1, 24, 48, 168], # 1h, 1d, 2d, 1w
        'rolling_windows': [24, 168],
        'has_intraday_patterns': True,
    },
    'dunnhumby': {
        'name': 'Dunnhumby',
        'file_format': 'csv',
        'temporal_unit': 'week',
        'time_column': 'WEEK_NO',
        'target_column': 'SALES_VALUE',
        'groupby_keys': ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],

        # --- Feature Workstream Toggles ---
        'has_relational': True,  # (ws1 - product, hh_demographic)
        'has_stockout': False,   # (ws5)
        'has_weather': False,    # (ws6)
        'has_price_promo': True,   # (ws4 - causal_data)
        'has_behavior': True,    # (ws3 - clickstream_log)
        
        # --- WS2 Config ---
        'lag_periods': [1, 4, 8, 12],
        'rolling_windows': [4, 8, 12],
        'has_intraday_patterns': False,
    }
    # Thêm các dataset khác (M5, Olist) vào đây nếu cần
}

# --- 3. ACTIVE CONFIG ---
# Thay đổi 'freshretail' thành 'dunnhumby' để chạy pipeline cho dataset đó
ACTIVE_DATASET = 'freshretail'

# --- MEMORY OPTIMIZATION CONFIG ---
# Tùy chọn để giảm kích thước dataset cho máy có RAM hạn chế
# Để tắt optimization, set enable_sampling=False
# Xem docs/MEMORY_OPTIMIZATION.md để biết thêm chi tiết
MEMORY_OPTIMIZATION = {
    'enable_sampling': False,  # Tắt sampling để chạy full dataset
    'sample_fraction': 1.0,   # Fraction of data to use (1.0 = 100% = full dataset)
    'max_products': None,     # Giới hạn số products (None = không giới hạn)
    'max_stores': None,       # Giới hạn số stores (None = không giới hạn)
    'max_time_periods': None, # Giới hạn số time periods (None = không giới hạn)
    'use_chunking': True,     # Sử dụng chunking cho operations lớn
    'chunk_size': 100000,     # Kích thước chunk
} 

def get_dataset_config(dataset_name=None):
    """Lấy config cho dataset đang hoạt động."""
    dataset = dataset_name or ACTIVE_DATASET
    if dataset not in DATASET_CONFIGS:
        raise KeyError(f"Dataset '{dataset}' not found. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset]

# --- 4. FEATURE CONFIG (Modular với Type Metadata) ---
# Each feature has 'name' and 'type' ('num' for numeric, 'cat' for categorical)
# This enables automatic categorical feature detection in model training
ALL_FEATURES_CONFIG = {
    # WS1: Relational (all categorical)
    'relational': [
        {'name': 'DEPARTMENT', 'type': 'cat'},
        {'name': 'COMMODITY_DESC', 'type': 'cat'},
        {'name': 'BRAND', 'type': 'cat'},
        {'name': 'MANUFACTURER', 'type': 'cat'},
    ],
    
    # WS2: Time-Series (Base) - mostly numeric, some categorical
    'timeseries_base': [
        {'name': 'sales_lag_1', 'type': 'num'},
        {'name': 'sales_lag_4', 'type': 'num'},
        {'name': 'sales_lag_8', 'type': 'num'},
        {'name': 'sales_lag_12', 'type': 'num'},
        {'name': 'sales_quantity_lag_1', 'type': 'num'},
        {'name': 'sales_quantity_lag_24', 'type': 'num'},
        {'name': 'sales_quantity_lag_48', 'type': 'num'},
        {'name': 'sales_quantity_lag_168', 'type': 'num'},
        {'name': 'rolling_mean_4_lag_1', 'type': 'num'},
        {'name': 'rolling_std_4_lag_1', 'type': 'num'},
        {'name': 'rolling_mean_8_lag_1', 'type': 'num'},
        {'name': 'rolling_std_8_lag_1', 'type': 'num'},
        {'name': 'rolling_mean_12_lag_1', 'type': 'num'},
        {'name': 'rolling_std_12_lag_1', 'type': 'num'},
        {'name': 'rolling_mean_24_lag_1', 'type': 'num'},
        {'name': 'rolling_std_24_lag_1', 'type': 'num'},
        {'name': 'rolling_mean_168_lag_1', 'type': 'num'},
        {'name': 'rolling_std_168_lag_1', 'type': 'num'},
        {'name': 'week_of_year', 'type': 'cat'},  # Cyclical, treat as categorical
        {'name': 'month_proxy', 'type': 'cat'},    # Cyclical, treat as categorical
        {'name': 'week_sin', 'type': 'num'},
        {'name': 'week_cos', 'type': 'num'},
    ],
    
    # WS2: Intraday Patterns (mix of categorical and numeric)
    'intraday_patterns': [
        {'name': 'hour_of_day', 'type': 'cat'},      # 0-23, categorical
        {'name': 'day_of_week', 'type': 'cat'},      # 0-6, categorical
        {'name': 'is_morning_peak', 'type': 'cat'},  # Binary flag
        {'name': 'is_evening_peak', 'type': 'cat'},  # Binary flag
        {'name': 'is_weekend', 'type': 'cat'},       # Binary flag
        {'name': 'hour_sin', 'type': 'num'},
        {'name': 'hour_cos', 'type': 'num'},
        {'name': 'dow_sin', 'type': 'num'},
        {'name': 'dow_cos', 'type': 'num'},
    ],

    # WS3: Behavior (all numeric)
    'behavior': [
        {'name': 'total_views', 'type': 'num'},
        {'name': 'total_addtocart', 'type': 'num'},
        {'name': 'total_transactions', 'type': 'num'},
        {'name': 'rate_view_to_cart', 'type': 'num'},
        {'name': 'rate_cart_to_buy', 'type': 'num'},
        {'name': 'rate_view_to_buy', 'type': 'num'},
        {'name': 'days_since_last_action', 'type': 'num'},
    ],
    
    # WS4: Price/Promo (mix)
    'price_promo': [
        {'name': 'base_price', 'type': 'num'},
        {'name': 'total_discount', 'type': 'num'},
        {'name': 'discount_pct', 'type': 'num'},
        {'name': 'is_on_display', 'type': 'cat'},         # Binary flag
        {'name': 'is_on_mailer', 'type': 'cat'},          # Binary flag
        {'name': 'is_on_retail_promo', 'type': 'cat'},    # Binary flag
        {'name': 'is_on_coupon_promo', 'type': 'cat'},    # Binary flag
    ],
    
    # WS5: Stockout (all numeric)
    'stockout': [
        {'name': 'latent_demand', 'type': 'num'},
        {'name': 'stockout_duration', 'type': 'num'},
        {'name': 'time_since_last_stockout', 'type': 'num'},
        {'name': 'time_to_next_stockout', 'type': 'num'},
        {'name': 'stockout_frequency', 'type': 'num'},
        {'name': 'stockout_severity', 'type': 'num'},
    ],
    
    # WS6: Weather (mix)
    'weather': [
        {'name': 'temperature', 'type': 'num'},
        {'name': 'precipitation', 'type': 'num'},
        {'name': 'humidity', 'type': 'num'},
        {'name': 'temp_category', 'type': 'cat'},         # Low/Medium/High
        {'name': 'is_rainy', 'type': 'cat'},              # Binary flag
        {'name': 'rain_intensity', 'type': 'cat'},        # None/Light/Heavy
        {'name': 'temp_lag_1d', 'type': 'num'},
        {'name': 'temp_change_24h', 'type': 'num'},
        {'name': 'is_extreme_heat', 'type': 'cat'},       # Binary flag
        {'name': 'is_extreme_cold', 'type': 'cat'},       # Binary flag
        {'name': 'is_high_humidity', 'type': 'cat'},      # Binary flag
    ]
}

# Helper function to extract features by type from ALL_FEATURES_CONFIG
def get_features_by_type(feature_type: str = 'all') -> List[str]:
    """
    Extract feature names from ALL_FEATURES_CONFIG.
    
    Args:
        feature_type: 'all', 'num' (numeric), or 'cat' (categorical)
        
    Returns:
        List of feature names
    """
    features = []
    for ws_features in ALL_FEATURES_CONFIG.values():
        for feat in ws_features:
            if feature_type == 'all':
                features.append(feat['name'])
            elif feat['type'] == feature_type:
                features.append(feat['name'])
    return features

# --- 5. LOGGING CONFIG ---
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_to_console': True,
    'log_file': DATA_DIRS['logs'] / 'pipeline.log',
}

# --- 6. PERFORMANCE & MODELING CONFIG (Tối ưu cho i5-14600K + 32GB RAM) ---
PERFORMANCE_CONFIG = {
    'use_polars': False,      # Pandas ổn định hơn cho các phép toán phức tạp
    'memory_limit_gb': 24,    # Dành 24GB cho pipeline
    'parallel_threads': 12,   # Tận dụng 12 luồng (trên 20 luồng)
}

# Quantile levels - CHỈ TRAIN 5 QUANTILES (Q05, Q25, Q50, Q75, Q95)
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# Model types to train
MODEL_TYPES = ['lightgbm']  # Mặc định chỉ train LightGBM để test nhanh. Có thể thêm: ['lightgbm', 'catboost', 'random_forest']

# LightGBM hyperparameters (Tối ưu cho tốc độ/hiệu năng trên 32GB RAM)
# STABILITY PARAMETERS: Added for improved model stability and reproducibility
LIGHTGBM_PARAMS = {
    'n_estimators': 600,
    'learning_rate': 0.03,
    'num_leaves': 48,
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': PERFORMANCE_CONFIG['parallel_threads'],
    'verbose': -1,
    
    # Stability parameters (CRITICAL for reproducible and stable models)
    # Prevents "No further splits with positive gain, best gain: -inf" warnings
    # Ensures reproducible results across multiple runs
    'force_col_wise': True,          # Force column-wise tree building (more stable than row-wise)
    'min_data_in_leaf': 20,          # Minimum data points in leaf (prevents overfitting, improves stability)
    'min_gain_to_split': 0.001,      # Minimum gain to split (prevents -inf gain warnings)
    'min_split_gain': 0.001,         # Minimum gain to make a split (alias, prevents -inf gain)
    'max_bin': 255,                  # Maximum number of bins (default 255, lower = more stable but slower)
    # Note: colsample_bytree (line 261) already controls feature fraction (0.7), no duplicate needed
    'bagging_freq': 1,               # Frequency for bagging (1 = every iteration, improves stability)
    'min_child_samples': 20,         # Minimum number of data in child leaf (alias for min_data_in_leaf)
    'subsample_for_bin': 200000,     # Number of samples for constructing bins (higher = more stable)
    'max_delta_step': 0.0,           # Maximum delta step (0.0 = no constraint, can set to 1-10 for stability)
    'deterministic': True,           # Enable deterministic mode (ensures reproducibility)
    'force_row_wise': False,         # Disable row-wise (use column-wise for stability)
    'feature_pre_filter': False,     # Disable feature pre-filtering for stability
    'num_threads': PERFORMANCE_CONFIG['parallel_threads'],  # Explicit thread control
}

# CatBoost hyperparameters (OPTIONAL - only used if CatBoost is installed and enabled)
# CatBoost is an optional alternative to LightGBM. To use it:
# 1. Install CatBoost: pip install catboost (or see requirements.txt for Windows instructions)
# 2. Add 'catboost' to MODEL_TYPES list in config
# Note: LightGBM is the default and recommended model for this project.
CATBOOST_PARAMS = {
    'iterations': 600,
    'learning_rate': 0.03,
    'depth': 10,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'thread_count': PERFORMANCE_CONFIG['parallel_threads'],
    'verbose': False,
}

# Random Forest hyperparameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': PERFORMANCE_CONFIG['parallel_threads'],
    'verbose': 0,
}

# Model configurations dictionary
MODEL_CONFIGS = {
    'lightgbm': {
        'class': 'LGBMRegressor',
        'params': LIGHTGBM_PARAMS,
        'quantile_support': True,  # Hỗ trợ quantile regression trực tiếp
    },
    'catboost': {
        'class': 'CatBoostRegressor',
        'params': CATBOOST_PARAMS,
        'quantile_support': False,  # Cần wrapper
        'optional': True,  # CatBoost is optional - requires separate installation
    },
    'random_forest': {
        'class': 'RandomForestRegressor',
        'params': RANDOM_FOREST_PARAMS,
        'quantile_support': False,  # Cần wrapper
    },
}

# Training configuration
TRAINING_CONFIG = {
    'quantiles': QUANTILES,
    'model_types': MODEL_TYPES,  # List các model types để train
    'hyperparameters': LIGHTGBM_PARAMS,  # Default (backward compatible)
    'train_test_split': {
        'method': 'time_based',
        'cutoff_percentile': 0.8, # 80% train, 20% test
    },
    'save_shap_values': True,  # Lưu SHAP values sau khi train
    'shap_sample_size': 1000,  # Số lượng samples để tính SHAP (None = all)
}

# SHAP configuration
SHAP_CONFIG = {
    'enabled': True,
    'sample_size': 1000,  # Số lượng samples để tính SHAP trong prediction
    'max_display_features': 20,  # Số lượng features tối đa để hiển thị trong SHAP plots
    'save_plots': True,
    'plot_type': 'summary',  # 'summary', 'waterfall', 'bar', 'beeswarm'
}

# --- 6. DATA QUALITY & MONITORING CONFIG ---
DATA_QUALITY_CONFIG = {
    'enable_great_expectations': True,
    'enable_drift_detection': True,
    'quality_thresholds': {
        'excellent': 95,
        'good': 85,
        'fair': 70,
        'poor': 50
    },
    'drift_thresholds': {
        'mean_drift_sigma': 2.0,  # Standard deviations
        'null_drift_percentage': 0.1,  # 10% change
        'distribution_drift_threshold': 0.1  # KS test threshold
    },
    'alerting': {
        'enable_email_alerts': False,
        'enable_slack_alerts': False,
        'alert_on_quality_below': 70,
        'alert_on_drift_detected': True
    }
}

# Caching configuration
CACHE_CONFIG = {
    'cache_dir': PROJECT_ROOT / 'cache',
    'max_size_gb': 20.0,  # Increased for larger datasets
    'default_expiry_hours': 48,
    'enable_incremental_processing': True,
    'chunk_size': 50000
}

# --- 7. PERFORMANCE & MONITORING CONFIG (Enhanced) ---
PERFORMANCE_CONFIG.update({
    'enable_caching': True,
    'enable_parallel_processing': True,
    'memory_monitoring': True,
    'performance_logging': True,
    'retry_config': {
        'max_retries': 3,
        'retry_delay_seconds': 60,
        'exponential_backoff': True
    }
})

# --- 8. HELPER FUNCTIONS ---
def get_model_config(quantile: float, model_type: str = 'lightgbm') -> dict[str, Any]:
    """Lấy config model cho 1 quantile và model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Model type '{model_type}' not supported. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[model_type]
    config = model_config['params'].copy()
    
    # Thêm quantile-specific config
    if model_config['quantile_support']:
        # LightGBM hỗ trợ quantile regression trực tiếp
        config['objective'] = 'quantile'
        config['alpha'] = quantile
    else:
        # Các model khác sẽ dùng wrapper (QuantileRegressor từ sklearn)
        config['quantile'] = quantile
    
    return config

def get_data_directory() -> Path:
    """Luôn dùng 'raw_data'."""
    return DATA_DIRS['raw_data']

def ensure_directories() -> None:
    """Tạo các thư mục nếu chưa tồn tại."""
    for dir_path in DATA_DIRS.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    (DATA_DIRS['reports'] / 'metrics').mkdir(parents=True, exist_ok=True)
    (DATA_DIRS['reports'] / 'dashboard').mkdir(parents=True, exist_ok=True)
    (DATA_DIRS['reports'] / 'shap_values').mkdir(parents=True, exist_ok=True)

def setup_logging(level: str = None, log_to_file: bool = None) -> None:
    """Cài đặt logging tập trung."""
    import logging

    level = level or LOGGING_CONFIG.get('level', 'INFO')  # pyright: ignore[reportUndefinedVariable]
    log_format = LOGGING_CONFIG.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # pyright: ignore[reportUndefinedVariable]
    log_to_file = log_to_file if log_to_file is not None else LOGGING_CONFIG.get('log_to_file', True)  # pyright: ignore[reportUndefinedVariable]
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Cấu hình root logger
    logging.basicConfig(level=numeric_level, format=log_format, handlers=[], force=True)

    if LOGGING_CONFIG.get('log_to_console', True):  # pyright: ignore[reportUndefinedVariable]
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)

    if log_to_file:
        log_file = LOGGING_CONFIG.get('log_file', DATA_DIRS['logs'] / 'pipeline.log')  # pyright: ignore[reportUndefinedVariable]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w') # 'w' = write new log each run
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)

# --- 9. CONFIG VALIDATION ---
def validate_dataset_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate dataset configuration and return list of errors.
    
    Args:
        config: Dataset configuration dictionary
        
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Required fields
    required_fields = ['name', 'temporal_unit', 'time_column', 'target_column', 'groupby_keys']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate temporal_unit
    if 'temporal_unit' in config:
        valid_units = ['hour', 'day', 'week', 'month']
        if config['temporal_unit'] not in valid_units:
            errors.append(f"Invalid temporal_unit: {config['temporal_unit']}. Must be one of: {valid_units}")
    
    # Validate groupby_keys
    if 'groupby_keys' in config:
        if not isinstance(config['groupby_keys'], list) or len(config['groupby_keys']) < 2:
            errors.append("groupby_keys must be a list with at least 2 elements")
    
    # Validate time_column exists in groupby_keys
    if 'time_column' in config and 'groupby_keys' in config:
        if config['time_column'] not in config['groupby_keys']:
            errors.append(f"time_column '{config['time_column']}' must be in groupby_keys")
    
    # Validate lag_periods
    if 'lag_periods' in config:
        if not isinstance(config['lag_periods'], list) or len(config['lag_periods']) == 0:
            errors.append("lag_periods must be a non-empty list")
        elif not all(isinstance(x, (int, float)) and x > 0 for x in config['lag_periods']):
            errors.append("lag_periods must contain positive numbers")
    
    # Validate rolling_windows
    if 'rolling_windows' in config:
        if not isinstance(config['rolling_windows'], list):
            errors.append("rolling_windows must be a list")
        elif not all(isinstance(x, int) and x > 0 for x in config['rolling_windows']):
            errors.append("rolling_windows must contain positive integers")
    
    # Validate boolean flags
    boolean_flags = ['has_relational', 'has_stockout', 'has_weather', 'has_price_promo', 
                     'has_behavior', 'has_intraday_patterns']
    for flag in boolean_flags:
        if flag in config and not isinstance(config[flag], bool):
            errors.append(f"{flag} must be a boolean")
    
    return errors

def validate_training_config() -> List[str]:
    """
    Validate training configuration.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate quantiles
    if not isinstance(QUANTILES, list) or len(QUANTILES) == 0:
        errors.append("QUANTILES must be a non-empty list")
    elif not all(0 < q < 1 for q in QUANTILES):
        errors.append("All quantiles must be between 0 and 1")
    
    # Validate model types
    if not isinstance(MODEL_TYPES, list) or len(MODEL_TYPES) == 0:
        errors.append("MODEL_TYPES must be a non-empty list")
    else:
        for model_type in MODEL_TYPES:
            if model_type not in MODEL_CONFIGS:
                errors.append(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    
    # Validate train_test_split
    if 'train_test_split' in TRAINING_CONFIG:
        split_config = TRAINING_CONFIG['train_test_split']
        if 'cutoff_percentile' in split_config:
            cutoff = split_config['cutoff_percentile']
            if not isinstance(cutoff, (int, float)) or not (0 < cutoff < 1):
                errors.append("cutoff_percentile must be between 0 and 1")
    
    return errors

def validate_performance_config() -> List[str]:
    """
    Validate performance configuration.
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate memory_limit_gb
    if 'memory_limit_gb' in PERFORMANCE_CONFIG:
        mem_limit = PERFORMANCE_CONFIG['memory_limit_gb']
        if not isinstance(mem_limit, (int, float)) or mem_limit <= 0:
            errors.append("memory_limit_gb must be a positive number")
    
    # Validate parallel_threads
    if 'parallel_threads' in PERFORMANCE_CONFIG:
        threads = PERFORMANCE_CONFIG['parallel_threads']
        if not isinstance(threads, int) or threads <= 0:
            errors.append("parallel_threads must be a positive integer")
    
    return errors

def validate_all_configs() -> Dict[str, List[str]]:
    """
    Validate all configurations.
    
    Returns:
        Dictionary mapping config section names to lists of errors
    """
    all_errors = {}
    
    # Validate active dataset config
    try:
        active_config = get_dataset_config()
        dataset_errors = validate_dataset_config(active_config)
        if dataset_errors:
            all_errors['dataset_config'] = dataset_errors
    except Exception as e:
        all_errors['dataset_config'] = [f"Error loading dataset config: {e}"]
    
    # Validate training config
    training_errors = validate_training_config()
    if training_errors:
        all_errors['training_config'] = training_errors
    
    # Validate performance config
    performance_errors = validate_performance_config()
    if performance_errors:
        all_errors['performance_config'] = performance_errors
    
    return all_errors

def assert_config_valid() -> None:
    """
    Assert that all configurations are valid. Raises ValueError if invalid.
    
    Raises:
        ValueError: If any configuration is invalid
    """
    errors = validate_all_configs()
    if errors:
        error_messages = []
        for section, section_errors in errors.items():
            error_messages.append(f"{section}: {', '.join(section_errors)}")
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(error_messages))