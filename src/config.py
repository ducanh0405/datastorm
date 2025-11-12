"""
Project Configuration (Phiên bản đầy đủ, modular)
=================================================
Quản lý config tập trung cho SmartGrocy.
"""
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List

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
    'model_features': DATA_DIRS['models'] / 'model_features.json',
    'model_metrics': DATA_DIRS['reports'] / 'metrics' / 'model_metrics.json',
    'models_dir': DATA_DIRS['models'],
    'reports_dir': DATA_DIRS['reports'],
    'dashboard_dir': DATA_DIRS['reports'] / 'dashboard',
    'predictions_test': DATA_DIRS['reports'] / 'predictions_test_set.csv',
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

def get_dataset_config(dataset_name=None):
    """Lấy config cho dataset đang hoạt động."""
    dataset = dataset_name or ACTIVE_DATASET
    if dataset not in DATASET_CONFIGS:
        raise KeyError(f"Dataset '{dataset}' not found. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset]

# --- 4. FEATURE CONFIG (Modular) ---
# Ánh xạ tên toggle trong config tới danh sách features
# Điều này giúp _03_model_training.py tự động chọn features
ALL_FEATURES_CONFIG = {
    # WS1: Relational
    'relational': ['DEPARTMENT', 'COMMODITY_DESC', 'BRAND', 'MANUFACTURER'],
    
    # WS2: Time-Series (Base)
    'timeseries_base': [
        'sales_lag_1', 'sales_lag_4', 'sales_lag_8', 'sales_lag_12', # (Dunnhumby)
        'sales_quantity_lag_1', 'sales_quantity_lag_24', 'sales_quantity_lag_48', 'sales_quantity_lag_168', # (FreshRetail)
        'rolling_mean_4_lag_1', 'rolling_std_4_lag_1',
        'rolling_mean_8_lag_1', 'rolling_std_8_lag_1',
        'rolling_mean_12_lag_1', 'rolling_std_12_lag_1',
        'rolling_mean_24_lag_1', 'rolling_std_24_lag_1',
        'rolling_mean_168_lag_1', 'rolling_std_168_lag_1',
        'week_of_year', 'month_proxy', 'week_sin', 'week_cos', # (Calendar)
    ],
    
    # WS2: Intraday
    'intraday_patterns': [
        'hour_of_day', 'day_of_week', 'is_morning_peak', 
        'is_evening_peak', 'is_weekend', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ],

    # WS3: Behavior
    'behavior': ['total_views', 'total_addtocart', 'total_transactions', 
                 'rate_view_to_cart', 'rate_cart_to_buy', 'rate_view_to_buy', 
                 'days_since_last_action'],
    
    # WS4: Price/Promo
    'price_promo': ['base_price', 'total_discount', 'discount_pct', 
                    'is_on_display', 'is_on_mailer', 'is_on_retail_promo', 'is_on_coupon_promo'],
    
    # WS5: Stockout
    'stockout': ['latent_demand', 'stockout_duration', 'time_since_last_stockout', 
                 'time_to_next_stockout', 'stockout_frequency', 'stockout_severity'],
    
    # WS6: Weather
    'weather': ['temperature', 'precipitation', 'humidity', 'temp_category', 
                'is_rainy', 'rain_intensity', 'temp_lag_1d', 'temp_change_24h', 
                'is_extreme_heat', 'is_extreme_cold', 'is_high_humidity']
}

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

# Quantile levels
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# LightGBM hyperparameters (Tối ưu cho tốc độ/hiệu năng trên 32GB RAM)
LIGHTGBM_PARAMS = {
    'n_estimators': 600,      # Tăng số lượng cây
    'learning_rate': 0.03,
    'num_leaves': 48,         # Tăng
    'max_depth': 10,          # Tăng
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': PERFORMANCE_CONFIG['parallel_threads'], # Dùng 16 luồng
    'verbose': -1,
}

# Training configuration
TRAINING_CONFIG = {
    'quantiles': QUANTILES,
    'hyperparameters': LIGHTGBM_PARAMS,
    'train_test_split': {
        'method': 'time_based',
        'cutoff_percentile': 0.8, # 80% train, 20% test
    },
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
def get_model_config(quantile: float) -> dict[str, Any]:
    """Lấy config model cho 1 quantile."""
    config = LIGHTGBM_PARAMS.copy()
    config['objective'] = 'quantile'
    config['alpha'] = quantile
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