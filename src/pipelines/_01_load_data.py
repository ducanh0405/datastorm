"""
Data Loading Module (Config-Driven)
=====================================
Loads raw data based on the ACTIVE_DATASET setting in config.py.
"""
import pandas as pd
import logging
from pathlib import Path
import sys
from typing import Dict

# Import centralized configuration
try:
    from src.config import setup_project_path, setup_logging, get_data_directory, get_dataset_config
    setup_project_path()
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

def _load_file(data_dir: Path, file_stem: str) -> pd.DataFrame:
    """Helper to load .parquet or .csv."""
    parquet_path = data_dir / f"{file_stem}.parquet"
    csv_path = data_dir / f"{file_stem}.csv"
    
    if parquet_path.exists():
        logger.info(f"  Loading {file_stem}.parquet...")
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        logger.info(f"  Loading {file_stem}.csv...")
        return pd.read_csv(csv_path)
    else:
        logger.warning(f"  File not found: {file_stem}.parquet/csv")
        return None

def _clean_raw_data(dataframes: Dict[str, pd.DataFrame], config: Dict) -> Dict[str, pd.DataFrame]:
    """
    Dọn dẹp dữ liệu thô ngay sau khi tải, trước khi xử lý.
    Xử lý các giá trị lỗi và outliers.
    
    Args:
        dataframes: Dictionary chứa các dataframes đã load
        config: Dataset configuration dictionary
        
    Returns:
        Dictionary chứa các dataframes đã được clean
    """
    logger.info("Cleaning raw data (handling errors and outliers)...")
    
    # Clean sales data
    if 'sales' in dataframes and dataframes['sales'] is not None:
        df = dataframes['sales']
        target_col = config.get('target_column')  # 'sales_quantity' hoặc 'SALES_VALUE'
        
        if target_col and target_col in df.columns:
            # Lọc (Filtering): Loại bỏ sales âm
            original_rows = len(df)
            df = df[df[target_col] >= 0].copy()
            if len(df) < original_rows:
                logger.warning(f"  Removed {original_rows - len(df)} rows with negative {target_col}.")
        else:
            logger.warning(f"  Target column '{target_col}' not found in sales data, skipping filtering.")
        
        # Ví dụ khác (nếu là Dunnhumby):
        if 'QUANTITY' in df.columns:
            original_rows = len(df)
            df = df[df['QUANTITY'] >= 0].copy()
            if len(df) < original_rows:
                logger.warning(f"  Removed {original_rows - len(df)} rows with negative QUANTITY.")
        
        dataframes['sales'] = df
    
    # Clean weather data
    if 'weather' in dataframes and dataframes['weather'] is not None:
        df_weather = dataframes['weather']
        
        # Cắt (Clipping): Giới hạn nhiệt độ và độ ẩm
        if 'temperature' in df_weather.columns:
            original_min = df_weather['temperature'].min()
            original_max = df_weather['temperature'].max()
            df_weather['temperature'] = df_weather['temperature'].clip(-20, 50)  # Giả định khoảng -20°C đến 50°C
            clipped_count = ((df_weather['temperature'] == -20) | (df_weather['temperature'] == 50)).sum()
            if clipped_count > 0:
                logger.info(f"  Clipped {clipped_count} temperature values (range: {original_min:.1f} to {original_max:.1f})")
        
        if 'humidity' in df_weather.columns:
            original_min = df_weather['humidity'].min()
            original_max = df_weather['humidity'].max()
            df_weather['humidity'] = df_weather['humidity'].clip(0, 100)  # 0% đến 100%
            clipped_count = ((df_weather['humidity'] == 0) | (df_weather['humidity'] == 100)).sum()
            if clipped_count > 0:
                logger.info(f"  Clipped {clipped_count} humidity values (range: {original_min:.1f} to {original_max:.1f})")
        
        dataframes['weather'] = df_weather
    
    logger.info("✓ Raw data cleaning complete.")
    return dataframes

def _load_freshretail_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Loads all data files for FreshRetailNet-50K."""
    logger.info("Loading FreshRetailNet-50K dataset...")
    dataframes = {}
    
    # 1. Sales (Bắt buộc)
    sales_df = _load_file(data_dir, 'sales_hourly') # Tên tệp giả định
    if sales_df is None:
        sales_df = _load_file(data_dir, 'freshretail_train') # Tên tệp từ file upload
    if sales_df is None:
        logger.error("FATAL: FreshRetail sales data not found (sales_hourly or freshretail_train).")
        sys.exit(1)
        
    # Convert dt column to hour_timestamp if needed
    if 'dt' in sales_df.columns and 'hour_timestamp' not in sales_df.columns:
        sales_df['hour_timestamp'] = pd.to_datetime(sales_df['dt'])
        logger.info("✓ Converted 'dt' to 'hour_timestamp'")

    # Convert sale_amount to sales_quantity if needed
    if 'sale_amount' in sales_df.columns and 'sales_quantity' not in sales_df.columns:
        sales_df['sales_quantity'] = sales_df['sale_amount']
        logger.info("✓ Converted 'sale_amount' to 'sales_quantity'")

    # 2. Stockout (Bắt buộc)
    stockout_df = _load_file(data_dir, 'stockout_labels')
    if stockout_df is None:
        logger.warning("Stockout labels not found, 'is_stockout' will be 0.")
        sales_df['is_stockout'] = 0
    else:
        # Merge stockout
        merge_keys = ['product_id', 'store_id', 'hour_timestamp']
        sales_df['hour_timestamp'] = pd.to_datetime(sales_df['hour_timestamp'])
        stockout_df['hour_timestamp'] = pd.to_datetime(stockout_df['hour_timestamp'])

        sales_df = sales_df.merge(stockout_df[merge_keys + ['is_stockout']], on=merge_keys, how='left')
        sales_df['is_stockout'] = sales_df['is_stockout'].fillna(0).astype('int8')
        logger.info("✓ Stockout labels merged.")
        
    dataframes['sales'] = sales_df

    # 3. Weather (Tùy chọn)
    dataframes['weather'] = _load_file(data_dir, 'weather_data')
    if dataframes['weather'] is not None:
         dataframes['weather']['date'] = pd.to_datetime(dataframes['weather']['date'])

    # 4. Products (Tùy chọn)
    dataframes['products'] = _load_file(data_dir, 'product_info')
    
    return dataframes

def _load_dunnhumby_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Loads all data files for Dunnhumby."""
    logger.info("Loading Dunnhumby dataset...")
    dataframes = {}
    
    # Tên tệp dựa trên các file WS bạn đã upload
    dataframes['transaction_data'] = _load_file(data_dir, 'transaction_data')
    dataframes['product'] = _load_file(data_dir, 'product')
    dataframes['hh_demographic'] = _load_file(data_dir, 'hh_demographic')
    dataframes['causal_data'] = _load_file(data_dir, 'causal_data')
    dataframes['clickstream_log'] = _load_file(data_dir, 'clickstream_log') # Giả định tên này
    
    if dataframes['transaction_data'] is None:
        logger.error("FATAL: Dunnhumby 'transaction_data.csv' not found.")
        sys.exit(1)
        
    # Đổi tên 'sales' để pipeline thống nhất
    dataframes['sales'] = dataframes.pop('transaction_data')
        
    return dataframes

def load_data() -> (Dict[str, pd.DataFrame], Dict):
    """
    Loads data based on the ACTIVE_DATASET in config.
    Returns the dictionary of dataframes and the config.
    """
    config = get_dataset_config()
    data_dir = get_data_directory()

    logger.info("=" * 70)
    logger.info(f"[PIPELINE STEP 1: LOAD DATA]")
    logger.info(f"Active Dataset: {config['name']}")
    logger.info(f"Data Directory: {data_dir}")
    logger.info("=" * 70)

    if config['name'] == 'FreshRetailNet-50K':
        dataframes = _load_freshretail_data(data_dir)
    elif config['name'] == 'Dunnhumby':
        dataframes = _load_dunnhumby_data(data_dir)
    else:
        raise ValueError(f"Unknown dataset name in config: {config['name']}")

    # Clean raw data before returning
    dataframes = _clean_raw_data(dataframes, config)

    logger.info(f"✅ Data loading complete. Loaded {len(dataframes)} dataframes: {list(dataframes.keys())}")
    return dataframes, config

if __name__ == "__main__":
    dataframes, config = load_data()
    logger.info(f"\nLoaded config for: {config['name']}")
    for key, df in dataframes.items():
        if df is not None:
            logger.info(f"  {key}: {df.shape}\n{df.head()}")