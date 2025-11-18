"""
WS6: Weather Features for FreshRetailNet-50K
=============================================

Weather data integration and feature creation for FreshRetail dataset.
Only applicable when dataset config has 'has_weather': True

FUNCTIONS:
- merge_weather_data(): Merge weather data onto sales dataframe
- create_weather_features(): Create weather-related features for modeling
- load_weather_data(): Load weather data from file (optional helper)

DEPENDENCIES:
- Requires weather data with columns: store_id, date, temperature, precipitation, humidity
- Works with hourly sales data (FreshRetail)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Import centralized config
try:
    from src.config import get_dataset_config, setup_logging

    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:  # pragma: no cover - fallback when config unavailable
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def get_dataset_config() -> dict:
        return {
            'temporal_unit': 'hour',
            'time_column': 'hour_timestamp',
            'groupby_keys': ['product_id', 'store_id', 'hour_timestamp'],
        }


def load_weather_data(data_dir=None) -> pd.DataFrame:
    """
    Load weather data from file (helper function).

    Args:
        data_dir: Directory containing weather data file

    Returns:
        Weather DataFrame with columns: store_id, date, temperature, precipitation, humidity
    """
    if data_dir is None:
        try:
            from src.config import DATA_DIRS
            data_dir = DATA_DIRS['raw_data']
        except ImportError:
            data_dir = Path('data/2_raw')

    # Use Path object properly - convert string to Path if needed
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    weather_path = data_dir / "weather_data.csv"

    try:
        weather_df = pd.read_csv(weather_path)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        logger.info(f"Loaded weather data: {weather_df.shape}")
        return weather_df
    except FileNotFoundError:
        logger.warning(f"Weather data not found at {weather_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading weather data: {e}")
        return None


def merge_weather_data(
    sales_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    dataset_config: dict | None = None,
) -> pd.DataFrame:
    """
    Merge weather data onto sales dataframe based on store_id and date.

    Args:
        sales_df: Sales data with hour_timestamp column
        weather_df: Weather data with (store_id, date) columns

    Returns:
        Merged dataframe with weather columns added
    """
    if weather_df is None:
        logger.warning("Weather data is None, skipping merge")
        return sales_df

    config = dataset_config or get_dataset_config()
    groupby_keys = config['groupby_keys']
    if len(groupby_keys) < 2:
        logger.warning("WS6: Invalid groupby_keys in dataset config. Skipping merge.")
        return sales_df

    _product_col, store_col, time_col = groupby_keys[0], groupby_keys[1], config['time_column']

    logger.info(f"Merging weather data: {weather_df.shape}")

    sales_df_copy = sales_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(sales_df_copy[time_col]):
        try:
            sales_df_copy[time_col] = pd.to_datetime(sales_df_copy[time_col])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"WS6: Failed to parse '{time_col}' as datetime: {exc}")
            return sales_df

    sales_df_copy['_weather_merge_date'] = sales_df_copy[time_col].dt.date

    weather_subset = weather_df.copy()
    if 'date' not in weather_subset.columns:
        logger.error("WS6: Weather dataframe missing 'date' column. Skipping merge.")
        return sales_df

    weather_subset['date'] = pd.to_datetime(weather_subset['date']).dt.date

    if store_col not in weather_subset.columns and 'store_id' in weather_subset.columns:
        weather_subset = weather_subset.rename(columns={'store_id': store_col})

    weather_cols = [store_col, 'date', 'temperature', 'precipitation']
    if 'humidity' in weather_subset.columns:
        weather_cols.append('humidity')

    weather_subset = weather_subset[weather_cols].copy()

    result = sales_df_copy.merge(
        weather_subset,
        left_on=[store_col, '_weather_merge_date'],
        right_on=[store_col, 'date'],
        how='left'
    )

    weather_fill_cols = [col for col in ['temperature', 'precipitation', 'humidity'] if col in result.columns]
    if weather_fill_cols:
        result[weather_fill_cols] = (
            result.groupby(store_col)[weather_fill_cols]
            .transform(lambda col: col.ffill().bfill())
        )

    null_counts = result[weather_fill_cols].isnull().sum() if weather_fill_cols else pd.Series(dtype=int)
    total_nulls = null_counts.sum() if not null_counts.empty else 0

    logger.info(f"Weather merge complete: {total_nulls} nulls remaining")
    if total_nulls > 0:
        logger.warning(f"Weather null breakdown: {dict(null_counts)}")

    result = result.drop(columns=['date', '_weather_merge_date'])

    return result


def create_weather_features(df: pd.DataFrame, dataset_config: dict | None = None) -> pd.DataFrame:
    """
    Create weather-related features for modeling.

    Features created:
    - Temperature categories (cold/mild/hot)
    - Rainfall intensity categories
    - Weather lags (previous day)
    - Temperature changes
    - Extreme weather flags

    Args:
        df: DataFrame with weather columns (temperature, precipitation, humidity)

    Returns:
        DataFrame with additional weather features
    """
    logger.info("Creating weather features...")

    config = dataset_config or get_dataset_config()
    product_col, store_col, time_col = config['groupby_keys']

    df_result = df.copy()

    if time_col not in df_result.columns:
        logger.warning(f"WS6: Time column '{time_col}' not found, skipping weather features")
        return df_result

    if not pd.api.types.is_datetime64_any_dtype(df_result[time_col]):
        try:
            df_result[time_col] = pd.to_datetime(df_result[time_col])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"WS6: Failed to parse '{time_col}' as datetime: {exc}")
            return df_result

    # Check for required weather columns
    required_weather_cols = ['temperature', 'precipitation']
    missing_weather = [col for col in required_weather_cols if col not in df_result.columns]

    if missing_weather:
        logger.warning(f"Missing weather columns: {missing_weather}, skipping weather features")
        return df_result

    # 1. Temperature categories
    df_result['temp_category'] = pd.cut(
        df_result['temperature'],
        bins=[-np.inf, 10, 25, np.inf],
        labels=[0, 1, 2],  # cold, mild, hot
        right=False
    ).astype('int8')

    # 2. Rainfall features
    df_result['is_rainy'] = (df_result['precipitation'] > 0).astype('int8')

    df_result['rain_intensity'] = pd.cut(
        df_result['precipitation'],
        bins=[0, 5, 20, np.inf],
        labels=[0, 1, 2],  # none/light, moderate, heavy
        right=False
    ).astype('int8')

    # 3. Weather lags (1 day ago = 24 hours for hourly data)
    sort_cols = [product_col, store_col, time_col]
    df_result = df_result.sort_values(sort_cols)

    df_result['temp_lag_1d'] = (
        df_result.groupby([product_col, store_col])['temperature']
        .shift(24)  # 24 hours = 1 day
        .astype('float32')
    )

    df_result['rain_lag_1d'] = (
        df_result.groupby([product_col, store_col])['is_rainy']
        .shift(24)
        .astype('float32')
    )

    # 4. Temperature change (24-hour change)
    df_result['temp_change_24h'] = (
        (df_result['temperature'] - df_result['temp_lag_1d'])
        .astype('float32')
    )

    # 5. Extreme weather flags
    df_result['is_extreme_heat'] = (df_result['temperature'] > 30).astype('int8')
    df_result['is_extreme_cold'] = (df_result['temperature'] < 5).astype('int8')

    # 6. Humidity-based features (if available)
    if 'humidity' in df_result.columns:
        df_result['is_high_humidity'] = (df_result['humidity'] > 80).astype('int8')
        df_result['humidity_category'] = pd.cut(
            df_result['humidity'],
            bins=[0, 40, 70, 100],
            labels=[0, 1, 2],  # dry, normal, humid
            right=False
        ).astype('int8')

        logger.info("Created 11 weather features (including humidity)")
    else:
        logger.info("Created 9 weather features (humidity data not available)")

    # Fill NaN values from lagging operations
    lag_cols = ['temp_lag_1d', 'rain_lag_1d', 'temp_change_24h']
    df_result[lag_cols] = df_result[lag_cols].fillna(0)
    if 'rain_lag_1d' in df_result.columns:
        df_result['rain_lag_1d'] = df_result['rain_lag_1d'].astype('int8')

    return df_result


# Backward compatibility aliases
merge_weather_data = merge_weather_data
create_weather_features = create_weather_features
