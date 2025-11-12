"""
WS5: Stockout Recovery Module (Optional - FreshRetail Only)
===========================================================
Recovers latent demand during stockout periods and adds stockout context features.

This module is ONLY loaded when dataset config has 'has_stockout': True
(FreshRetail dataset). Dunnhumby dataset skips this entirely.

FUNCTIONS:
- recover_latent_demand(): Estimates demand during stockout periods
- add_stockout_features(): Adds stockout context features for modeling

DEPENDENCIES:
- Requires 'is_stockout' column in the dataframe
- Works best with hourly data (FreshRetail)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

# Import centralized config
try:
    from ..config import setup_logging, get_dataset_config

    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:  # pragma: no cover - fallback for standalone usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    def get_dataset_config() -> dict:
        return {
            'temporal_unit': 'hour',
            'time_column': 'hour_timestamp',
            'groupby_keys': ['product_id', 'store_id', 'hour_timestamp'],
            'target_column': 'sales_quantity',
        }


def recover_latent_demand(df: pd.DataFrame, dataset_config: Optional[dict] = None) -> pd.DataFrame:
    """
    Recover latent demand during stockout periods.

    When a product is out of stock, the recorded target column is 0, but there
    might have been demand that went unfulfilled. This function attempts to
    estimate that latent demand using various strategies.
    """
    logger.info("WS5: Starting latent demand recovery...")

    config = dataset_config or get_dataset_config()
    product_col, store_col, time_col = config['groupby_keys']
    target_col = config.get('target_column', 'SALES_VALUE')
    stockout_col = 'is_stockout'

    df_result = df.copy()

    required_cols = [stockout_col, target_col, product_col, store_col]
    missing_cols = [col for col in required_cols if col not in df_result.columns]
    if missing_cols:
        logger.warning(f"WS5: Missing required columns for latent demand recovery: {missing_cols}")
        df_result['latent_demand'] = 0.0
        return df_result

    if time_col not in df_result.columns:
        logger.warning(f"WS5: Time column '{time_col}' not found. Skipping latent demand recovery.")
        df_result['latent_demand'] = 0.0
        return df_result

    if config.get('temporal_unit') == 'hour' and not pd.api.types.is_datetime64_any_dtype(df_result[time_col]):
        try:
            df_result[time_col] = pd.to_datetime(df_result[time_col])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"WS5: Failed to parse '{time_col}' as datetime: {exc}")

    df_result['latent_demand'] = 0.0

    groups_processed = 0
    total_latent_demand = 0.0

    group_columns = [product_col, store_col]
    for _, group_df in df_result.groupby(group_columns, sort=False):
        group_sorted = (
            group_df.sort_values(time_col)
            .reset_index()
            .rename(columns={'index': '__orig_index__'})
        )

        if '_hour_of_day' not in group_sorted.columns and pd.api.types.is_datetime64_any_dtype(group_sorted[time_col]):
            group_sorted['_hour_of_day'] = group_sorted[time_col].dt.hour.astype('int16')

        stockout_mask = group_sorted[stockout_col] == 1
        if not stockout_mask.any():
            continue

        stockout_positions = stockout_mask[stockout_mask].index

        for row_idx in stockout_positions:
            latent_estimate = _estimate_latent_demand_at_point(
                group_sorted,
                row_idx,
                target_col=target_col,
                stockout_col=stockout_col,
                hour_col='_hour_of_day' if '_hour_of_day' in group_sorted.columns else None,
            )
            orig_idx = group_sorted.loc[row_idx, '__orig_index__']
            df_result.at[orig_idx, 'latent_demand'] = latent_estimate
            total_latent_demand += latent_estimate

        groups_processed += 1
        if groups_processed % 100 == 0:
            logger.info(f"WS5: Processed {groups_processed} product-store groups...")

    logger.info(f"WS5: Latent demand recovery complete")
    logger.info(f"WS5: Processed {groups_processed} product-store groups")
    logger.info(f"WS5: Total latent demand estimated: {total_latent_demand:.2f}")

    return df_result


def _estimate_latent_demand_at_point(
    group_df: pd.DataFrame,
    stockout_idx: int,
    *,
    target_col: str,
    stockout_col: str,
    hour_col: Optional[str] = None,
) -> float:
    """
    Estimate latent demand for a single stockout point using multiple strategies.
    """
    try:
        row = group_df.loc[stockout_idx]

        if hour_col and hour_col in group_df.columns:
            same_hour_mask = (
                (group_df[hour_col] == row[hour_col]) &
                (group_df[stockout_col] == 0) &
                (group_df.index != stockout_idx)
            )
            if same_hour_mask.sum() >= 3:
                historical_avg = group_df.loc[same_hour_mask, target_col].tail(10).mean()
                if not np.isnan(historical_avg) and historical_avg > 0:
                    return float(historical_avg)

        recent_mask = (
            (group_df.index < stockout_idx) &
            (group_df[stockout_col] == 0) &
            (group_df[target_col] > 0)
        )
        if recent_mask.sum() >= 3:
            recent_sales = group_df.loc[recent_mask, target_col].tail(7)
            if len(recent_sales) > 0:
                weights = np.exp(np.linspace(-1, 0, len(recent_sales)))
                weighted_avg = np.average(recent_sales, weights=weights)
                return float(weighted_avg)

        overall_avg = group_df.loc[group_df[stockout_col] == 0, target_col].mean()
        if not np.isnan(overall_avg) and overall_avg > 0:
            return float(overall_avg * 0.8)

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug(f"WS5: Error estimating latent demand at index {stockout_idx}: {exc}")

    return 0.1


def add_stockout_features(df: pd.DataFrame, dataset_config: Optional[dict] = None) -> pd.DataFrame:
    """
    Add stockout context features for better modeling.
    """
    logger.info("WS5: Adding stockout context features...")

    config = dataset_config or get_dataset_config()
    product_col, store_col, time_col = config['groupby_keys']
    target_col = config.get('target_column', 'SALES_VALUE')
    stockout_col = 'is_stockout'

    df_result = df.copy()

    if stockout_col not in df_result.columns:
        logger.warning("WS5: 'is_stockout' column not found, skipping stockout features")
        return df_result

    if time_col not in df_result.columns:
        logger.warning(f"WS5: Time column '{time_col}' not found, skipping stockout features")
        return df_result

    if config.get('temporal_unit') == 'hour' and not pd.api.types.is_datetime64_any_dtype(df_result[time_col]):
        try:
            df_result[time_col] = pd.to_datetime(df_result[time_col])
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"WS5: Failed to parse '{time_col}' as datetime: {exc}")

    frequency_window = 168 if config.get('temporal_unit') == 'hour' else 12
    expected_demand_window = 24 if config.get('temporal_unit') == 'hour' else max(config.get('rolling_windows', [4]))

    group_columns = [product_col, store_col]
    assign_columns = [
        'stockout_duration',
        'time_since_last_stockout',
        'time_to_next_stockout',
        'stockout_frequency',
        'post_stockout_bounce',
        'stockout_severity',
        'expected_demand',
    ]

    groups_processed = 0

    for _, group_df in df_result.groupby(group_columns, sort=False):
        group_copy = (
            group_df.sort_values(time_col)
            .reset_index()
            .rename(columns={'index': '__orig_index__'})
        )

        stockout_mask = group_copy[stockout_col].astype(bool)

        group_copy['stockout_duration'] = _calculate_consecutive_periods(stockout_mask)
        group_copy['time_since_last_stockout'] = _calculate_time_since_last_event(stockout_mask, forward=False)
        group_copy['time_to_next_stockout'] = _calculate_time_since_last_event(stockout_mask, forward=True)

        stockout_freq_series = stockout_mask.astype(int).rolling(window=frequency_window, min_periods=1).sum()
        group_copy['stockout_frequency'] = stockout_freq_series.astype('int32')

        if target_col in group_copy.columns:
            group_copy['expected_demand'] = (
                group_copy[target_col]
                .rolling(window=expected_demand_window, min_periods=3)
                .mean()
            )
        else:
            group_copy['expected_demand'] = 0.0

        group_copy['post_stockout_bounce'] = 0.0
        if target_col in group_copy.columns and 'latent_demand' in group_copy.columns:
            for pos in range(1, len(group_copy)):
                if not stockout_mask.iloc[pos] and stockout_mask.iloc[pos - 1]:
                    latent_value = group_copy.loc[pos - 1, 'latent_demand']
                    actual_value = group_copy.loc[pos, target_col]
                    if latent_value > 0:
                        group_copy.loc[pos, 'post_stockout_bounce'] = actual_value / latent_value

        group_copy['stockout_severity'] = 0.0
        if 'latent_demand' in group_copy.columns and target_col in group_copy.columns:
            expected = group_copy['expected_demand'].replace(0, np.nan)

            stockout_positions = stockout_mask & (group_copy['latent_demand'] > 0)
            group_copy.loc[stockout_positions, 'stockout_severity'] = (
                group_copy.loc[stockout_positions, 'latent_demand'] /
                expected.loc[stockout_positions]
            )

            normal_positions = (~stockout_mask) & (group_copy[target_col] > 0)
            group_copy.loc[normal_positions, 'stockout_severity'] = (
                group_copy.loc[normal_positions, target_col] /
                expected.loc[normal_positions]
            )

        for col in assign_columns:
            if col in group_copy.columns:
                df_result.loc[group_copy['__orig_index__'], col] = group_copy[col].fillna(0).values

        groups_processed += 1
        if groups_processed % 100 == 0:
            logger.info(f"WS5: Processed stockout features for {groups_processed} product-store groups...")

    feature_cols = ['latent_demand'] + assign_columns
    for col in feature_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna(0)

    n_new_features = len([col for col in feature_cols if col in df_result.columns])
    logger.info(f"WS5: Stockout features complete - processed {groups_processed} groups (Added {n_new_features} features)")

    return df_result


def _calculate_consecutive_periods(mask: pd.Series) -> pd.Series:
    """Calculate consecutive periods of True values in a boolean mask."""
    arr = mask.to_numpy(dtype=bool)
    consecutive = np.zeros_like(arr, dtype=int)
    streak = 0

    for idx, flag in enumerate(arr):
        if flag:
            streak += 1
        else:
            streak = 0
        consecutive[idx] = streak

    return pd.Series(consecutive, index=mask.index, name='stockout_duration')


def _calculate_time_since_last_event(mask: pd.Series, forward: bool = False) -> pd.Series:
    """Calculate time periods since last event (or until next event)."""
    arr = mask.to_numpy(dtype=bool)
    result = np.zeros_like(arr, dtype=int)

    if forward:
        next_event = -1
        for idx in range(len(arr) - 1, -1, -1):
            if arr[idx]:
                next_event = idx
                result[idx] = 0
            elif next_event != -1:
                result[idx] = next_event - idx
            else:
                result[idx] = 0
    else:
        last_event = -1
        for idx, flag in enumerate(arr):
            if flag:
                last_event = idx
                result[idx] = 0
            elif last_event != -1:
                result[idx] = idx - last_event
            else:
                result[idx] = 0

    return pd.Series(result, index=mask.index, name='time_delta')


# Backward compatibility aliases
recover_latent_demand = recover_latent_demand
add_stockout_features = add_stockout_features
