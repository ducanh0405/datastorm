"""
WS5: Stockout Recovery Module (TỐI ƯU SONG SONG)
=================================================
Recovers latent demand and adds stockout context features using parallel processing.
"""
import logging
import numpy as np
import pandas as pd

# Import parallel processing utilities
try:
    from ..utils.parallel_processing import parallel_groupby_apply
    from ..config import PERFORMANCE_CONFIG
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False

# Fallback logger
try:
    from ..config import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# --- Helper functions (chạy trên từng group) ---

def _estimate_latent_demand_group(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ước tính latent demand cho MỘT group (product-store).
    """
    group_result = group_df.copy()
    group_result['latent_demand'] = 0.0

    stockout_mask = group_result['is_stockout'] == 1
    if not stockout_mask.any():
        return group_result # Không có stockout

    stockout_indices = group_result[stockout_mask].index

    for idx in stockout_indices:
        # Strategy 1: Same hour of day, same day of week (nếu có)
        latent_estimate = 0.1 # default
        row = group_result.loc[idx]
        
        if 'hour_of_day' in group_result.columns and 'day_of_week' in group_result.columns:
            same_hour_dow_mask = (
                (group_result['hour_of_day'] == row['hour_of_day']) &
                (group_result['day_of_week'] == row['day_of_week']) &
                (group_result['is_stockout'] == 0) & # Chỉ non-stockout
                (group_result.index < idx) # Chỉ trong quá khứ
            )
            if same_hour_dow_mask.sum() >= 2: # Cần ít nhất 2 điểm
                historical_avg = group_result.loc[same_hour_dow_mask, 'sales_quantity'].mean()
                if not np.isnan(historical_avg) and historical_avg > 0:
                    latent_estimate = historical_avg

        # Strategy 2: Fallback - Trung bình 24h gần nhất (non-stockout)
        if latent_estimate == 0.1:
            recent_mask = (
                (group_result.index < idx) &
                (group_result.index >= idx - pd.Timedelta(hours=24)) & # 24h qua
                (group_result['is_stockout'] == 0)
            )
            if recent_mask.sum() > 0:
                recent_avg = group_result.loc[recent_mask, 'sales_quantity'].mean()
                if not np.isnan(recent_avg) and recent_avg > 0:
                    latent_estimate = recent_avg

        group_result.loc[idx, 'latent_demand'] = latent_estimate

    return group_result

def _add_stockout_features_group(group_df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm stockout features cho MỘT group (product-store).
    """
    group_result = group_df.copy()
    
    if 'is_stockout' not in group_result.columns:
        return group_result

    # Feature 1: Stockout duration
    stockout_mask = group_result['is_stockout'] == 1
    consecutive_stockouts = stockout_mask.groupby((stockout_mask != stockout_mask.shift()).cumsum()).cumsum()
    group_result['stockout_duration'] = consecutive_stockouts
    
    # Feature 2: Time since last stockout
    group_result['time_since_last_stockout'] = stockout_mask.groupby((stockout_mask != stockout_mask.shift()).cumsum() == 0).cumsum()
    group_result.loc[stockout_mask, 'time_since_last_stockout'] = 0 # Reset về 0 trong khi stockout

    # Feature 3: Stockout frequency (rolling 1 week = 168 hours)
    group_result['stockout_frequency'] = group_result['is_stockout'].rolling(window=168, min_periods=1).sum()

    # Feature 4: Time to next stockout (khó tính song song, có thể bỏ qua)
    group_result['time_to_next_stockout'] = 0 
    
    # Feature 5: Stockout severity
    if 'latent_demand' in group_result.columns:
        expected_demand = group_result['sales_quantity'].rolling(window=24, min_periods=1).mean().shift(1).fillna(0)
        group_result['stockout_severity'] = group_result['latent_demand'] / (expected_demand + 1e-6)
    
    return group_result

# --- Hàm chính (gọi từ pipeline) ---

def recover_latent_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Parallel) Ước tính latent demand cho tất cả group.
    """
    logger.info("WS5: Starting latent demand recovery (Parallel)...")
    if not HAS_PARALLEL:
        logger.warning("WS5: Parallel processing not available. Falling back to slow loop.")
        # Thêm logic loop ở đây nếu cần
        return df

    n_jobs = PERFORMANCE_CONFIG.get('parallel_threads', -1)
    df_result = parallel_groupby_apply(
        df,
        group_cols=['product_id', 'store_id'],
        func=_estimate_latent_demand_group,
        n_jobs=n_jobs,
        verbose=5
    )
    
    total_latent = df_result['latent_demand'].sum()
    logger.info(f"WS5: Latent demand recovery complete. Total estimated: {total_latent:.2f}")
    return df_result

def add_stockout_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Parallel) Thêm stockout context features cho tất cả group.
    """
    logger.info("WS5: Adding stockout context features (Parallel)...")
    if not HAS_PARALLEL:
        logger.warning("WS5: Parallel processing not available. Falling back to slow loop.")
        return df
        
    n_jobs = PERFORMANCE_CONFIG.get('parallel_threads', -1)
    df_result = parallel_groupby_apply(
        df,
        group_cols=['product_id', 'store_id'],
        func=_add_stockout_features_group,
        n_jobs=n_jobs,
        verbose=5
    )
    
    # Fill NaNs
    feature_cols = ['stockout_duration', 'time_since_last_stockout', 'stockout_frequency', 'stockout_severity']
    for col in feature_cols:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna(0)

    logger.info("WS5: Stockout features complete.")
    return df_result