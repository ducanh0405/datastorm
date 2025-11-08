"""
WS3: Behavioral Features
=========================
Creates user behavior features from clickstream data.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _process_clickstream_logs(df_events):
    """
    Internal function: Implements logic from 'EDA_Data_Preprocess.ipynb'.
    Cleans data, processes timestamps, and prepares log.
    """
    logging.info("[WS3] Starting behavior log processing (clickstream)...")
    
    # Assume df_events has columns: 'timestamp', 'visitorid', 'event', 'itemid'
    if 'timestamp' in df_events.columns:
        df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms')  # Assume timestamp is in ms

    # Handle NaNs (if any)
    df_events = df_events.dropna(subset=['visitorid', 'itemid'])
    
    logging.info(f"[WS3] Log processing complete. Total events: {len(df_events)}")
    return df_events

def _create_user_features(df_logs):
    """
    Internal function: Implements logic from 'Feature_Engineering.ipynb'.
    Creates feature table at user-level.
    """
    logging.info("[WS3] Starting behavior feature engineering...")
    
    # 1. Create basic features (example from your PoC)
    # This is the conversion funnel calculation logic
    user_features = df_logs.pivot_table(
        index='visitorid', 
        columns='event', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Rename columns if needed (e.g., 'addtocart' -> 'total_addtocart')
    user_features = user_features.rename(columns={
        'view': 'total_views',
        'addtocart': 'total_addtocart',
        'transaction': 'total_transactions'
    })
    
    # 2. Create conversion rate features
    # View -> Add to cart rate
    user_features['rate_view_to_cart'] = user_features['total_addtocart'] / (user_features['total_views'] + 1e-6)
    
    # Add to cart -> Purchase rate
    user_features['rate_cart_to_buy'] = user_features['total_transactions'] / (user_features['total_addtocart'] + 1e-6)
    
    # View -> Purchase rate (overall conversion rate)
    user_features['rate_view_to_buy'] = user_features['total_transactions'] / (user_features['total_views'] + 1e-6)

    # 3. Create time-based features (session-based)
    # (This is an example, you will replace with more complex logic from your notebook)
    if 'timestamp' in df_logs.columns:
        time_stats = df_logs.groupby('visitorid')['timestamp'].agg(['min', 'max'])
        time_stats['session_duration_days'] = (time_stats['max'] - time_stats['min']).dt.total_seconds() / (60 * 60 * 24)
        user_features = user_features.join(time_stats['session_duration_days'], how='left')

    # 4. Create feature for time since last interaction
    # (If timestamp column exists in df_logs)
    if 'timestamp' in df_logs.columns:
        latest_timestamp = df_logs['timestamp'].max()
        last_interaction = df_logs.groupby('visitorid')['timestamp'].max()
        user_features['days_since_last_action'] = (latest_timestamp - last_interaction).dt.total_seconds() / (60 * 60 * 24)

    logging.info(f"[WS3] Feature engineering complete. Number of users: {len(user_features)}")
    return user_features


# ===================================================================
# MAIN FUNCTION (CALLED BY _02_feature_enrichment.py)
# ===================================================================

def add_behavioral_features(master_df, dataframes_dict):
    """
    Master function for Workstream 3.
    Takes Master Table and dictionary of raw data (especially 'clickstream_log').

    Creates user behavior features and merges them into Master Table.
    """

    # 1. Check if behavior data exists
    required_keys = ['clickstream_log']  # Assume file name is 'clickstream_log'
    if not all(key in dataframes_dict for key in required_keys):
        logging.warning("SKIPPING Workstream 3: Missing clickstream data.")
        return master_df

    # 2. Process clickstream data
    df_events = dataframes_dict['clickstream_log'].copy()
    df_processed = _process_clickstream_logs(df_events)

    # 3. Create user features
    user_features = _create_user_features(df_processed)

    # 4. Merge into Master Table
    # Assume master_df has column 'visitorid' or 'customer_id' for merging
    merge_keys = ['visitorid']  # Change if column name is different
    if all(key in master_df.columns for key in merge_keys):
        original_rows = master_df.shape[0]
        master_df = pd.merge(master_df, user_features, on=merge_keys, how='left')

        # Fill NaN for users without behavior data
        numeric_cols = user_features.select_dtypes(include=[np.number]).columns
        master_df[numeric_cols] = master_df[numeric_cols].fillna(0)

        logging.info("OK. Workstream 3 (Behavior) integration successful.")
    else:
        logging.warning("SKIPPING WS3 merge: Merge keys not found in Master Table.")

    return master_df