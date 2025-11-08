"""
WS4: Price & Promotion Features
================================
Creates price and promotion features from transaction and causal data.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _clean_causal_data(df_causal: pd.DataFrame) -> pd.DataFrame:
    """
    Internal function: Cleans causal_data.csv for promotion features.
    
    Args:
        df_causal: Raw causal data dataframe
    
    Returns:
        Cleaned causal dataframe with promotion flags
    """
    logging.info("[WS4] Cleaning causal data (promotions)...")
    
    # Handle both uppercase and lowercase column names
    df_causal = df_causal.copy()
    
    # Standardize column names to uppercase
    df_causal.columns = df_causal.columns.str.upper()
    
    # Convert data types
    if 'DISPLAY' in df_causal.columns:
        df_causal['DISPLAY'] = df_causal['DISPLAY'].astype(str)
    if 'MAILER' in df_causal.columns:
        df_causal['MAILER'] = df_causal['MAILER'].astype(str)
    
    # Rename columns to avoid conflicts when merging
    rename_map = {}
    if 'DISPLAY' in df_causal.columns:
        rename_map['DISPLAY'] = 'promo_display_type'
    if 'MAILER' in df_causal.columns:
        rename_map['MAILER'] = 'promo_mailer_type'
    
    if rename_map:
        df_causal = df_causal.rename(columns=rename_map)
    
    # Create binary flags
    if 'promo_display_type' in df_causal.columns:
        df_causal['is_on_display'] = (df_causal['promo_display_type'] != '0').astype(int)
    if 'promo_mailer_type' in df_causal.columns:
        df_causal['is_on_mailer'] = (df_causal['promo_mailer_type'] != '0').astype(int)
    
    # Keep only necessary columns for merge
    # Key columns for Dunnhumby causal: 'STORE_ID', 'PRODUCT_ID', 'WEEK_NO'
    causal_features = ['STORE_ID', 'PRODUCT_ID', 'WEEK_NO']
    if 'is_on_display' in df_causal.columns:
        causal_features.append('is_on_display')
    if 'is_on_mailer' in df_causal.columns:
        causal_features.append('is_on_mailer')
    
    # Remove duplicates if any
    df_causal_clean = df_causal[causal_features].drop_duplicates()
    
    logging.info(f"  Cleaned causal data: {len(df_causal_clean):,} rows")
    return df_causal_clean

def _clean_transaction_data(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal function: Creates price/promotion features from transaction data.
    
    Args:
        master_df: Master dataframe with transaction columns
    
    Returns:
        Dataframe with price and promotion features added
    """
    logging.info("[WS4] Creating price/promotion features from transaction data...")
    
    # These columns come from transaction_data.csv
    price_cols = ['SALES_VALUE', 'RETAIL_DISC', 'COUPON_DISC']
    
    # Check which columns exist
    existing_price_cols = [col for col in price_cols if col in master_df.columns]
    
    if not existing_price_cols:
        logging.warning("SKIPPING WS4 price features: No price columns found in master_df")
        return master_df
    
    # Fill NaNs (if any)
    for col in existing_price_cols:
        master_df[col] = master_df[col].fillna(0)
    
    # 1. Calculate base price
    # Base price = (Sales value - total discounts)
    # Note: Dunnhumby discounts are NEGATIVE numbers, so we subtract them
    if 'SALES_VALUE' in master_df.columns:
        discount_sum = 0
        if 'RETAIL_DISC' in master_df.columns:
            discount_sum += master_df['RETAIL_DISC']
        if 'COUPON_DISC' in master_df.columns:
            discount_sum += master_df['COUPON_DISC']
        
        master_df['base_price'] = master_df['SALES_VALUE'] - discount_sum
    
    # 2. Create discount percentage feature
    # (Avoid division by zero)
    if 'RETAIL_DISC' in master_df.columns and 'COUPON_DISC' in master_df.columns:
        master_df['total_discount'] = (master_df['RETAIL_DISC'] + master_df['COUPON_DISC']).abs()
        if 'base_price' in master_df.columns:
            master_df['discount_pct'] = master_df['total_discount'] / (master_df['base_price'] + 1e-6)
    
    # 3. Create binary flags
    if 'RETAIL_DISC' in master_df.columns:
        master_df['is_on_retail_promo'] = (master_df['RETAIL_DISC'] < 0).astype(int)
    if 'COUPON_DISC' in master_df.columns:
        master_df['is_on_coupon_promo'] = (master_df['COUPON_DISC'] < 0).astype(int)
    
    logging.info("  Created price/promotion features from transaction data")
    return master_df


def add_price_promotion_features(
    master_df: pd.DataFrame,
    dataframes_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Main function for Workstream 4: Price & Promotion Features.
    
    Creates price and promotion features from:
    1. Transaction data (base price, discounts, promo flags)
    2. Causal data (display, mailer promotions)
    
    Args:
        master_df: Master dataframe from transaction_data
        dataframes_dict: Dictionary containing 'causal_data' (optional)
    
    Returns:
        Dataframe with price and promotion features added
    
    Raises:
        pd.errors.MergeError: If merge operations fail
    """
    # 1. Process features from Master Table (from transaction_data)
    master_df = _clean_transaction_data(master_df)

    # 2. Process and merge Causal data (Promotions)
    if 'causal_data' not in dataframes_dict:
        logging.warning(
            "SKIPPING WS4 (Causal): 'causal_data' not found in dataframes_dict. "
            "Returning with transaction-based features only."
        )
        return master_df
    
    try:
        df_causal_clean = _clean_causal_data(dataframes_dict['causal_data'])
    except Exception as e:
        logging.error(f"ERROR cleaning causal data: {e}. Continuing without causal features.")
        return master_df
    
    # 3. Merge into Master Table
    # Key columns for Dunnhumby: 'STORE_ID', 'PRODUCT_ID', 'WEEK_NO'
    merge_keys = ['STORE_ID', 'PRODUCT_ID', 'WEEK_NO']
    
    missing_keys = [key for key in merge_keys if key not in master_df.columns]
    if missing_keys:
        logging.warning(
            f"SKIPPING WS4 merge: Missing required columns in master_df: {missing_keys}. "
            "Returning with transaction-based features only."
        )
        return master_df
    
    # Verify merge keys exist in causal data
    missing_causal_keys = [key for key in merge_keys if key not in df_causal_clean.columns]
    if missing_causal_keys:
        logging.warning(
            f"SKIPPING WS4 merge: Missing required columns in causal_data: {missing_causal_keys}. "
            "Returning with transaction-based features only."
        )
        return master_df
    
    try:
        original_rows = master_df.shape[0]
        master_df = pd.merge(master_df, df_causal_clean, on=merge_keys, how='left')
        
        if master_df.shape[0] != original_rows:
            logging.error(
                f"ERROR (WS4): Merge changed row count: {original_rows} -> {master_df.shape[0]}. "
                "Possible row explosion in causal data!"
            )
        
        # Fill 0 for products/weeks not in causal file (meaning no promotion)
        if 'is_on_display' in master_df.columns:
            master_df['is_on_display'] = master_df['is_on_display'].fillna(0).astype(int)
        if 'is_on_mailer' in master_df.columns:
            master_df['is_on_mailer'] = master_df['is_on_mailer'].fillna(0).astype(int)
        
        logging.info("OK. WS4 (Price/Promotion) integration complete.")
        
    except pd.errors.MergeError as e:
        logging.error(f"ERROR in WS4 merge operation: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in WS4 merge: {e}", exc_info=True)
        raise

    return master_df