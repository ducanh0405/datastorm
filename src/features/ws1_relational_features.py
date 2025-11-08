"""
WS1: Relational Features
=========================
Enriches master dataframe with relational data from product and household demographics.
"""
import pandas as pd
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def enrich_relational_features(
    master_df: pd.DataFrame,
    dataframes_dict: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Enriches master dataframe with relational features from product and household data.
    
    Performs left joins on:
    - Product data (on PRODUCT_ID)
    - Household demographics (on household_key, if available)
    
    Args:
        master_df: Master dataframe with PRODUCT_ID column
        dataframes_dict: Dictionary containing 'product' and optionally 'hh_demographic' dataframes
    
    Returns:
        Enriched dataframe with product and household features added
    
    Raises:
        pd.errors.MergeError: If merge operations fail
    """
    logging.info("[WS1] Enriching relational features (Product, Household)...")

    # Check if required dataframes exist
    if 'product' not in dataframes_dict:
        logging.warning("SKIPPING WS1: 'product' dataframe not found in dataframes_dict.")
        return master_df

    df_prod = dataframes_dict['product']

    # Validate required columns exist
    if 'PRODUCT_ID' not in master_df.columns:
        logging.error("SKIPPING WS1: 'PRODUCT_ID' not found in master_df")
        return master_df
    
    if 'PRODUCT_ID' not in df_prod.columns:
        logging.error("SKIPPING WS1: 'PRODUCT_ID' not found in product dataframe")
        return master_df

    try:
        # Merge Product data
        original_shape = master_df.shape
        master_df = pd.merge(master_df, df_prod, on='PRODUCT_ID', how='left')

        # Check merge success
        merged_rows = master_df.shape[0] - original_shape[0]
        matched_products = master_df['MANUFACTURER'].notna().sum()

        logging.info(f"  Merged product data: {original_shape} -> {master_df.shape}")
        logging.info(f"  Product matching: {matched_products:,}/{len(master_df):,} rows have product info")

        if matched_products == 0:
            logging.error("  CRITICAL ERROR: No products matched! WS1 relational features failed.")
            logging.error("  SOLUTION: Recreate POC data with PRODUCT_ID matching validation")
            # Fill with defaults to prevent NaN issues downstream
            product_cols = ['MANUFACTURER', 'DEPARTMENT', 'BRAND', 'COMMODITY_DESC', 'SUB_COMMODITY_DESC', 'CURR_SIZE_OF_PRODUCT']
            for col in product_cols:
                if col in master_df.columns:
                    master_df[col] = master_df[col].fillna('Unknown')
            logging.info("  ✅ Filled missing product data with 'Unknown' defaults")
        elif matched_products < len(master_df) * 0.1:
            logging.warning(f"  ⚠️  Only {matched_products/len(master_df)*100:.1f}% products have info - limited coverage")
        else:
            logging.info(f"  ✅ Good product data coverage: {matched_products/len(master_df)*100:.1f}%")

    except pd.errors.MergeError as e:
        logging.error(f"ERROR in WS1 product merge: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in WS1 product merge: {e}", exc_info=True)
        raise

    # Merge Household Demographics (optional - only if available)
    if 'hh_demographic' in dataframes_dict:
        df_hh = dataframes_dict['hh_demographic']
        
        if 'household_key' in master_df.columns and 'household_key' in df_hh.columns:
            try:
                original_shape = master_df.shape
                master_df = pd.merge(master_df, df_hh, on='household_key', how='left')
                logging.info(f"  Merged household demographics: {original_shape} -> {master_df.shape}")
            except pd.errors.MergeError as e:
                logging.error(f"ERROR in WS1 household merge: {e}")
                # Don't raise - continue without household data
            except Exception as e:
                logging.warning(f"Unexpected error in WS1 household merge: {e}. Continuing without household data.")
        else:
            logging.info("  Skipping household merge: 'household_key' not found in required dataframes")
    else:
        logging.info("  Skipping household merge: 'hh_demographic' not found in dataframes_dict")

    logging.info("OK. WS1 (Relational) integration complete.")
    return master_df