"""
WS1: Relational Features
=========================
Enriches master dataframe with relational data from product and household demographics.
"""
import logging

import pandas as pd

# Logger will be configured by parent pipeline
logger = logging.getLogger(__name__)


def enrich_relational_features(
    master_df: pd.DataFrame,
    dataframes_dict: dict[str, pd.DataFrame]
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
    logger.info("[WS1] Enriching relational features (Product, Household)...")

    # Check if required dataframes exist
    if 'product' not in dataframes_dict:
        logger.warning("SKIPPING WS1: 'product' dataframe not found in dataframes_dict.")
        return master_df

    df_prod = dataframes_dict['product']

    # Validate required columns exist
    if 'PRODUCT_ID' not in master_df.columns:
        logger.error("SKIPPING WS1: 'PRODUCT_ID' not found in master_df")
        return master_df

    if 'PRODUCT_ID' not in df_prod.columns:
        logger.error("SKIPPING WS1: 'PRODUCT_ID' not found in product dataframe")
        return master_df

    try:
        # Merge Product data
        original_shape = master_df.shape
        master_df = pd.merge(master_df, df_prod, on='PRODUCT_ID', how='left')

        # Check merge success
        matched_products = master_df['MANUFACTURER'].notna().sum() if 'MANUFACTURER' in master_df.columns else 0

        logging.info(f"  Merged product data: {original_shape} -> {master_df.shape}")
        logging.info(f"  Product matching: {matched_products:,}/{len(master_df):,} rows have product info")

        # LỚP 2: Fill NaN cho categorical columns sau join (luôn luôn)
        # Đảm bảo không có NaN trong categorical features trước khi chuyển sang category dtype
        product_cols = ['MANUFACTURER', 'DEPARTMENT', 'BRAND', 'COMMODITY_DESC', 'SUB_COMMODITY_DESC']
        for col in product_cols:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna('Unknown')
        logger.info("  ✓ Filled NaN in product categorical columns with 'Unknown'")

        if matched_products == 0:
            logger.error("  CRITICAL ERROR: No products matched! WS1 relational features failed.")
            logger.error("  SOLUTION: Recreate POC data with PRODUCT_ID matching validation")
        elif matched_products < len(master_df) * 0.1:
            logger.warning(f"  ⚠️  Only {matched_products/len(master_df)*100:.1f}% products have info - limited coverage")
        else:
            logging.info(f"  ✅ Good product data coverage: {matched_products/len(master_df)*100:.1f}%")

    except pd.errors.MergeError as e:
        logger.error(f"ERROR in WS1 product merge: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in WS1 product merge: {e}", exc_info=True)
        raise

    # Merge Household Demographics (optional - only if available)
    if 'hh_demographic' in dataframes_dict:
        df_hh = dataframes_dict['hh_demographic']

        if 'household_key' in master_df.columns and 'household_key' in df_hh.columns:
            try:
                original_shape = master_df.shape
                master_df = pd.merge(master_df, df_hh, on='household_key', how='left')
                logging.info(f"  Merged household demographics: {original_shape} -> {master_df.shape}")

                # LỚP 2: Fill NaN cho household categorical columns sau join (luôn luôn)
                hh_categorical_cols = ['AGE_DESC', 'MARITAL_STATUS_CODE', 'INCOME_DESC', 'HOMEOWNER_DESC',
                                      'HH_COMP_DESC', 'HOUSEHOLD_SIZE_DESC', 'KID_CATEGORY_DESC']
                for col in hh_categorical_cols:
                    if col in master_df.columns:
                        master_df[col] = master_df[col].fillna('Unknown')
                logger.info("  ✓ Filled NaN in household categorical columns with 'Unknown'")

            except pd.errors.MergeError as e:
                logger.error(f"ERROR in WS1 household merge: {e}")
                # Don't raise - continue without household data
            except Exception as e:
                logger.warning(f"Unexpected error in WS1 household merge: {e}. Continuing without household data.")
        else:
            logging.info("  Skipping household merge: 'household_key' not found in required dataframes")
    else:
        logging.info("  Skipping household merge: 'hh_demographic' not found in dataframes_dict")

    logging.info("OK. WS1 (Relational) integration complete.")
    return master_df
