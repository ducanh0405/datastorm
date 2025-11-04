import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def enrich_relational_features(master_df, dataframes_dict):
    """
    (WS1 - Dunnhumby) Join cac bang quan he (Product, Household).
    """
    logging.info("[WS1] Enriching relational features (Product, Household)...")

    # Kiem tra data
    if 'product' not in dataframes_dict or 'hh_demographic' not in dataframes_dict:
        logging.warning("SKIPPING WS1: Khong tim thay 'product' hoac 'hh_demographic'.")
        return master_df

    df_prod = dataframes_dict['product']
    df_hh = dataframes_dict['hh_demographic']

    # Merge Product
    master_df = pd.merge(master_df, df_prod, on='PRODUCT_ID', how='left')

    # Merge Household Demographics
    master_df = pd.merge(master_df, df_hh, on='household_key', how='left')

    logging.info("OK. WS1 (Relational) integration complete.")
    return master_df