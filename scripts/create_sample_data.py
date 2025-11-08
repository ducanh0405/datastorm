"""
Script to create 1% sample dataset for smoke tests.
This creates data/poc_data/ with small samples from raw data.
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / '2_raw'
POC_DATA_DIR = PROJECT_ROOT / 'data' / 'poc_data'

SAMPLE_FRACTION = 0.01  # 1% sample

def create_sample_data():
    """Creates 1% stratified sample from Dunnhumby dataset."""
    
    logging.info("Creating 1% sample dataset for smoke tests...")
    
    if not RAW_DATA_DIR.exists():
        logging.error(f"Raw data directory not found: {RAW_DATA_DIR}")
        return
    
    POC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # List of files to sample
    files_to_sample = [
        'transaction_data.csv',
        'product.csv',
        'hh_demographic.csv',
        'causal_data.csv',
        'coupon.csv',
        'coupon_redempt.csv',
        'campaign_table.csv',
        'campaign_desc.csv'
    ]
    
    # Step 1: Process transaction data first to get sampled products
    transaction_products = None
    transaction_stores = None

    transaction_path = RAW_DATA_DIR / 'transaction_data.csv'
    if transaction_path.exists():
        logging.info("Processing transaction_data.csv (determines sample)...")
        df_trans = pd.read_csv(transaction_path)

        # Sample products (ensure they exist in product data for WS1 relational features)
        unique_products = df_trans['PRODUCT_ID'].unique()

        # Load product data to ensure matching PRODUCT_IDs
        try:
            product_path = RAW_DATA_DIR / 'product.csv'
            if product_path.exists():
                df_product = pd.read_csv(product_path)
                available_products = set(df_product['PRODUCT_ID'].unique())
                valid_products = [p for p in unique_products if p in available_products]

                if len(valid_products) == 0:
                    logging.error("No products in transaction data match product catalog!")
                    logging.error("WS1 relational features will not work.")
                elif len(valid_products) < len(unique_products) * 0.1:
                    logging.warning(f"Only {len(valid_products)}/{len(unique_products)} products have matching product data.")
                    logging.warning("WS1 relational features will have limited coverage.")

                unique_products = valid_products
            else:
                logging.warning("Product data not found, proceeding without PRODUCT_ID validation")
        except Exception as e:
            logging.warning(f"Could not validate PRODUCT_ID matching: {e}")

        transaction_products = pd.Series(unique_products).sample(
            frac=min(SAMPLE_FRACTION, 1.0),  # Don't exceed available valid products
            random_state=42
        ).values

        # Sample stores
        unique_stores = df_trans['STORE_ID'].unique()
        transaction_stores = pd.Series(unique_stores).sample(
            frac=SAMPLE_FRACTION,
            random_state=42
        ).values

        # Filter transactions for sampled products and stores
        df_trans_sample = df_trans[
            df_trans['PRODUCT_ID'].isin(transaction_products) &
            df_trans['STORE_ID'].isin(transaction_stores)
        ]

        # Save transaction sample
        trans_output = POC_DATA_DIR / 'transaction_data.csv'
        df_trans_sample.to_csv(trans_output, index=False)
        logging.info(f"  Saved transaction_data.csv: {len(df_trans):,} -> {len(df_trans_sample):,} rows")

    # Step 2: Process other files with referential integrity
    for filename in files_to_sample:
        if filename == 'transaction_data.csv':
            continue  # Already processed

        input_path = RAW_DATA_DIR / filename
        output_path = POC_DATA_DIR / filename

        if not input_path.exists():
            logging.warning(f"File not found, skipping: {filename}")
            continue

        try:
            logging.info(f"Processing {filename}...")

            # Read file
            df = pd.read_csv(input_path)
            original_rows = len(df)

            # Apply referential sampling based on transaction data
            if filename == 'product.csv' and transaction_products is not None:
                # Keep only products that appear in sampled transactions
                df_sample = df[df['PRODUCT_ID'].isin(transaction_products)]
                logging.info(f"  Referential filter: kept {len(df_sample)}/{len(df)} products from transactions")

            elif filename == 'hh_demographic.csv':
                # Keep all household demographics (they're referenced by household_key in transactions)
                df_sample = df  # Don't sample - keep all for referential integrity
                logging.info("  Referential: kept all household demographics")

            elif filename in ['causal_data.csv', 'coupon.csv', 'coupon_redempt.csv', 'campaign_table.csv', 'campaign_desc.csv']:
                # For promotion/campaign data, filter by sampled products and stores if available
                df_sample = df
                if 'PRODUCT_ID' in df.columns and transaction_products is not None:
                    df_sample = df_sample[df_sample['PRODUCT_ID'].isin(transaction_products)]
                    logging.info(f"  Referential filter: kept promotion data for {len(df_sample)} products")
                else:
                    df_sample = df.sample(frac=SAMPLE_FRACTION, random_state=42)
            else:
                # Default random sampling for other files
                df_sample = df.sample(frac=SAMPLE_FRACTION, random_state=42)

            # Save sample
            df_sample.to_csv(output_path, index=False)

            logging.info(f"  Saved {filename}: {original_rows:,} -> {len(df_sample):,} rows ({len(df_sample)/original_rows*100:.2f}%)")

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
    
    logging.info(f"\nSample data created in: {POC_DATA_DIR}")
    logging.info("Use these files for quick smoke tests without loading full dataset.")

if __name__ == "__main__":
    create_sample_data()
