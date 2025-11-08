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
    
    for filename in files_to_sample:
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
            
            # For transaction data, use stratified sampling by PRODUCT_ID and STORE_ID
            if filename == 'transaction_data.csv':
                # Sample products first
                unique_products = df['PRODUCT_ID'].unique()
                sample_products = pd.Series(unique_products).sample(
                    frac=SAMPLE_FRACTION, 
                    random_state=42
                ).values
                
                # Filter transactions for sampled products
                df_sample = df[df['PRODUCT_ID'].isin(sample_products)]
            else:
                # Simple random sample for other files
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
