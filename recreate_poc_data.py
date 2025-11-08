#!/usr/bin/env python3
"""
Script to recreate POC data with proper PRODUCT_ID matching for WS1 relational features.
"""
import pandas as pd
from pathlib import Path
import shutil
import sys

def recreate_poc_data():
    """Recreate POC data ensuring PRODUCT_ID relationships are preserved."""

    print("🔄 RECREATING POC DATA WITH PRODUCT_ID MATCHING")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / 'data' / '2_raw'
    poc_dir = project_root / 'data' / 'poc_data'

    # Check if raw data exists
    transaction_path = raw_dir / 'transaction_data.csv'
    product_path = raw_dir / 'product.csv'

    if not transaction_path.exists():
        print(f"❌ Transaction data not found: {transaction_path}")
        return False

    if not product_path.exists():
        print(f"❌ Product data not found: {product_path}")
        return False

    print("📖 Loading raw data...")
    df_trans = pd.read_csv(transaction_path)
    df_product = pd.read_csv(product_path)

    print(f"   Transactions: {len(df_trans):,} rows")
    print(f"   Products: {len(df_product):,} rows")

    # Get valid PRODUCT_IDs that exist in both files
    trans_products = set(df_trans['PRODUCT_ID'].unique())
    product_ids = set(df_product['PRODUCT_ID'].unique())
    valid_products = trans_products.intersection(product_ids)

    print("
🔗 PRODUCT_ID Analysis:"    print(f"   Transaction products: {len(trans_products):,}")
    print(f"   Product catalog: {len(product_ids):,}")
    print(f"   Matching products: {len(valid_products):,}")
    print(".1f"
    if len(valid_products) == 0:
        print("❌ No matching products! Cannot create valid POC data.")
        return False

    # Sample from valid products only
    sample_fraction = 0.01  # 1%
    sample_size = int(len(valid_products) * sample_fraction)

    if sample_size == 0:
        sample_size = min(100, len(valid_products))  # At least 100 products

    import random
    random.seed(42)
    sampled_products = random.sample(list(valid_products), sample_size)

    print("
📊 Creating POC Sample:"    print(f"   Sample size: {len(sampled_products):,} products ({len(sampled_products)/len(valid_products)*100:.1f}%)")

    # Filter transactions for sampled products
    df_trans_sample = df_trans[df_trans['PRODUCT_ID'].isin(sampled_products)]
    print(f"   Transaction rows: {len(df_trans):,} -> {len(df_trans_sample):,}")

    # Filter products for sampled products
    df_product_sample = df_product[df_product['PRODUCT_ID'].isin(sampled_products)]
    print(f"   Product rows: {len(df_product):,} -> {len(df_product_sample):,}")

    # Create POC directory
    poc_dir.mkdir(parents=True, exist_ok=True)

    # Save samples
    print("
💾 Saving POC data..."    df_trans_sample.to_csv(poc_dir / 'transaction_data.csv', index=False)
    df_product_sample.to_csv(poc_dir / 'product.csv', index=False)

    # Copy other files (sample them if they're large)
    other_files = [
        'hh_demographic.csv',
        'causal_data.csv',
        'coupon.csv',
        'coupon_redempt.csv',
        'campaign_table.csv',
        'campaign_desc.csv'
    ]

    for filename in other_files:
        src_path = raw_dir / filename
        dst_path = poc_dir / filename

        if src_path.exists():
            if src_path.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                # Sample large files
                print(f"   Sampling {filename}...")
                df = pd.read_csv(src_path)
                df_sample = df.sample(frac=sample_fraction, random_state=42)
                df_sample.to_csv(dst_path, index=False)
                print(f"     {len(df):,} -> {len(df_sample):,} rows")
            else:
                # Copy small files as-is
                shutil.copy2(src_path, dst_path)
                print(f"   Copied {filename}")
        else:
            print(f"   ⚠️  {filename} not found, skipping")

    print("
✅ POC Data Recreation Complete!"    print(f"   📁 Location: {poc_dir}")
    print("   🎯 All PRODUCT_IDs in transactions match product catalog"
    print(f"   📊 Sample size: {len(sampled_products)} products")
    print("   🚀 WS1 relational features will now work properly"
    print("
🔄 Next Steps:"    print("   1. Run: python scripts/run_optimized_pipeline.py")
    print("   2. Check logs for: 'Product matching: X/X rows have product info'")
    print("   3. Verify MANUFACTURER,DEPARTMENT,BRAND,COMMODITY_DESC are populated")

    return True

if __name__ == "__main__":
    success = recreate_poc_data()
    if not success:
        print("\n❌ POC data recreation failed!")
        print("💡 Ensure data/2_raw/ contains transaction_data.csv and product.csv")
        sys.exit(1)
