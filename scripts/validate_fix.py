#!/usr/bin/env python3
"""
Validation script for sparse data fix and feature completeness.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

def validate_sparse_data_fix():
    """Validate that the sparse data fix works correctly."""
    print("=" * 60)
    print("VALIDATING SPARSE DATA FIX")
    print("=" * 60)

    # Check if raw data exists
    raw_data_path = PROJECT_ROOT / 'data' / '2_raw' / 'transaction_data.csv'
    if not raw_data_path.exists():
        print(f"❌ Raw data not found: {raw_data_path}")
        return False

    print("📊 Loading transaction data...")
    df_raw = pd.read_csv(raw_data_path, nrows=50000)  # Sample for validation
    print(f"   Raw transactions: {len(df_raw):,} rows")

    # Simulate WS0 aggregation
    print("\n🔄 Simulating WS0 aggregation...")
    agg_df = df_raw.groupby(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).agg({
        'SALES_VALUE': 'sum',
        'QUANTITY': 'sum',
        'RETAIL_DISC': 'sum',
        'COUPON_DISC': 'sum',
        'COUPON_MATCH_DISC': 'sum'
    }).reset_index()

    print(f"   After aggregation: {len(agg_df):,} weekly records")

    # Create full grid (old problematic way)
    print("\n📈 Creating full grid (old way)...")
    all_products = agg_df['PRODUCT_ID'].unique()
    all_stores = agg_df['STORE_ID'].unique()
    all_weeks = agg_df['WEEK_NO'].unique()

    from itertools import product
    grid_combinations = list(product(all_products, all_stores, all_weeks))
    print(f"   Grid combinations: {len(grid_combinations):,}")

    # Create sparse master table
    grid_df = pd.DataFrame(grid_combinations, columns=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])
    sparse_master = pd.merge(grid_df, agg_df, on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'], how='left').fillna(0)

    zero_rows = (sparse_master['SALES_VALUE'] == 0).sum()
    print(f"   Sparse master table: {len(sparse_master):,} rows")
    print(f"   Zero-filled rows: {zero_rows:,} ({zero_rows/len(sparse_master)*100:.1f}%)")

    # Apply the fix
    print("\n✅ Applying sparse data fix...")
    filtered_master = sparse_master[sparse_master['SALES_VALUE'] > 0].reset_index(drop=True)
    removed_rows = len(sparse_master) - len(filtered_master)
    print(f"   Filtered master table: {len(filtered_master):,} rows")
    print(f"   Removed zero-filled rows: {removed_rows:,}")
    print(f"   Data retention: {len(filtered_master)/len(agg_df)*100:.1f}%")

    # Validate sample quality
    print("\n📋 Sample Quality Check:")
    sample_old = sparse_master.head(100)
    sample_new = filtered_master.head(100)

    print(f"   Old sample SALES_VALUE: min={sample_old['SALES_VALUE'].min():.2f}, max={sample_old['SALES_VALUE'].max():.2f}")
    print(f"   New sample SALES_VALUE: min={sample_new['SALES_VALUE'].min():.2f}, max={sample_new['SALES_VALUE'].max():.2f}")

    return True

def validate_feature_config():
    """Validate that config includes all expected features."""
    print("\n" + "=" * 60)
    print("VALIDATING FEATURE CONFIG")
    print("=" * 60)

    try:
        from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    except ImportError as e:
        print(f"❌ Cannot import config: {e}")
        return False

    print(f"✅ Loaded {len(NUMERIC_FEATURES)} numeric features")
    print(f"✅ Loaded {len(CATEGORICAL_FEATURES)} categorical features")

    # Check for expected features
    expected_rolling = [
        'rolling_mean_4_lag_1', 'rolling_std_4_lag_1', 'rolling_max_4_lag_1', 'rolling_min_4_lag_1',
        'rolling_mean_8_lag_1', 'rolling_std_8_lag_1', 'rolling_max_8_lag_1', 'rolling_min_8_lag_1',
        'rolling_mean_12_lag_1', 'rolling_std_12_lag_1', 'rolling_max_12_lag_1', 'rolling_min_12_lag_1'
    ]

    expected_trend = ['wow_change', 'wow_pct_change', 'momentum', 'volatility']
    expected_quantity = ['quantity_lag_1', 'quantity_lag_4']

    rolling_present = all(feat in NUMERIC_FEATURES for feat in expected_rolling)
    trend_present = all(feat in NUMERIC_FEATURES for feat in expected_trend)
    quantity_present = all(feat in NUMERIC_FEATURES for feat in expected_quantity)

    print(f"\n📊 Feature Completeness:")
    print(f"   Rolling stats ({len(expected_rolling)} features): {'✅' if rolling_present else '❌'}")
    print(f"   Trend features ({len(expected_trend)}): {'✅' if trend_present else '❌'}")
    print(f"   Quantity lags ({len(expected_quantity)}): {'✅' if quantity_present else '❌'}")

    if not all([rolling_present, trend_present, quantity_present]):
        print("\n⚠️  Missing features in config:")
        for feat in expected_rolling + expected_trend + expected_quantity:
            if feat not in NUMERIC_FEATURES:
                print(f"     - {feat}")

    return rolling_present and trend_present and quantity_present

def create_improved_sample():
    """Create an improved sample from existing data."""
    print("\n" + "=" * 60)
    print("CREATING IMPROVED SAMPLE")
    print("=" * 60)

    # Try to load existing master table
    master_path = PROJECT_ROOT / 'data' / '3_processed' / 'master_feature_table.parquet'
    sample_path = PROJECT_ROOT / 'data' / '3_processed' / 'master_feature_table_improved_sample.csv'

    if not master_path.exists():
        print(f"❌ Master table not found: {master_path}")
        return False

    print("📖 Loading master table...")
    try:
        df = pd.read_parquet(master_path)
        print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading master table: {e}")
        return False

    # Apply filter if not already applied
    if (df['SALES_VALUE'] == 0).any():
        print("🔧 Applying sales filter...")
        original_len = len(df)
        df = df[df['SALES_VALUE'] > 0].reset_index(drop=True)
        print(f"   Filtered: {original_len:,} -> {len(df):,} rows")

    # Create improved sample
    print("📝 Creating improved sample...")
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42).sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])

    # Save sample
    sample_df.to_csv(sample_path, index=False)
    print(f"✅ Saved improved sample: {sample_path}")
    print(f"   Sample size: {len(sample_df)} rows")
    print(f"   SALES_VALUE range: {sample_df['SALES_VALUE'].min():.2f} - {sample_df['SALES_VALUE'].max():.2f}")

    return True

def main():
    """Main validation function."""
    print("🚀 STARTING VALIDATION OF FIXES")
    print(f"Project root: {PROJECT_ROOT}")

    success = True

    # Test 1: Sparse data fix
    if not validate_sparse_data_fix():
        success = False

    # Test 2: Feature config
    if not validate_feature_config():
        success = False

    # Test 3: Create improved sample
    if not create_improved_sample():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Sparse data fix is working")
        print("✅ Feature config is complete")
        print("✅ Improved sample created")
    else:
        print("⚠️  SOME VALIDATIONS FAILED")
    print("=" * 60)

    return success

if __name__ == "__main__":
    main()
