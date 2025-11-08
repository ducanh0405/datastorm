"""
Quick validation script to verify the refactored pipeline.
Tests core functionality without running full pipeline.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("DATASTORM PIPELINE VALIDATION")
print("=" * 70)

# Test 1: Module imports
print("\n[1/5] Testing module imports...")
try:
    from src.features.ws0_aggregation import prepare_master_dataframe
    from src.features.ws2_timeseries_features import add_lag_rolling_features
    from src.pipelines._01_load_data import load_competition_data
    print("  [OK] All modules import successfully")
except ImportError as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Directory structure
print("\n[2/5] Testing directory structure...")
required_dirs = [
    PROJECT_ROOT / 'data' / 'poc_data',
    PROJECT_ROOT / 'data' / 'processed',
    PROJECT_ROOT / 'models',
    PROJECT_ROOT / 'reports' / 'metrics',
    PROJECT_ROOT / 'tests',
]
all_exist = True
for directory in required_dirs:
    if directory.exists():
        print(f"  [OK] {directory.relative_to(PROJECT_ROOT)}")
    else:
        print(f"  [MISSING] {directory.relative_to(PROJECT_ROOT)}")
        all_exist = False

if not all_exist:
    print("  ERROR: Some directories missing!")
    sys.exit(1)

# Test 3: POC data existence
print("\n[3/5] Testing POC data...")
poc_data_dir = PROJECT_ROOT / 'data' / 'poc_data'
tx_file = poc_data_dir / 'transaction_data.csv'

if tx_file.exists():
    import pandas as pd
    df = pd.read_csv(tx_file, nrows=5)
    print(f"  [OK] POC transaction data exists: {len(pd.read_csv(tx_file)):,} rows")
    print(f"    Columns: {df.columns.tolist()[:5]}...")
else:
    print("  [FAIL] POC data not found. Run: python scripts/create_sample_data.py")

# Test 4: Configuration files
print("\n[4/5] Testing configuration files...")
config_files = [
    'pyproject.toml',
    '.pre-commit-config.yaml',
    'requirements-dev.txt',
]
for filename in config_files:
    filepath = PROJECT_ROOT / filename
    if filepath.exists():
        print(f"  [OK] {filename}")
    else:
        print(f"  [MISSING] {filename}")

# Test 5: Quick functional test
print("\n[5/5] Testing WS0 aggregation (functional test)...")
if tx_file.exists():
    import pandas as pd
    from src.features.ws0_aggregation import aggregate_to_weekly
    
    df_raw = pd.read_csv(tx_file, nrows=1000)  # Small sample
    df_agg = aggregate_to_weekly(df_raw)
    
    required_cols = ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'SALES_VALUE', 'QUANTITY']
    has_cols = all(col in df_agg.columns for col in required_cols)
    
    if has_cols:
        print(f"  [OK] Aggregation successful: {len(df_raw):,} rows -> {len(df_agg):,} rows")
        print(f"    Columns: {required_cols}")
    else:
        print(f"  [FAIL] Aggregation missing columns!")
else:
    print("  [SKIP] Skipped (no POC data)")

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("  1. Install dev tools: pip install -r requirements-dev.txt")
print("  2. Run smoke tests: pytest tests/test_smoke.py -v -m smoke")
print("  3. Run full pipeline: python src/pipelines/_04_run_pipeline.py")
