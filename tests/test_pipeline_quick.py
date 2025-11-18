#!/usr/bin/env python3
"""
Quick Pipeline Test - Test cơ bản để đảm bảo pipeline hoạt động
"""
import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("[TEST] Testing Pipeline Components...")

try:
    # Test imports
    print("[IMPORT] Testing imports...")
    from src.config import get_dataset_config, setup_project_path
    from src.features.ws0_aggregation import prepare_master_dataframe
    from src.pipelines._01_load_data import load_data
    print("[OK] All imports successful")

    # Test config
    print("[CONFIG] Testing config...")
    config = get_dataset_config()
    assert config['name'] == 'FreshRetailNet-50K'
    print(f"[OK] Config loaded: {config['name']}")

    # Test data loading
    print("[DATA] Testing data loading...")
    dataframes, config = load_data()
    print(f"[OK] Data loaded: {len(dataframes)} dataframes")

    # Test sample data creation
    print("[PROCESS] Testing sample data processing...")
    if 'sales' in dataframes and dataframes['sales'] is not None:
        df = dataframes['sales']
        master_df = prepare_master_dataframe(df.head(1000))  # Small sample
        print(f"[OK] Master dataframe created: {master_df.shape}")
    elif 'freshretail_train' in dataframes:
        df = dataframes['freshretail_train']
        master_df = prepare_master_dataframe(df.head(1000))  # Small sample
        print(f"[OK] Master dataframe created: {master_df.shape}")

    print("\n[SUCCESS] ALL TESTS PASSED!")
    print("Pipeline is ready to run with: python run_pipeline.py --full-data")

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("Please check your setup and try again.")
    import traceback
    traceback.print_exc()
    sys.exit(1)

