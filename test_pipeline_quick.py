#!/usr/bin/env python3
"""
Quick Pipeline Test - Test cơ bản để đảm bảo pipeline hoạt động
"""
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("[TEST] Testing Pipeline Components...")

try:
    # Test imports
    print("[IMPORT] Testing imports...")
    from src.config import setup_project_path, get_dataset_config
    from src.pipelines._01_load_data import load_data
    from src.features.ws0_aggregation import prepare_master_dataframe
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
    if 'freshretail_train' in dataframes:
        df = dataframes['freshretail_train']
        master_df = prepare_master_dataframe(df.head(1000))  # Small sample
        print(f"[OK] Master dataframe created: {master_df.shape}")

    print("\n[SUCCESS] ALL TESTS PASSED!")
    print("Pipeline is ready to run with: python run_pipeline.py --full-data")

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("Please check your setup and try again.")
    sys.exit(1)
