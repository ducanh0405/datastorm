"""
Test script để kiểm tra import config
"""
import sys
from pathlib import Path

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Python path (first 3): {sys.path[:3]}")

try:
    from src.config import (
        setup_project_path, setup_logging, ensure_directories,
        OUTPUT_FILES, TRAINING_CONFIG, get_dataset_config, PROJECT_ROOT as PR
    )
    print(f"✓ Config imported successfully!")
    print(f"  Project root from config: {PR}")
    print(f"  Active dataset: {get_dataset_config()['name']}")
    print(f"  Quantiles: {TRAINING_CONFIG['quantiles']}")
    print(f"  Model types: {TRAINING_CONFIG.get('model_types', ['lightgbm'])}")
    print(f"  Output files:")
    for key, path in OUTPUT_FILES.items():
        if isinstance(path, Path):
            print(f"    {key}: {path}")
    print("\n✓ All imports successful!")
except ImportError as e:
    print(f"✗ Error importing config: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

