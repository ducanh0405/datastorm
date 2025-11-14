#!/usr/bin/env python3
"""
Phase 1 Fix: Auto-update LightGBM Parameters for Stability
============================================================

This script automatically updates src/config.py to add LightGBM 4.5.0+
stability parameters that prevent "best gain: -inf" warnings and ensure
reproducible results.

Usage:
    python scripts/apply_phase1_lightgbm_fix.py

Changes:
    - Adds 'deterministic': True
    - Adds 'force_col_wise': True
    - Adds 'min_split_gain': 0.001
    - Adds 'min_child_samples': 20
    - Adds 'feature_pre_filter': False
"""

import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / 'src' / 'config.py'

# New LightGBM params with stability fixes
NEW_LIGHTGBM_PARAMS = """# LightGBM hyperparameters (Tối ưu cho tốc độ/hiệu năng trên 32GB RAM)
# FIX Phase 1: Added stability params for LightGBM 4.5.0+ (2025-11-15)
LIGHTGBM_PARAMS = {
    'n_estimators': 600,
    'learning_rate': 0.03,
    'num_leaves': 48,
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': PERFORMANCE_CONFIG['parallel_threads'],
    'verbose': -1,
    
    # FIX: Stability improvements for LightGBM 4.5.0+
    # Prevents \"No further splits with positive gain, best gain: -inf\" warnings
    # Ensures reproducible results across multiple runs
    # Reference: https://github.com/microsoft/LightGBM/issues/6964
    'deterministic': True,        # Deterministic tree building
    'force_col_wise': True,       # Force column-wise histogram building (more stable)
    'min_split_gain': 0.001,      # Minimum gain to make a split (prevents -inf gain)
    'min_child_samples': 20,      # Minimum samples in leaf (prevents overfitting)
    'feature_pre_filter': False,  # Disable feature pre-filtering for stability
}"""

def backup_config():
    """Create backup of config.py"""
    backup_path = CONFIG_PATH.with_suffix('.py.backup')
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Backup created: {backup_path}")
    return backup_path

def apply_fix():
    """Apply LightGBM stability fix to config.py"""
    
    if not CONFIG_PATH.exists():
        print(f"❌ Error: {CONFIG_PATH} not found")
        sys.exit(1)
    
    # Read current config
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'deterministic' in content and "# FIX Phase 1" in content:
        print("✅ LightGBM stability params already applied!")
        print("   No changes needed.")
        return False
    
    # Create backup
    backup_path = backup_config()
    
    # Find and replace LIGHTGBM_PARAMS
    # Pattern: Match from "LIGHTGBM_PARAMS = {" to the closing "}"
    pattern = r"# LightGBM hyperparameters.*?\nLIGHTGBM_PARAMS = \{[^}]*\}"
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("❌ Error: Could not find LIGHTGBM_PARAMS in config.py")
        print("   Manual update required.")
        return False
    
    # Replace old params with new params
    updated_content = content[:match.start()] + NEW_LIGHTGBM_PARAMS + content[match.end():]
    
    # Write updated config
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print("✅ LightGBM stability params applied successfully!")
    print(f"   Updated: {CONFIG_PATH}")
    print(f"   Backup: {backup_path}")
    
    # Show diff
    print("\n" + "="*70)
    print("CHANGES APPLIED:")
    print("="*70)
    print("Added 5 new stability parameters:")
    print("  + 'deterministic': True")
    print("  + 'force_col_wise': True")
    print("  + 'min_split_gain': 0.001")
    print("  + 'min_child_samples': 20")
    print("  + 'feature_pre_filter': False")
    print("="*70)
    
    return True

def verify_fix():
    """Verify that fix was applied correctly"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_params = [
        "'deterministic': True",
        "'force_col_wise': True",
        "'min_split_gain': 0.001",
        "'min_child_samples': 20",
        "'feature_pre_filter': False"
    ]
    
    missing = []
    for param in required_params:
        if param not in content:
            missing.append(param)
    
    if missing:
        print(f"\n⚠️  Warning: Some params not found: {missing}")
        return False
    
    print("\n✅ Verification passed: All stability params present")
    return True

def main():
    print("="*70)
    print("Phase 1 Fix: LightGBM Stability Parameters")
    print("="*70)
    print(f"\nTarget file: {CONFIG_PATH}")
    print("\nThis will add 5 stability parameters to LIGHTGBM_PARAMS:")
    print("  - deterministic")
    print("  - force_col_wise")
    print("  - min_split_gain")
    print("  - min_child_samples")
    print("  - feature_pre_filter")
    print("\nThese params fix LightGBM 4.5.0+ instability issues.")
    print("="*70)
    
    # Apply fix
    success = apply_fix()
    
    if success:
        # Verify
        verify_fix()
        
        print("\n" + "="*70)
        print("✅ Phase 1 Fix Applied Successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review changes: git diff src/config.py")
        print("  2. Test: python src/pipelines/_03_model_training.py")
        print("  3. Commit: git add src/config.py")
        print("           git commit -m 'fix: Add LightGBM 4.5.0 stability params'")
        print("="*70)
    else:
        print("\n⚠️  Fix not applied. Manual update may be required.")
        print("    See PHASE1_FIXES.md for instructions.")

if __name__ == "__main__":
    main()
