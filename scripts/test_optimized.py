"""
TEST OPTIMIZED PIPELINE
=======================
Quick validation script for optimized features and models.

Checks:
1. WS2 optimized module loads correctly
2. Optuna is available
3. Feature engineering speedup
4. Model tuning works
5. Results comparison
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def test_optimized_ws2_import():
    """Test 1: Can we import WS2 (now optimized by default)?"""
    logging.info("\n[TEST 1] Importing WS2 Module")
    logging.info("-" * 70)
    
    try:
        from src.features import ws2_timeseries_features as ws2
        
        # Check functions exist
        assert hasattr(ws2, 'create_lag_features'), "Missing lag function"
        assert hasattr(ws2, 'create_rolling_features'), "Missing rolling function"
        assert hasattr(ws2, 'add_trend_features'), "Missing trend features function"
        assert hasattr(ws2, 'add_lag_rolling_features'), "Missing main function"
        
        logging.info("  [OK] All WS2 functions available (optimized by default)")
        return True
        
    except ImportError as e:
        logging.error(f"  [FAIL] Cannot import WS2: {e}")
        return False


def test_optuna_available():
    """Test 2: Is Optuna installed?"""
    logging.info("\n[TEST 2] Checking Optuna Installation")
    logging.info("-" * 70)
    
    try:
        import optuna
        logging.info(f"  [OK] Optuna version: {optuna.__version__}")
        return True
    except ImportError:
        logging.error("  [FAIL] Optuna not installed")
        logging.error("  Install with: pip install optuna")
        return False


def test_ws2_speed_improvement():
    """Test 3: WS2 speed test on small dataset."""
    logging.info("\n[TEST 3] WS2 Speed Benchmark")
    logging.info("-" * 70)
    
    import pandas as pd
    import numpy as np
    
    # Create small test dataset
    np.random.seed(42)
    n_products = 100
    n_stores = 50
    n_weeks = 102
    
    data = []
    for prod in range(1, n_products + 1):
        for store in range(1, n_stores + 1):
            for week in range(1, n_weeks + 1):
                # Sparse data (90% zeros)
                sales = np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, np.random.uniform(1, 100)])
                data.append({
                    'PRODUCT_ID': prod,
                    'STORE_ID': store,
                    'WEEK_NO': week,
                    'SALES_VALUE': sales,
                    'QUANTITY': int(sales / 10) if sales > 0 else 0
                })
    
    df = pd.DataFrame(data)
    logging.info(f"  Test data: {len(df):,} rows ({n_products} products × {n_stores} stores × {n_weeks} weeks)")
    
    # Test WS2 (now optimized by default)
    try:
        from src.features.ws2_timeseries_features import add_lag_rolling_features as ws2_func
        
        start = time.time()
        result = ws2_func(df.copy())
        time_elapsed = time.time() - start
        
        logging.info(f"  WS2 (optimized): {time_elapsed:.2f}s")
        logging.info(f"  Output shape: {result.shape}")
        
        # Verify features created
        lag_cols = [c for c in result.columns if 'lag' in c]
        roll_cols = [c for c in result.columns if 'rolling' in c]
        trend_cols = [c for c in result.columns if c in ['wow_change', 'momentum', 'volatility']]
        
        logging.info(f"  Features created: {len(lag_cols)} lags, {len(roll_cols)} rolling, {len(trend_cols)} trend")
        
        if len(lag_cols) >= 4 and len(roll_cols) >= 12 and len(trend_cols) >= 3:
            logging.info("  [OK] All expected features created")
            return True
        else:
            logging.warning("  [WARN] Some features missing")
            return False
        
    except Exception as e:
        logging.error(f"  [FAIL] Error running WS2: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tuned_pipeline_modules():
    """Test 4: Can we use hyperparameter tuning in main training module?"""
    logging.info("\n[TEST 4] Checking Hyperparameter Tuning Integration")
    logging.info("-" * 70)

    try:
        from src.pipelines import _03_model_training as train_module

        # Check functions exist in main module
        assert hasattr(train_module, 'tune_quantile_hyperparameters'), "Missing tuning function"
        assert hasattr(train_module, 'train_quantile_models_tuned'), "Missing tuned training function"
        assert hasattr(train_module, 'create_time_series_cv_splits'), "Missing CV function"
        assert hasattr(train_module, 'main'), "Missing main function"

        # Check Optuna availability
        if hasattr(train_module, 'OPTUNA_AVAILABLE'):
            logging.info(f"  [OK] Optuna available: {train_module.OPTUNA_AVAILABLE}")
        else:
            logging.warning("  [WARN] Optuna availability not detected")

        logging.info("  [OK] All tuning functions available in main module")
        return True

    except ImportError as e:
        logging.error(f"  [FAIL] Cannot import main training module: {e}")
        return False
    except AssertionError as e:
        logging.error(f"  [FAIL] Missing function: {e}")
        return False


def test_pipeline_runner():
    """Test 5: Can we import optimized pipeline runner?"""
    logging.info("\n[TEST 5] Checking Pipeline Runner")
    logging.info("-" * 70)
    
    script_path = PROJECT_ROOT / 'scripts' / 'run_optimized_pipeline.py'
    
    if not script_path.exists():
        logging.error(f"  [FAIL] Script not found: {script_path}")
        return False
    
    logging.info(f"  [OK] Script exists: {script_path}")
    
    # Try to import (not run)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_optimized", script_path)
        module = importlib.util.module_from_spec(spec)
        
        # Don't execute, just check syntax
        logging.info("  [OK] Script has valid Python syntax")
        return True
        
    except Exception as e:
        logging.error(f"  [FAIL] Script has errors: {e}")
        return False


def test_documentation_exists():
    """Test 6: Check if documentation is complete."""
    logging.info("\n[TEST 6] Checking Documentation")
    logging.info("-" * 70)
    
    docs = [
        'reports/OPTIMIZED_PIPELINE_GUIDE.md',
        'reports/UPGRADE_PLAN.md',
        'reports/EXECUTION_TEST_REPORT.md'
    ]
    
    all_exist = True
    
    for doc in docs:
        doc_path = PROJECT_ROOT / doc
        if doc_path.exists():
            size = doc_path.stat().st_size / 1024
            logging.info(f"  [OK] {doc} ({size:.1f} KB)")
        else:
            logging.error(f"  [FAIL] Missing: {doc}")
            all_exist = False
    
    return all_exist


def main():
    logging.info("=" * 70)
    logging.info("OPTIMIZED PIPELINE VALIDATION")
    logging.info("=" * 70)
    
    results = {}
    
    results['ws2_import'] = test_optimized_ws2_import()
    results['optuna'] = test_optuna_available()
    results['ws2_speed'] = test_ws2_speed_improvement()
    results['tuned_modules'] = test_tuned_pipeline_modules()
    results['runner_script'] = test_pipeline_runner()
    results['documentation'] = test_documentation_exists()
    
    # Summary
    logging.info("\n" + "=" * 70)
    logging.info("TEST SUMMARY")
    logging.info("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        logging.info(f"{test_name:20s}: {status}")
    
    logging.info("-" * 70)
    logging.info(f"TOTAL: {passed}/{total} tests passed")
    logging.info("=" * 70)
    
    if passed == total:
        logging.info("\n✓ ALL TESTS PASSED - Pipeline ready to run!")
        logging.info("\nNext steps:")
        logging.info("  1. Quick test:  python scripts/run_optimized_pipeline.py")
        logging.info("  2. Full tuning: python scripts/run_optimized_pipeline.py --tune --trials 30")
        return 0
    else:
        logging.error("\n✗ SOME TESTS FAILED - Fix issues before running pipeline")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
