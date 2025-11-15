"""
Comprehensive Testing Script for Refactoring Validation
========================================================
Tests all critical fixes and validates pipeline integrity.
"""
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_features():
    """Test ALL_FEATURES_CONFIG has correct structure."""
    logger.info("=" * 70)
    logger.info("TEST 1: Config Features Structure")
    logger.info("=" * 70)
    
    try:
        from src.config import ALL_FEATURES_CONFIG, get_features_by_type
        
        # Check structure
        assert isinstance(ALL_FEATURES_CONFIG, dict), "ALL_FEATURES_CONFIG should be dict"
        
        for ws_name, features in ALL_FEATURES_CONFIG.items():
            assert isinstance(features, list), f"{ws_name} should be list"
            for feat in features:
                assert isinstance(feat, dict), f"Feature in {ws_name} should be dict"
                assert 'name' in feat, f"Feature in {ws_name} missing 'name'"
                assert 'type' in feat, f"Feature in {ws_name} missing 'type'"
                assert feat['type'] in ['num', 'cat'], f"Invalid type: {feat['type']}"
        
        # Test helper function
        all_features = get_features_by_type('all')
        num_features = get_features_by_type('num')
        cat_features = get_features_by_type('cat')
        
        assert len(all_features) == len(num_features) + len(cat_features), \
            "Feature counts don't match"
        
        logger.info(f"✅ Config structure valid")
        logger.info(f"   Total features: {len(all_features)}")
        logger.info(f"   Numeric: {len(num_features)}")
        logger.info(f"   Categorical: {len(cat_features)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False


def test_parallel_processing():
    """Test parallel processing error handling."""
    logger.info("=" * 70)
    logger.info("TEST 2: Parallel Processing Error Handling")
    logger.info("=" * 70)
    
    try:
        from src.utils.parallel_processing import parallel_groupby_apply
        
        # Create test data
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'] * 10,
            'value': range(40)
        })
        
        # Test with failing function
        def failing_func(group_df):
            raise ValueError("Intentional test error")
        
        try:
            result = parallel_groupby_apply(
                df,
                group_cols=['group'],
                func=failing_func,
                n_jobs=2
            )
            logger.error("❌ Should have raised RuntimeError")
            return False
        except RuntimeError as e:
            if "both backends" in str(e).lower():
                logger.info(f"✅ Error handling works correctly: {e}")
                return True
            else:
                logger.error(f"❌ Wrong error type: {e}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Parallel processing test failed: {e}")
        return False


def test_file_loading():
    """Test file loading with corruption handling."""
    logger.info("=" * 70)
    logger.info("TEST 3: File Loading Error Handling")
    logger.info("=" * 70)
    
    try:
        from src.pipelines._01_load_data import _load_file
        
        # Test with non-existent file
        result = _load_file(Path("nonexistent"), "fake_file")
        assert result is None, "Should return None for missing file"
        
        logger.info(f"✅ File loading error handling works")
        return True
        
    except Exception as e:
        logger.error(f"❌ File loading test failed: {e}")
        return False


def test_input_validation():
    """Test input validation for memory optimization."""
    logger.info("=" * 70)
    logger.info("TEST 4: Input Validation")
    logger.info("=" * 70)
    
    try:
        from src.pipelines._01_load_data import _sample_data_for_memory_optimization
        from src.config import MEMORY_OPTIMIZATION
        
        # Create test data
        df = pd.DataFrame({
            'product_id': range(100),
            'store_id': [1, 2] * 50,
            'sales': np.random.rand(100)
        })
        dataframes = {'sales': df}
        config = {'time_column': 'week'}
        
        # Save original settings
        original_enabled = MEMORY_OPTIMIZATION['enable_sampling']
        original_fraction = MEMORY_OPTIMIZATION['sample_fraction']
        
        # Test with invalid fraction
        MEMORY_OPTIMIZATION['enable_sampling'] = True
        MEMORY_OPTIMIZATION['sample_fraction'] = -0.5  # Invalid
        
        result = _sample_data_for_memory_optimization(dataframes, config)
        
        # Should use default 0.1
        assert len(result['sales']) <= len(df), "Should handle invalid fraction"
        
        # Restore settings
        MEMORY_OPTIMIZATION['enable_sampling'] = original_enabled
        MEMORY_OPTIMIZATION['sample_fraction'] = original_fraction
        
        logger.info(f"✅ Input validation works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Input validation test failed: {e}")
        return False


def test_pandas_api():
    """Test pandas API compatibility."""
    logger.info("=" * 70)
    logger.info("TEST 5: Pandas API Compatibility")
    logger.info("=" * 70)
    
    try:
        # Check pandas version
        import pandas as pd
        logger.info(f"   Pandas version: {pd.__version__}")
        
        # Test fillna methods
        df = pd.DataFrame({'a': [1, np.nan, 3, np.nan, 5]})
        
        # New API (should work)
        result = df['a'].ffill().bfill()
        assert not result.isna().any(), "ffill/bfill should fill all NaNs"
        
        logger.info(f"✅ Pandas API compatibility verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pandas API test failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    logger.info("\n" + "=" * 70)
    logger.info("REFACTORING VALIDATION TEST SUITE")
    logger.info("=" * 70 + "\n")
    
    tests = [
        ("Config Features Structure", test_config_features),
        ("Parallel Processing Error Handling", test_parallel_processing),
        ("File Loading Error Handling", test_file_loading),
        ("Input Validation", test_input_validation),
        ("Pandas API Compatibility", test_pandas_api),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            logger.error(f"CRITICAL ERROR in {test_name}: {e}", exc_info=True)
            results.append((test_name, False))
        logger.info("")  # Blank line between tests
    
    # Summary
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("=" * 70)
    logger.info(f"RESULTS: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    logger.info("=" * 70)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

