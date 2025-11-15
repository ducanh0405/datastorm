#!/usr/bin/env python3
"""
Phase 2 Integration Tests
==========================

Comprehensive test suite for Phase 2 data quality monitoring features.

Tests:
1. GX setup verification
2. Data quality validator initialization
3. Validation runner execution
4. Pipeline v2 integration
5. CLI sampling configuration

Usage:
    pytest tests/test_phase2_integration.py -v
    python tests/test_phase2_integration.py  # Run directly

Author: SmartGrocy Team
Date: 2025-11-15
"""

import sys
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase2Integration:
    """Test suite for Phase 2 components"""
    
    def test_01_gx_setup_script_exists(self):
        """Test 1: Verify GX setup script exists"""
        script_path = PROJECT_ROOT / "scripts" / "setup_great_expectations.py"
        assert script_path.exists(), f"GX setup script not found: {script_path}"
        print("✓ Test 1 PASSED: GX setup script exists")
    
    def test_02_validation_runner_exists(self):
        """Test 2: Verify validation runner exists"""
        script_path = PROJECT_ROOT / "scripts" / "run_data_quality_check.py"
        assert script_path.exists(), f"Validation runner not found: {script_path}"
        print("✓ Test 2 PASSED: Validation runner exists")
    
    def test_03_gx_utility_module_exists(self):
        """Test 3: Verify GX utility module exists and imports"""
        module_path = PROJECT_ROOT / "src" / "utils" / "data_quality_gx.py"
        assert module_path.exists(), f"GX utility module not found: {module_path}"
        
        # Test import
        try:
            from src.utils.data_quality_gx import DataQualityValidator, validate_dataframe
            print("✓ Test 3 PASSED: GX utility module imports successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import GX utility: {e}")
    
    def test_04_orchestrator_v2_exists(self):
        """Test 4: Verify orchestrator v2 exists and imports"""
        module_path = PROJECT_ROOT / "src" / "pipelines" / "_00_modern_orchestrator_v2.py"
        assert module_path.exists(), f"Orchestrator v2 not found: {module_path}"
        
        # Test import - handle Prefect/Pydantic compatibility issues with Python 3.13
        try:
            from src.pipelines._00_modern_orchestrator_v2 import modern_pipeline_flow_v2
            print("✓ Test 4 PASSED: Orchestrator v2 imports successfully")
        except (TypeError, AttributeError) as e:
            # Prefect/Pydantic compatibility issue with Python 3.13
            # This is a known issue with Prefect 2.20.3 and Python 3.13
            if "model_config" in str(e) or "ConfigWrapper" in str(e) or "not iterable" in str(e):
                pytest.skip(f"Prefect/Pydantic compatibility issue with Python 3.13: {e}")
            else:
                pytest.fail(f"Failed to import orchestrator v2: {e}")
        except ImportError as e:
            pytest.fail(f"Failed to import orchestrator v2: {e}")
        except Exception as e:
            # Catch any other exceptions and check if it's the known compatibility issue
            if "model_config" in str(e) or "ConfigWrapper" in str(e) or "not iterable" in str(e):
                pytest.skip(f"Prefect/Pydantic compatibility issue with Python 3.13: {e}")
            else:
                pytest.fail(f"Failed to import orchestrator v2: {e}")
    
    def test_05_pipeline_runner_v2_exists(self):
        """Test 5: Verify pipeline runner v2 exists"""
        runner_path = PROJECT_ROOT / "run_modern_pipeline_v2.py"
        assert runner_path.exists(), f"Pipeline runner v2 not found: {runner_path}"
        print("✓ Test 5 PASSED: Pipeline runner v2 exists")
    
    def test_06_data_quality_validator_initialization(self):
        """Test 6: Verify DataQualityValidator can be initialized"""
        try:
            from src.utils.data_quality_gx import DataQualityValidator
            
            validator = DataQualityValidator()
            
            # Check if GX is available (may be False if not setup)
            is_available = validator.is_available()
            
            print(f"✓ Test 6 PASSED: DataQualityValidator initialized (GX available: {is_available})")
            
        except Exception as e:
            pytest.fail(f"Failed to initialize DataQualityValidator: {e}")
    
    def test_07_validate_dataframe_with_sample_data(self):
        """Test 7: Validate sample dataframe"""
        try:
            from src.utils.data_quality_gx import validate_dataframe
            
            # Create sample dataframe
            sample_df = pd.DataFrame({
                'sales_lag_1': np.random.rand(100),
                'sales_lag_4': np.random.rand(100),
                'week_of_year': np.random.randint(1, 53, 100),
                'DEPARTMENT': ['Dept_A'] * 50 + ['Dept_B'] * 50
            })
            
            # Run validation (should not fail even if GX not setup)
            result = validate_dataframe(
                sample_df,
                asset_name="test_data",
                fail_on_error=False
            )
            
            assert isinstance(result, dict), "Validation result should be a dict"
            assert 'success' in result, "Result should contain 'success' key"
            assert 'timestamp' in result, "Result should contain 'timestamp' key"
            
            print(f"✓ Test 7 PASSED: validate_dataframe works (success={result['success']})")
            
        except Exception as e:
            pytest.fail(f"validate_dataframe test failed: {e}")
    
    def test_08_memory_sampling_configuration(self):
        """Test 8: Verify memory sampling can be configured"""
        try:
            from src import config
            
            # Save original values
            original_enable = config.MEMORY_OPTIMIZATION['enable_sampling']
            original_fraction = config.MEMORY_OPTIMIZATION['sample_fraction']
            
            # Test configuration change
            config.MEMORY_OPTIMIZATION['enable_sampling'] = True
            config.MEMORY_OPTIMIZATION['sample_fraction'] = 0.1
            
            assert config.MEMORY_OPTIMIZATION['enable_sampling'] == True
            assert config.MEMORY_OPTIMIZATION['sample_fraction'] == 0.1
            
            # Restore original values
            config.MEMORY_OPTIMIZATION['enable_sampling'] = original_enable
            config.MEMORY_OPTIMIZATION['sample_fraction'] = original_fraction
            
            print("✓ Test 8 PASSED: Memory sampling configuration works")
            
        except Exception as e:
            pytest.fail(f"Memory sampling config test failed: {e}")
    
    def test_09_phase2_documentation_exists(self):
        """Test 9: Verify Phase 2 documentation exists"""
        docs = [
            "PHASE2_COMPLETION_REPORT.md",
            "QUICKSTART_PHASE2.md"
        ]
        
        for doc in docs:
            doc_path = PROJECT_ROOT / doc
            assert doc_path.exists(), f"Documentation not found: {doc}"
        
        print("✓ Test 9 PASSED: All Phase 2 documentation exists")
    
    def test_10_config_quality_thresholds(self):
        """Test 10: Verify data quality thresholds configured"""
        from src.config import DATA_QUALITY_CONFIG
        
        # Check thresholds exist
        assert 'quality_thresholds' in DATA_QUALITY_CONFIG
        assert 'alerting' in DATA_QUALITY_CONFIG
        
        thresholds = DATA_QUALITY_CONFIG['quality_thresholds']
        assert 'excellent' in thresholds
        assert 'good' in thresholds
        assert 'fair' in thresholds
        assert 'poor' in thresholds
        
        # Verify values are reasonable
        assert thresholds['excellent'] > thresholds['good']
        assert thresholds['good'] > thresholds['fair']
        assert thresholds['fair'] > thresholds['poor']
        
        print("✓ Test 10 PASSED: Quality thresholds properly configured")
        print(f"  Thresholds: Excellent={thresholds['excellent']}, Good={thresholds['good']}, Fair={thresholds['fair']}, Poor={thresholds['poor']}")


def run_all_tests():
    """Run all tests without pytest"""
    print("="*70)
    print("PHASE 2 INTEGRATION TEST SUITE")
    print("="*70)
    print(f"Project: {PROJECT_ROOT}")
    print("\nRunning 10 tests...\n")
    
    test_suite = TestPhase2Integration()
    tests = [
        test_suite.test_01_gx_setup_script_exists,
        test_suite.test_02_validation_runner_exists,
        test_suite.test_03_gx_utility_module_exists,
        test_suite.test_04_orchestrator_v2_exists,
        test_suite.test_05_pipeline_runner_v2_exists,
        test_suite.test_06_data_quality_validator_initialization,
        test_suite.test_07_validate_dataframe_with_sample_data,
        test_suite.test_08_memory_sampling_configuration,
        test_suite.test_09_phase2_documentation_exists,
        test_suite.test_10_config_quality_thresholds,
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        try:
            print(f"\n[{i}/10] Running {test.__name__}...")
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ Test {i} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Test {i} ERROR: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {passed/len(tests)*100:.0f}%")
    print("="*70)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED - Phase 2 Ready!")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED - Please review errors above")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
