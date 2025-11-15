#!/usr/bin/env python3
"""
Run All Tests Script
====================
Chạy tất cả test cases cho SmartGrocy project.

Usage:
    python run_all_tests.py              # Chạy tất cả test
    python run_all_tests.py --smoke      # Chỉ smoke tests
    python run_all_tests.py --config     # Chỉ config validation tests
    python run_all_tests.py --features   # Chỉ feature tests
    python run_all_tests.py --verbose    # Verbose output
"""
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*70)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        return False

def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SmartGrocy tests')
    parser.add_argument('--smoke', action='store_true', help='Run only smoke tests')
    parser.add_argument('--config', action='store_true', help='Run only config validation tests')
    parser.add_argument('--features', action='store_true', help='Run only feature tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    
    args = parser.parse_args()
    
    # Check pytest is available
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is not installed. Please install it:")
        print("  pip install pytest")
        sys.exit(1)
    
    # Step 1: Validate config first
    print("\n" + "="*70)
    print("STEP 1: Validating Configuration")
    print("="*70)
    try:
        from src.config import assert_config_valid
        assert_config_valid()
        print("[OK] Config validation passed")
    except Exception as e:
        print(f"[FAIL] Config validation failed: {e}")
        print("\nPlease fix configuration errors before running tests.")
        sys.exit(1)
    
    # Build pytest command
    pytest_cmd = ['python', '-m', 'pytest']
    
    if args.verbose:
        pytest_cmd.append('-v')
    else:
        pytest_cmd.append('-q')
    
    if args.coverage:
        pytest_cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])
    
    # Determine which tests to run
    tests_passed = True
    
    if args.smoke:
        pytest_cmd.extend(['tests/test_smoke.py', '-m', 'smoke'])
        tests_passed = run_command(pytest_cmd, "Smoke Tests")
    
    elif args.config:
        pytest_cmd.append('tests/test_config_validation.py')
        tests_passed = run_command(pytest_cmd, "Config Validation Tests")
    
    elif args.features:
        pytest_cmd.append('tests/test_features.py')
        tests_passed = run_command(pytest_cmd, "Feature Engineering Tests")
    
    elif args.integration:
        pytest_cmd.append('tests/test_integration.py')
        tests_passed = run_command(pytest_cmd, "Integration Tests")
    
    else:
        # Run all tests in sequence
        test_suites = [
            ('tests/test_config_validation.py', 'Config Validation Tests'),
            ('tests/test_features.py', 'Feature Engineering Tests'),
            ('tests/test_optional_dependencies.py', 'Optional Dependencies Tests'),
            ('tests/test_smoke.py', 'Smoke Tests'),
            ('tests/test_integration.py', 'Integration Tests'),
            ('tests/test_phase2_integration.py', 'Phase 2 Integration Tests'),
        ]
        
        for test_file, description in test_suites:
            cmd = pytest_cmd + [test_file]
            if not run_command(cmd, description):
                tests_passed = False
                print(f"\n⚠️  {description} had failures, but continuing...")
    
    # Summary
    print("\n" + "="*70)
    if tests_passed:
        print("[SUCCESS] ALL TESTS PASSED")
    else:
        print("[WARNING] SOME TESTS FAILED - Please check output above")
    print("="*70)
    
    sys.exit(0 if tests_passed else 1)

if __name__ == '__main__':
    main()

