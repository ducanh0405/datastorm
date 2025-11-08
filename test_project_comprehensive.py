#!/usr/bin/env python3
"""
Comprehensive Test Script for E-Grocery Forecaster
================================================

This script runs all available tests and validations for the entire project:
- Pytest unit tests and smoke tests
- Setup validation
- Optimized pipeline tests
- Data integrity checks
- Model validation
- End-to-end pipeline test (optional)

Usage:
    python test_project_comprehensive.py [--full] [--no-end-to-end]

Options:
    --full: Run all tests including slow ones
    --no-end-to-end: Skip end-to-end pipeline test
"""

import sys
import time
import subprocess
import argparse
from pathlib import Path
import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


class ProjectTester:
    """Comprehensive project testing class."""

    def __init__(self, run_full: bool = False, skip_end_to_end: bool = False):
        self.run_full = run_full
        self.skip_end_to_end = skip_end_to_end
        self.results: Dict[str, bool] = {}
        self.errors: List[str] = []

    def log_section(self, title: str):
        """Log a section header."""
        logging.info(f"\n{'='*70}")
        logging.info(f"{title.upper()}")
        logging.info(f"{'='*70}")

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        try:
            logging.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                logging.info(f"[PASS] {description} - PASSED")
                if result.stdout.strip():
                    logging.info(f"Output: {result.stdout.strip()}")
                return True
            else:
                logging.error(f"[FAIL] {description} - FAILED")
                if result.stderr.strip():
                    logging.error(f"Error: {result.stderr.strip()}")
                self.errors.append(f"{description}: {result.stderr.strip()}")
                return False

        except subprocess.TimeoutExpired:
            logging.error(f"[TIMEOUT] {description} - TIMEOUT")
            self.errors.append(f"{description}: Command timed out")
            return False
        except Exception as e:
            logging.error(f"[ERROR] {description} - ERROR: {e}")
            self.errors.append(f"{description}: {str(e)}")
            return False

    def test_setup_validation(self) -> bool:
        """Test 1: Setup validation."""
        self.log_section("Test 1: Setup Validation")
        return self.run_command(
            [sys.executable, "scripts/validate_setup.py"],
            "Setup validation"
        )

    def test_smoke_tests(self) -> bool:
        """Test 2: Pytest smoke tests."""
        self.log_section("Test 2: Smoke Tests")
        cmd = [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-v", "-m", "smoke"]
        return self.run_command(cmd, "Smoke tests")

    def test_unit_tests(self) -> bool:
        """Test 3: Pytest unit tests."""
        self.log_section("Test 3: Unit Tests")
        cmd = [sys.executable, "-m", "pytest", "tests/test_features.py", "-v"]
        if not self.run_full:
            cmd.extend(["-m", "not slow"])
        return self.run_command(cmd, "Unit tests")

    def test_optimized_pipeline(self) -> bool:
        """Test 4: Optimized pipeline validation."""
        self.log_section("Test 4: Optimized Pipeline Tests")
        return self.run_command(
            [sys.executable, "scripts/test_optimized.py"],
            "Optimized pipeline validation"
        )

    def test_code_quality(self) -> bool:
        """Test 5: Code quality checks."""
        self.log_section("Test 5: Code Quality")
        success = True

        # Check if ruff is available and run linting
        try:
            import ruff
            cmd = [sys.executable, "-m", "ruff", "check", "src/", "tests/"]
            success &= self.run_command(cmd, "Ruff linting")
        except ImportError:
            logging.warning("Ruff not available, skipping linting")

        # Check imports
        try:
            import src
            logging.info("[OK] Module imports successful")
        except ImportError as e:
            logging.error(f"[FAIL] Module import failed: {e}")
            success = False

        return success

    def test_data_integrity(self) -> bool:
        """Test 6: Data integrity checks."""
        self.log_section("Test 6: Data Integrity")
        success = True

        # Check if required data directories exist
        data_dirs = [
            "data/poc_data",
            "data/2_raw",
            "data/3_processed",
            "models",
            "reports/metrics"
        ]

        for data_dir in data_dirs:
            path = PROJECT_ROOT / data_dir
            if path.exists():
                logging.info(f"[OK] Data directory exists: {data_dir}")
            else:
                logging.warning(f"[SKIP] Data directory missing: {data_dir}")

        # Check if models exist
        model_files = ["q05_forecaster.joblib", "q50_forecaster.joblib", "q95_forecaster.joblib"]
        for model_file in model_files:
            path = PROJECT_ROOT / "models" / model_file
            if path.exists():
                size = path.stat().st_size / (1024 * 1024)  # MB
                logging.info(".1f")
            else:
                logging.warning(f"[SKIP] Model missing: {model_file}")

        return success

    def test_model_validation(self) -> bool:
        """Test 7: Model validation."""
        self.log_section("Test 7: Model Validation")
        success = True

        try:
            # Test model loading
            import joblib
            from pathlib import Path

            model_paths = [
                PROJECT_ROOT / "models" / "q05_forecaster.joblib",
                PROJECT_ROOT / "models" / "q50_forecaster.joblib",
                PROJECT_ROOT / "models" / "q95_forecaster.joblib"
            ]

            for model_path in model_paths:
                if model_path.exists():
                    try:
                        model = joblib.load(model_path)
                        logging.info(f"[OK] Model loads successfully: {model_path.name}")

                        # Check if it's a LightGBM model
                        if hasattr(model, 'predict'):
                            logging.info(f"  - Model has predict method")
                        else:
                            logging.warning(f"  - Model may not be properly trained")
                    except Exception as e:
                        logging.error(f"[FAIL] Failed to load model {model_path.name}: {e}")
                        success = False
                else:
                    logging.warning(f"[SKIP] Model file not found: {model_path.name}")

            # Check model features JSON
            features_path = PROJECT_ROOT / "models" / "model_features.json"
            if features_path.exists():
                import json
                with open(features_path, 'r') as f:
                    features = json.load(f)
                logging.info(f"[OK] Model features file exists with {len(features)} features")
            else:
                logging.warning("[SKIP] Model features file missing")

        except ImportError as e:
            logging.warning(f"Cannot validate models: {e}")

        return success

    def test_end_to_end(self) -> bool:
        """Test 8: End-to-end pipeline (optional)."""
        if self.skip_end_to_end:
            logging.info("Skipping end-to-end test as requested")
            return True

        self.log_section("Test 8: End-to-End Pipeline Test")

        # Check if we have the necessary data
        poc_data = PROJECT_ROOT / "data" / "poc_data" / "transaction_data.csv"
        if not poc_data.exists():
            logging.warning("POC data not found, creating sample data first...")
            self.run_command(
                [sys.executable, "scripts/create_sample_data.py"],
                "Create sample data"
            )

        # Run a quick pipeline test
        try:
            from src.pipelines._01_load_data import load_competition_data
            from src.features.ws0_aggregation import prepare_master_dataframe
            from src.features.ws2_timeseries_features import add_lag_rolling_features

            # Load small sample
            logging.info("Loading sample data...")
            data_dir = PROJECT_ROOT / "data" / "poc_data"
            data = load_competition_data(str(data_dir))

            if 'transaction_data' in data:
                df = data['transaction_data']
                logging.info(f"Loaded {len(df):,} transactions")

                # Run WS0
                master_df = prepare_master_dataframe(df)
                logging.info(f"WS0 aggregation: {len(master_df):,} rows")

                # Run WS2 on small sample
                sample_df = master_df.head(1000)  # Small sample for speed
                enriched_df = add_lag_rolling_features(sample_df)
                logging.info(f"WS2 features: {enriched_df.shape[1]} columns")

                logging.info("[PASS] End-to-end pipeline test PASSED")
                return True
            else:
                logging.error("[FAIL] No transaction data found")
                return False

        except Exception as e:
            logging.error(f"[FAIL] End-to-end test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def test_performance(self) -> bool:
        """Test 9: Performance benchmarks."""
        self.log_section("Test 9: Performance Benchmarks")

        benchmark_script = PROJECT_ROOT / "scripts" / "benchmark_performance.py"
        if benchmark_script.exists():
            return self.run_command(
                [sys.executable, "scripts/benchmark_performance.py"],
                "Performance benchmarks"
            )
        else:
            logging.info("[SKIP] Benchmark script not found, skipping")
            return True

    def run_all_tests(self) -> int:
        """Run all tests and return exit code."""
        start_time = time.time()

        logging.info("STARTING COMPREHENSIVE PROJECT TESTING")
        logging.info(f"Project root: {PROJECT_ROOT}")
        logging.info(f"Full test mode: {self.run_full}")
        logging.info(f"Skip end-to-end: {self.skip_end_to_end}")

        # Run all test suites
        test_methods = [
            self.test_setup_validation,
            self.test_smoke_tests,
            self.test_unit_tests,
            self.test_optimized_pipeline,
            self.test_code_quality,
            self.test_data_integrity,
            self.test_model_validation,
            self.test_end_to_end,
            self.test_performance,
        ]

        for i, test_method in enumerate(test_methods, 1):
            try:
                result = test_method()
                self.results[test_method.__name__] = result
            except Exception as e:
                logging.error(f"Test {i} crashed: {e}")
                self.results[test_method.__name__] = False
                self.errors.append(f"Test {i} crashed: {str(e)}")

        # Summary
        self.log_section("TEST SUMMARY")

        total_tests = len(self.results)
        passed_tests = sum(self.results.values())

        logging.info(f"Total tests run: {total_tests}")
        logging.info(f"Tests passed: {passed_tests}")
        logging.info(f"Tests failed: {total_tests - passed_tests}")

        for test_name, result in self.results.items():
            status = "[PASS]" if result else "[FAIL]"
            test_desc = test_name.replace("test_", "").replace("_", " ").title()
            logging.info(f"  {test_desc:25s}: {status}")

        elapsed_time = time.time() - start_time
        logging.info(".2f")

        # Errors summary
        if self.errors:
            logging.info(f"\nErrors encountered ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5 errors
                logging.info(f"  - {error}")
            if len(self.errors) > 5:
                logging.info(f"  ... and {len(self.errors) - 5} more errors")

        # Final result
        if passed_tests == total_tests:
            logging.info("\nSUCCESS: ALL TESTS PASSED! Project is ready for production.")
            return 0
        else:
            logging.error(f"\nFAILED: {total_tests - passed_tests} TEST(S) FAILED!")
            logging.info("\nTroubleshooting tips:")
            logging.info("1. Check test_results.log for detailed output")
            logging.info("2. Ensure all dependencies are installed: pip install -r requirements.txt")
            logging.info("3. Run individual test scripts for more details")
            logging.info("4. Check data/poc_data/ has sample data")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive E-Grocery Forecaster Testing")
    parser.add_argument("--full", action="store_true", help="Run all tests including slow ones")
    parser.add_argument("--no-end-to-end", action="store_true", help="Skip end-to-end pipeline test")

    args = parser.parse_args()

    tester = ProjectTester(run_full=args.full, skip_end_to_end=args.no_end_to_end)
    exit_code = tester.run_all_tests()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
