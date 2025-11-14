#!/usr/bin/env python3
"""
Standalone Data Quality Validation Runner
==========================================

Runs Great Expectations validation on master feature table.
Can be used independently or as part of CI/CD pipeline.

Usage:
    python scripts/run_data_quality_check.py
    python scripts/run_data_quality_check.py --data-path path/to/data.parquet

Exit Codes:
    0 - All validations passed
    1 - Some validations failed
    2 - Critical error (missing data, GX not setup, etc.)

Author: SmartGrocy Team
Date: 2025-11-15
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import great_expectations as gx
    import pandas as pd
    import logging
    from datetime import datetime
    from src.config import (
        OUTPUT_FILES, setup_logging
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run: pip install great-expectations==0.18.19")
    sys.exit(2)

setup_logging()
logger = logging.getLogger(__name__)

GX_ROOT = PROJECT_ROOT / "great_expectations"


def load_data(data_path: Path = None) -> pd.DataFrame:
    """Load master feature table"""
    if data_path is None:
        data_path = OUTPUT_FILES['master_feature_table']
    
    if not data_path.exists():
        # Try CSV fallback
        csv_path = OUTPUT_FILES['master_feature_table_csv']
        if csv_path.exists():
            data_path = csv_path
        else:
            logger.error(f"❌ Data not found: {data_path}")
            logger.error("   Run feature engineering first: python src/pipelines/_02_feature_enrichment.py")
            sys.exit(2)
    
    logger.info(f"Loading data from: {data_path}")
    
    try:
        if data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        sys.exit(2)


def check_gx_setup() -> bool:
    """Check if Great Expectations is setup"""
    if not GX_ROOT.exists():
        logger.error("❌ Great Expectations not setup")
        logger.error("   Run: python scripts/setup_great_expectations.py")
        return False
    
    # Check for expectation suite
    suite_path = GX_ROOT / "expectations" / "master_feature_table_suite.json"
    if not suite_path.exists():
        logger.error("❌ Expectation suite not found")
        logger.error(f"   Expected: {suite_path}")
        logger.error("   Run: python scripts/setup_great_expectations.py")
        return False
    
    return True


def run_validation(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Run GX validation"""
    logger.info("=" * 70)
    logger.info("RUNNING DATA QUALITY VALIDATION")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Dataset: {len(df):,} rows x {len(df.columns)} columns")
    
    # Get context
    try:
        context = gx.get_context(context_root_dir=str(GX_ROOT))
    except Exception as e:
        logger.error(f"❌ Failed to get GX context: {e}")
        sys.exit(2)
    
    # Create batch request
    batch_request = {
        "datasource_name": "master_feature_datasource",
        "data_connector_name": "default_runtime_data_connector",
        "data_asset_name": "master_feature_table",
        "runtime_parameters": {"batch_data": df},
        "batch_identifiers": {"default_identifier_name": "validation_run"}
    }
    
    # Run checkpoint
    logger.info("\nRunning validation checkpoint...")
    try:
        result = context.run_checkpoint(
            checkpoint_name="master_feature_checkpoint",
            batch_request=batch_request,
            expectation_suite_name="master_feature_table_suite"
        )
    except Exception as e:
        logger.error(f"❌ Checkpoint failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
    
    # Parse results
    success = result["success"]
    run_results = result.get("run_results", {})
    
    # Extract statistics
    validation_stats = {}
    failed_expectations = []
    
    for run_id, run_result in run_results.items():
        validation_result = run_result.get("validation_result", {})
        statistics = validation_result.get("statistics", {})
        results = validation_result.get("results", [])
        
        validation_stats = statistics
        
        # Collect failed expectations
        for exp_result in results:
            if not exp_result.get("success", True):
                failed_expectations.append({
                    "expectation": exp_result.get("expectation_config", {}).get("expectation_type"),
                    "column": exp_result.get("expectation_config", {}).get("kwargs", {}).get("column"),
                    "details": exp_result.get("result", {})
                })
    
    # Display results
    logger.info("\n" + "=" * 70)
    if success:
        logger.info("✓ VALIDATION PASSED")
    else:
        logger.warning("⚠ VALIDATION FAILED")
    logger.info("=" * 70)
    
    # Statistics
    logger.info("\nValidation Statistics:")
    logger.info(f"  Total expectations: {validation_stats.get('evaluated_expectations', 0)}")
    logger.info(f"  ✓ Passed: {validation_stats.get('successful_expectations', 0)}")
    logger.info(f"  ❌ Failed: {validation_stats.get('unsuccessful_expectations', 0)}")
    logger.info(f"  Success rate: {validation_stats.get('success_percent', 0):.1f}%")
    
    # Failed expectations details
    if failed_expectations and verbose:
        logger.info("\nFailed Expectations:")
        for i, fail in enumerate(failed_expectations[:10], 1):  # Show first 10
            logger.warning(f"  {i}. {fail['expectation']}")
            if fail['column']:
                logger.warning(f"     Column: {fail['column']}")
            if fail['details']:
                observed = fail['details'].get('observed_value')
                if observed is not None:
                    logger.warning(f"     Observed: {observed}")
        
        if len(failed_expectations) > 10:
            logger.warning(f"  ... and {len(failed_expectations) - 10} more")
    
    # Data docs link
    data_docs_path = GX_ROOT / "uncommitted" / "data_docs" / "local_site" / "index.html"
    if data_docs_path.exists():
        logger.info(f"\n✓ Detailed report: {data_docs_path}")
        logger.info("  Open in browser to view full results")
    
    return {
        "success": success,
        "statistics": validation_stats,
        "failed_expectations": failed_expectations,
        "data_docs_path": str(data_docs_path) if data_docs_path.exists() else None
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run data quality validation on master feature table"
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to data file (default: master_feature_table.parquet)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed failure information"
    )
    parser.add_argument(
        "--no-fail", action="store_true",
        help="Don't exit with error code on validation failure"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("SMARTGROCY DATA QUALITY CHECK")
    logger.info("=" * 70)
    
    # Check GX setup
    if not check_gx_setup():
        sys.exit(2)
    
    logger.info("✓ Great Expectations setup verified")
    
    # Load data
    data_path = Path(args.data_path) if args.data_path else None
    df = load_data(data_path)
    
    # Run validation
    result = run_validation(df, verbose=args.verbose)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    if result["success"]:
        logger.info("✓ All quality checks passed")
        logger.info("  Data is ready for model training")
        exit_code = 0
    else:
        logger.warning("⚠ Some quality checks failed")
        logger.warning(f"  {len(result['failed_expectations'])} expectations did not pass")
        logger.warning("  Review the detailed report for more information")
        exit_code = 1 if not args.no_fail else 0
    
    logger.info("=" * 70)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
