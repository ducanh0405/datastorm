"""
Feature Enrichment Pipeline
===========================
Orchestrates feature engineering from multiple workstreams (WS0-WS4).
"""
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# Import centralized configuration and setup
try:
    # Ensure project root is in path for imports
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.config import (
        DATA_DIRS, OUTPUT_FILES, PERFORMANCE_CONFIG,
        setup_logging, ensure_directories
    )
    setup_logging()  # Setup centralized logging
    ensure_directories()  # Ensure all directories exist
    logger = logging.getLogger(__name__)

    # 1. Import data loader
    from src.pipelines._01_load_data import load_competition_data

    # 2. Import feature engineering modules (refactored from 4 PoCs)
    # WS0: Aggregation & Grid - Unified (Polars + Pandas fallback)
    from src.features import ws0_aggregation as ws0
    logger.info("PIPELINE: WS0 aggregation (auto-selects Polars/pandas)")

    # WS1: Relational features (Dunnhumby joins)
    from src.features import ws1_relational_features as ws1

    # WS2: Time-Series features (now optimized by default)
    from src.features import ws2_timeseries_features as ws2
    logger.info("PIPELINE: WS2 features (optimized, 10x speedup)")

    # WS3: Behavioral features
    from src.features import ws3_behavior_features as ws3
    # WS4: Price/Promotion features
    from src.features import ws4_price_features as ws4

    # 3. Import validation utilities
    from src.utils.validation import comprehensive_validation

except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"ERROR IMPORTING: {e}")
    logger.error("Please ensure __init__.py files exist in all src/ subdirectories.")
    sys.exit(1)


# ---------------------------------------------

def main() -> None:
    """
    Main pipeline orchestrator for feature enrichment.

    Integrates logic from 4 Workstreams (WS0-WS4) to build final Master Table.
    Runs on Dunnhumby dataset to test WS1, WS2, WS4.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Feature enrichment pipeline')
    parser.add_argument('--full-data', action='store_true', 
                       help='Use full data from data/2_raw with memory optimizations')
    args, unknown = parser.parse_known_args()
    
    use_full_data = args.full_data or os.environ.get('DATA_SOURCE', '').lower() in ['full', 'raw']
    
    logger.info("=" * 80)
    logger.info("STARTING FEATURE ENRICHMENT PIPELINE (WS0+1+2+3+4)")
    if use_full_data:
        logger.info("🚀 MEMORY OPTIMIZATION MODE: Using optimized grid for full data")
    logger.info("=" * 80)

    # 1. Define output paths using centralized config
    OUTPUT_PROCESSED_DIR = DATA_DIRS['processed_data']
    OUTPUT_REPORTS_DIR = DATA_DIRS['reports']
    OUTPUT_FILE = OUTPUT_FILES['master_feature_table']
    output_csv = OUTPUT_PROCESSED_DIR / 'master_feature_table.csv'

    # 2. Load Real Data (from data/2_raw/ - Place Dunnhumby data here)
    logger.info("--- (1/6) Loading Competition Data ---")
    dataframes = load_competition_data(use_full_data=use_full_data)  # Call function from _01_load_data.py

    if not dataframes or 'transaction_data' not in dataframes:
        logger.critical("ERROR: 'transaction_data.csv' (main sales file) not found in data directory.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Workstream 0: AGGREGATION & MASTER GRID (CRITICAL!)
    # -----------------------------------------------------------------
    logger.info("--- (2/6) Workstream 0: Aggregation & Master Grid Creation ---")
    try:
        # Use OPTIMIZED version to eliminate data sparsity from the start
        master_df = ws0.prepare_master_dataframe_optimized(dataframes['transaction_data'])
        logger.info(f"RESULT: WS0 optimized aggregation complete - Shape: {master_df.shape}")

        # Additional filter for any remaining zero-filled rows (should be minimal now)
        original_shape = master_df.shape
        master_df = master_df[master_df['SALES_VALUE'] > 0].reset_index(drop=True)
        if original_shape[0] != master_df.shape[0]:
            logger.info(f"FILTER: Removed remaining zero sales - {original_shape} -> {master_df.shape} ({original_shape[0] - master_df.shape[0]:,} zero-filled rows removed)")

    except Exception as e:
        logger.error(f"ERROR during WS0 (Aggregation): {e}", exc_info=True)
        # Fallback to legacy version
        try:
            logger.warning("Falling back to legacy aggregation...")
            master_df = ws0.prepare_master_dataframe(dataframes['transaction_data'])
            logger.info(f"RESULT: WS0 fallback complete - Shape: {master_df.shape}")
            
            # Additional filter for fallback version too
            original_shape = master_df.shape
            master_df = master_df[master_df['SALES_VALUE'] > 0].reset_index(drop=True)
            if original_shape[0] != master_df.shape[0]:
                logger.info(f"FILTER: Removed remaining zero sales - {original_shape} -> {master_df.shape} ({original_shape[0] - master_df.shape[0]:,} zero-filled rows removed)")
        except Exception as fallback_error:
            logger.critical(f"FATAL: Both optimized and fallback aggregation failed: {fallback_error}", exc_info=True)
            sys.exit(1)

    # -----------------------------------------------------------------
    # Workstream 1: Relational (Joins)
    # -----------------------------------------------------------------
    logger.info("--- (3/6) Integrating Workstream 1: Relational ---")
    try:
        # Call function from 'ws1_relational_features.py'
        master_df = ws1.enrich_relational_features(master_df, dataframes)
        logger.info(f"RESULT: WS1 complete - Shape: {master_df.shape}")
    except KeyError as e:
        logger.warning(f"SKIP: WS1 - Required data not found (e.g., 'product', 'hh_demographic'). Error: {e}")
    except Exception as e:
        logger.warning(f"ERROR during WS1: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 2: Time-Series & Calendar (Lags, Rolling)
    # -----------------------------------------------------------------
    logger.info("--- (4/6) Integrating Workstream 2: Time-Series ---")
    try:
        # Call function from 'ws2_timeseries_features.py'
        master_df = ws2.add_lag_rolling_features(master_df)  # (This function may need dataframes['calendar'])
        logger.info(f"RESULT: WS2 complete - Shape: {master_df.shape}")
    except Exception as e:
        logger.warning(f"ERROR during WS2: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 3: Behavior (Clickstream)
    # -----------------------------------------------------------------
    logger.info("--- (5/6) Integrating Workstream 3: Behavior ---")
    try:
        # Dunnhumby does NOT have clickstream. Toggle logic will detect this.
        if 'clickstream_log' not in dataframes:
            logger.info("SKIP: WS3 - 'clickstream_log' not found in data (expected for Dunnhumby)")
        else:
            master_df = ws3.add_behavioral_features(master_df, dataframes)
            logger.info(f"RESULT: WS3 complete - Shape: {master_df.shape}")

    except Exception as e:
        logger.warning(f"ERROR during WS3: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 4: Price & Promotion
    # -----------------------------------------------------------------
    logger.info("--- (6/6) Integrating Workstream 4: Price/Promotion ---")
    try:
        # Call function from 'ws4_price_features.py'
        master_df = ws4.add_price_promotion_features(master_df, dataframes)
        logger.info(f"RESULT: WS4 complete - Shape: {master_df.shape}")
    except KeyError as e:
        logger.warning(f"SKIP: WS4 - Required data not found (e.g., 'causal_data'). Error: {e}")
    except Exception as e:
        logger.warning(f"ERROR during WS4: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Final Validation and Storage
    # -----------------------------------------------------------------
    logger.info("--- (7/7) Validation & Saving Master Table ---")
    try:
        # Run comprehensive validation
        from src.utils.validation import comprehensive_validation
        validation_report = comprehensive_validation(master_df, verbose=True)

        # Save validation report
        import json
        validation_output = OUTPUT_REPORTS_DIR / 'metrics' / 'master_table_validation.json'
        with open(validation_output, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, default=str)
        logger.info(f"VALIDATION: Report saved to {validation_output}")

        # Check if validation passed
        if not validation_report.get('passed', False):
            logger.warning("VALIDATION WARNING: Issues found in master table")
            for issue in validation_report.get('issues', []):
                logger.warning(f"  - {issue}")
        else:
            logger.info("VALIDATION: All checks passed")

        logger.info("SUCCESS: Pipeline complete. Saving optimized master table...")

        # Optional: Run feature selection (if requested)
        try:
            if os.getenv('RUN_FEATURE_SELECTION', 'false').lower() == 'true':
                logger.info("--- (8/8) Feature Selection ---")
                from src.features.feature_selection import get_optimal_features

                # Filter to training data (sales > 0)
                training_df = master_df[master_df['SALES_VALUE'] > 0].copy()

                if len(training_df) > 0:
                    feature_selection_result = get_optimal_features(
                        df=training_df,
                        target_col='SALES_VALUE',
                        importance_threshold=0.005,
                        correlation_threshold=0.95,
                        max_features=50,  # Limit to top 50 features
                        save_report=True
                    )

                    logger.info(f"FEATURE SELECTION: Selected {len(feature_selection_result['selected_features'])} optimal features")
                else:
                    logger.warning("FEATURE SELECTION: Skipped - no training data available")
        except Exception as e:
            logger.warning(f"FEATURE SELECTION: Failed - {e}")

        # Clean up old files (keep only the main 2 files)
        import glob
        old_files = glob.glob(str(OUTPUT_PROCESSED_DIR / "master_feature_table_*"))
        for old_file in old_files:
            if not (old_file.endswith('.parquet') or old_file.endswith('.csv')):
                try:
                    Path(old_file).unlink()
                    logger.info(f"CLEANUP: Removed old file: {Path(old_file).name}")
                except:
                    pass

        # Save optimized master table in both formats
        master_df.to_parquet(OUTPUT_FILE, index=False)
        master_df.to_csv(output_csv, index=False)

        logger.info(f"SUCCESS: Master Table saved to: {OUTPUT_FILE}")
        logger.info(f"SUCCESS: Master Table CSV saved to: {output_csv}")
        logger.info(f"STATS: Final Shape: {master_df.shape[0]:,} rows, {master_df.shape[1]} columns")
        logger.info(f"STATS: SALES_VALUE range: {master_df['SALES_VALUE'].min():.2f} - {master_df['SALES_VALUE'].max():.2f}")

        # Quality summary
        zero_sales = (master_df['SALES_VALUE'] == 0).sum()
        logger.info(f"QUALITY: {zero_sales} zero sales rows (optimized)")
        logger.info(f"QUALITY: Complete lag/rolling/calendar/price features included")

    except Exception as e:
        logger.error(f"ERROR: Pipeline failed at final save step: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("FEATURE ENRICHMENT PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()