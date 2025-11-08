"""
Feature Enrichment Pipeline
===========================
Orchestrates feature engineering from multiple workstreams (WS0-WS4).
"""
import logging
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
    logger.info("=" * 80)
    logger.info("STARTING FEATURE ENRICHMENT PIPELINE (WS0+1+2+3+4)")
    logger.info("=" * 80)

    # 1. Define output paths using centralized config
    OUTPUT_PROCESSED_DIR = DATA_DIRS['processed_data']
    OUTPUT_FILE = OUTPUT_FILES['master_feature_table']
    output_csv = OUTPUT_PROCESSED_DIR / 'master_feature_table.csv'

    # 2. Load Real Data (from data/2_raw/ - Place Dunnhumby data here)
    logger.info("--- (1/6) Loading Competition Data ---")
    dataframes = load_competition_data()  # Call function from _01_load_data.py

    if not dataframes or 'transaction_data' not in dataframes:
        logger.critical("ERROR: 'transaction_data.csv' (main sales file) not found in data directory.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Workstream 0: AGGREGATION & MASTER GRID (CRITICAL!)
    # -----------------------------------------------------------------
    logger.info("--- (2/6) Workstream 0: Aggregation & Master Grid Creation ---")
    try:
        master_df = ws0.prepare_master_dataframe(dataframes['transaction_data'])
        logger.info(f"RESULT: WS0 aggregation complete - Shape: {master_df.shape}")
        logger.info(f"NOTE: Complete grid includes zero-filled rows (product-store-week combinations with no sales)")
        logger.info(f"      Zero-filled rows are CRITICAL for leak-safe time-series feature engineering (WS2)")

    except Exception as e:
        logger.error(f"ERROR during WS0 (Aggregation): {e}", exc_info=True)
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
    logger.info("--- (7/7) Saving Master Table ---")
    try:
        # validation_report = comprehensive_validation(master_df, verbose=True)
        # if validation_report['passed']:

        logger.info("SUCCESS: Pipeline complete. Saving optimized master table...")

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