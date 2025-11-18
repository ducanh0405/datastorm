"""
Feature Enrichment Pipeline (Config-Driven Orchestrator)
=========================================================
Orchestrates feature engineering from ALL workstreams (WS0-WS6)
based on the active dataset's configuration.
"""
import logging
import sys

# Import centralized configuration and setup
try:
    from src.config import (
        OUTPUT_FILES,
        PERFORMANCE_CONFIG,
        ensure_directories,
        setup_logging,
        setup_project_path,
    )
    setup_project_path()
    setup_logging()
    ensure_directories()
    logger = logging.getLogger(__name__)

    # 1. Import data loader
    # 2. Import ALL feature engineering modules
    from src.features import ws0_aggregation as ws0
    from src.features import ws1_relational_features as ws1
    from src.features import ws2_timeseries_features as ws2
    from src.features import ws3_behavior_features as ws3
    from src.features import ws4_price_features as ws4
    from src.features import ws5_stockout_recovery as ws5
    from src.features import ws6_weather_features as ws6
    from src.pipelines._01_load_data import load_data

    # 4. Import performance monitoring for stage-level timing
    from src.utils.performance_monitor import performance_monitor

    # 3. Import validation
    from src.utils.validation import comprehensive_validation

except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"ERROR IMPORTING: {e}", exc_info=True)
    sys.exit(1)

def main() -> None:
    """
    Main pipeline orchestrator.
    This function is fully config-driven.
    """
    logger.info("=" * 70)
    logger.info("STARTING FEATURE ENRICHMENT PIPELINE")
    logger.info("=" * 70)

    # 1. Load Data (Hàm này giờ trả về dataframes VÀ config)
    try:
        dataframes, config = load_data()
    except Exception as e:
        logger.critical(f"FATAL: Data loading failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Extract Base Dataframe
    sales_df = dataframes.get('sales')
    if sales_df is None:
        logger.critical("FATAL: 'sales' dataframe (hoặc 'transaction_data') not found. Halting.")
        sys.exit(1)

    logger.info(f"Base dataframe 'sales' loaded. Shape: {sales_df.shape}")
    logger.info("--- (2/8) Workstream 0: Aggregation & Master Grid ---")
    try:
        # WS0: Luôn luôn chạy
        master_df = ws0.create_master_grid(sales_df, config)
        logger.info(f"✓ WS0 complete - Shape: {master_df.shape}")
    except Exception as e:
        logger.error(f"ERROR during WS0 (Aggregation): {e}", exc_info=True)
        sys.exit(1) # WS0 là bắt buộc

    # --- Conditional Feature Engineering ---

    # 3. WS1: Relational (Product, Household)
    logger.info("--- (3/8) Workstream 1: Relational Features ---")
    if config['has_relational']:
        try:
            master_df = ws1.enrich_relational_features(master_df, dataframes)
            logger.info(f"✓ WS1 complete - Shape: {master_df.shape}")
        except Exception as e:
            logger.warning(f"SKIPPING WS1: Error occurred: {e}", exc_info=True)
    else:
        logger.info("SKIPPING WS1: 'has_relational' is False in config.")

    # 4. WS5: Stockout Recovery (Chạy TRƯỚC WS2)
    logger.info("--- (4/8) Workstream 5: Stockout Recovery ---")
    if config['has_stockout']:
        try:
            # WS5 functions handle grouping internally
            logger.info("Running WS5 (Latent Demand)...")
            master_df = ws5.recover_latent_demand(master_df, config)

            logger.info("Running WS5 (Stockout Features)...")
            master_df = ws5.add_stockout_features(master_df, config)

            logger.info(f"✓ WS5 complete - Shape: {master_df.shape}")
        except Exception as e:
            logger.warning(f"SKIPPING WS5: Error occurred: {e}", exc_info=True)
    else:
        logger.info("SKIPPING WS5: 'has_stockout' is False in config.")

    # 5. WS6: Weather Features
    logger.info("--- (5/8) Workstream 6: Weather Features ---")
    if config['has_weather']:
        if 'weather' in dataframes and dataframes['weather'] is not None:
            try:
                # Monitor WS6 performance (optional workstream)
                with performance_monitor.time_operation('WS6_weather_features', {'has_weather': True}):
                    master_df = ws6.merge_weather_data(master_df, dataframes['weather'])
                    master_df = ws6.create_weather_features(master_df)
                logger.info(f"✓ WS6 complete - Shape: {master_df.shape}")
            except Exception as e:
                logger.warning(f"SKIPPING WS6: Error occurred: {e}", exc_info=True)
        else:
            logger.warning("SKIPPING WS6: 'has_weather' is True, but 'weather' data not found.")
    else:
        logger.info("SKIPPING WS6: 'has_weather' is False in config.")

    # 6. WS2: Time-Series (Luôn chạy, hàm bên trong đã config-driven)
    logger.info("--- (6/8) Workstream 2: Time-Series Features ---")
    try:
        # Monitor WS2 performance (typically the slowest workstream)
        with performance_monitor.time_operation('WS2_timeseries_features', {
            'lag_periods': config.get('lag_periods', []),
            'rolling_windows': config.get('rolling_windows', [])
        }):
            # Hàm này sẽ tự động đọc config để chạy đúng lags, rolling, intraday
            master_df = ws2.add_timeseries_features_config(master_df, config)
        logger.info(f"✓ WS2 complete - Shape: {master_df.shape}")
    except Exception as e:
        logger.error(f"ERROR during WS2 (Time-Series): {e}", exc_info=True)
        sys.exit(1) # WS2 là bắt buộc

    # 7. WS3: Behavior (Clickstream)
    logger.info("--- (7/8) Workstream 3: Behavior Features ---")
    if config['has_behavior']:
        if 'clickstream_log' in dataframes and dataframes['clickstream_log'] is not None:
            try:
                master_df = ws3.add_behavioral_features(master_df, dataframes)
                logger.info(f"✓ WS3 complete - Shape: {master_df.shape}")
            except Exception as e:
                logger.warning(f"SKIPPING WS3: Error occurred: {e}", exc_info=True)
        else:
            logger.warning("SKIPPING WS3: 'has_behavior' is True, but 'clickstream_log' data not found.")
    else:
        logger.info("SKIPPING WS3: 'has_behavior' is False in config.")

    # 8. WS4: Price & Promotion
    logger.info("--- (8/8) Workstream 4: Price/Promotion Features ---")
    if config['has_price_promo']:
        if 'causal_data' in dataframes and dataframes['causal_data'] is not None:
            try:
                master_df = ws4.add_price_promotion_features(master_df, dataframes)
                logger.info(f"✓ WS4 complete - Shape: {master_df.shape}")
            except Exception as e:
                logger.warning(f"SKIPPING WS4: Error occurred: {e}", exc_info=True)
        else:
            logger.warning("SKIPPING WS4: 'has_price_promo' is True, but 'causal_data' data not found.")
    else:
        logger.info("SKIPPING WS4: 'has_price_promo' is False in config.")

    # --- Final Validation and Storage ---
    logger.info("--- (9/9) Validation & Saving Master Table ---")
    try:
        validation_report = comprehensive_validation(master_df, verbose=True)
        if not validation_report.get('passed', False):
            logger.warning("VALIDATION WARNING: Issues found in master table.")
            for issue in validation_report.get('issues', []): logger.warning(f"  - {issue}")
        else:
            logger.info("VALIDATION: All checks passed.")

        # Save as Parquet (efficient format)
        OUTPUT_FILE_PARQUET = OUTPUT_FILES['master_feature_table']
        master_df.to_parquet(OUTPUT_FILE_PARQUET, index=False, compression='snappy')
        logger.info(f"SUCCESS: Master Table (Parquet) saved to: {OUTPUT_FILE_PARQUET}")

        # Save as CSV (human-readable format)
        OUTPUT_FILE_CSV = OUTPUT_FILES['master_feature_table_csv']
        num_rows = master_df.shape[0]

        if num_rows > 1000000:
            logger.warning(f"Large dataset detected ({num_rows:,} rows). CSV export may take several minutes...")

        logger.info("Saving Master Table as CSV (this may take a moment for large datasets)...")
        try:
            # Use chunking for very large datasets to avoid memory issues
            if num_rows > 5000000:
                logger.info("Using chunked CSV writing for very large dataset...")
                chunk_size = 1000000
                master_df.iloc[:0].to_csv(OUTPUT_FILE_CSV, index=False, mode='w')  # Write header
                for i in range(0, num_rows, chunk_size):
                    chunk = master_df.iloc[i:i+chunk_size]
                    chunk.to_csv(OUTPUT_FILE_CSV, index=False, mode='a', header=False)
                    logger.info(f"  Written {min(i+chunk_size, num_rows):,} / {num_rows:,} rows...")
            else:
                master_df.to_csv(OUTPUT_FILE_CSV, index=False)

            logger.info(f"SUCCESS: Master Table (CSV) saved to: {OUTPUT_FILE_CSV}")
        except Exception as e:
            logger.error(f"WARNING: Failed to save CSV file: {e}")
            logger.info("Parquet file is still available for use.")

        logger.info(f"STATS: Final Shape: {master_df.shape[0]:,} rows, {master_df.shape[1]} columns")
        logger.info(f"STATS: Memory: {master_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    except Exception as e:
        logger.error(f"ERROR: Pipeline failed at final save step: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("FEATURE ENRICHMENT PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
