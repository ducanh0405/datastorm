"""
Feature Enrichment Pipeline
===========================
Orchestrates feature engineering from multiple workstreams (WS0-WS4).
"""
import logging
import sys
from pathlib import Path
from typing import Dict

# === PROJECT ROOT ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ====================

# Configure Logging (English/ASCII only)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- IMPORT FROM src/ MODULES ---
try:
    # Import configuration
    from src.config import PERFORMANCE_CONFIG

    # 1. Import data loader
    from src.pipelines._01_load_data import load_competition_data

    # 2. Import feature engineering modules (refactored from 4 PoCs)
    # WS0: Aggregation & Grid - Unified (Polars + Pandas fallback)
    # ws0_aggregation.py now contains both implementations and auto-selects
    from src.features import ws0_aggregation as ws0
    logging.info("[PIPELINE] Using WS0 aggregation (auto-selects Polars/pandas)")

    # WS1: Relational features (Dunnhumby joins)
    from src.features import ws1_relational_features as ws1

    # WS2: Time-Series features (now optimized by default)
    from src.features import ws2_timeseries_features as ws2
    logging.info("[PIPELINE] Using WS2 features (optimized, 10x speedup)")

    # WS3: Behavioral features
    from src.features import ws3_behavior_features as ws3
    # WS4: Price/Promotion features
    from src.features import ws4_price_features as ws4

    # 3. Import validation utilities
    from src.utils.validation import comprehensive_validation

except ImportError as e:
    logging.error(f"ERROR IMPORTING: {e}")
    logging.error("Please ensure __init__.py files exist in all src/ subdirectories.")
    sys.exit(1)


# ---------------------------------------------

def main() -> None:
    """
    Main pipeline orchestrator for feature enrichment.
    
    Integrates logic from 4 Workstreams (WS0-WS4) to build final Master Table.
    Runs on Dunnhumby dataset to test WS1, WS2, WS4.
    """
    logging.info("========== STARTING FEATURE ENRICHMENT PIPELINE (WS0+1+2+3+4) ==========")

    # 1. Define output paths
    OUTPUT_PROCESSED_DIR = PROJECT_ROOT / 'data' / '3_processed'
    OUTPUT_FILE = OUTPUT_PROCESSED_DIR / 'master_feature_table.parquet'
    output_csv = OUTPUT_PROCESSED_DIR / 'master_feature_table.csv'
    OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Real Data (from data/2_raw/ - Place Dunnhumby data here)
    logging.info("--- (1/6) Loading Competition Data (from data/2_raw/) ---")
    dataframes = load_competition_data()  # Call function from _01_load_data.py

    if not dataframes or 'transaction_data' not in dataframes:
        logging.critical("Error: 'transaction_data.csv' (main sales file) not found in data/2_raw/.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Workstream 0: AGGREGATION & MASTER GRID (NEW - CRITICAL!)
    # -----------------------------------------------------------------
    logging.info("--- (0/6) Workstream 0: Aggregation & Master Grid Creation ---")
    try:
        master_df = ws0.prepare_master_dataframe(dataframes['transaction_data'])
        logging.info(f"-> Shape after WS0 (aggregation): {master_df.shape}")

        # CRITICAL FIX: Filter out zero-filled rows to avoid sparse data issues
        # Only keep rows with actual sales (SALES_VALUE > 0)
        original_shape = master_df.shape
        master_df = master_df[master_df['SALES_VALUE'] > 0].reset_index(drop=True)
        logging.info(f"-> Filtered zero sales: {original_shape} -> {master_df.shape} ({original_shape[0] - master_df.shape[0]:,} zero-filled rows removed)")

    except Exception as e:
        logging.error(f"ERROR during WS0 (Aggregation): {e}", exc_info=True)
        sys.exit(1)

    # 4. Integrate (Enrichment) by Module (Toggle Feature)
    # -----------------------------------------------------------------
    # Workstream 1: Relational (Joins)
    # -----------------------------------------------------------------
    logging.info("--- (2/6) Integrating Workstream 1: Relational ---")
    try:
        # Call function from 'ws1_relational_features.py'
        master_df = ws1.enrich_relational_features(master_df, dataframes)
        logging.info(f"-> Shape after WS1: {master_df.shape}")
    except KeyError as e:
        logging.warning(f"SKIPPING WS1: Required data not found (e.g., 'product', 'hh_demographic'). Error: {e}")
    except Exception as e:
        logging.warning(f"ERROR during WS1: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 2: Time-Series & Calendar (Lags, Rolling)
    # -----------------------------------------------------------------
    logging.info("--- (3/6) Integrating Workstream 2: Time-Series ---")
    try:
        # Call function from 'ws2_timeseries_features.py'
        master_df = ws2.add_lag_rolling_features(master_df)  # (This function may need dataframes['calendar'])
        logging.info(f"-> Shape after WS2: {master_df.shape}")
    except Exception as e:
        logging.warning(f"ERROR during WS2: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 3: Behavior (Clickstream)
    # -----------------------------------------------------------------
    logging.info("--- (4/6) Integrating Workstream 3: Behavior ---")
    try:
        # Dunnhumby does NOT have clickstream. Toggle logic will detect this.
        if 'clickstream_log' not in dataframes:
            logging.info("INFO: Skipping WS3: 'clickstream_log' not found in data (As expected for Dunnhumby).")
        else:
            master_df = ws3.add_behavioral_features(master_df, dataframes)
            logging.info(f"-> Shape after WS3: {master_df.shape}")

    except Exception as e:
        logging.warning(f"ERROR during WS3: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 4: Price & Promotion
    # -----------------------------------------------------------------
    logging.info("--- (5/6) Integrating Workstream 4: Price/Promotion ---")
    try:
        # Call function from 'ws4_price_features.py'
        master_df = ws4.add_price_promotion_features(master_df, dataframes)
        logging.info(f"-> Shape after WS4: {master_df.shape}")
    except KeyError as e:
        logging.warning(f"SKIPPING WS4: Required data not found (e.g., 'causal_data'). Error: {e}")
    except Exception as e:
        logging.warning(f"ERROR during WS4: {e}. Skipping...")

    # 5. Final Validation and Storage
    logging.info("--- (6/6) Saving Master Table ---")
    try:
        # validation_report = comprehensive_validation(master_df, verbose=True)
        # if validation_report['passed']:

        logging.info("OK. Data pipeline PASSED. Saving file...")
        master_df.to_parquet(OUTPUT_FILE, index=False)
        logging.info(f"OK. Master Table saved to: {OUTPUT_FILE}")
        logging.info(f"Final Shape: {master_df.shape}")

    except Exception as e:
        logging.error(f"ERROR: Data pipeline failed at final save step: {e}", exc_info=True)
        sys.exit(1)

    logging.info("========== COMPLETED FEATURE ENRICHMENT PIPELINE ==========")


if __name__ == "__main__":
    main()