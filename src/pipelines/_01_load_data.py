"""
Data Loading Module
===================
Loads raw competition data from various sources (Dunnhumby, M5, etc.).
"""
import pandas as pd
import logging
import os
from pathlib import Path
import sys
from typing import Dict, Optional

# Import centralized configuration
try:
    from src.config import DATA_DIRS, get_data_directory, setup_logging
    setup_logging()  # Setup centralized logging
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback if config not available
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(PROJECT_ROOT))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def load_competition_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all raw data (Dunnhumby, M5, etc.) from data directory.
    Auto-detects data location with priority: 2_raw (full) > poc_data (sample).
    Reads .csv or .parquet files.

    Args:
        data_dir: Optional path to data directory. If None, auto-detects.

    Returns:
        Dictionary mapping filename (without extension) to DataFrame

    Raises:
        SystemExit: If data directory not found
    """
    if data_dir is None:
        try:
            # Check for environment variable to force specific data source
            data_source = os.environ.get('DATA_SOURCE', '').lower()
            if data_source == 'poc' or data_source == 'sample':
                data_dir = DATA_DIRS['poc_data']
                logger.info("FORCED: Using POC data due to DATA_SOURCE=poc environment variable")
            elif data_source == 'full' or data_source == 'raw':
                data_dir = DATA_DIRS['raw_data']
                logger.info("FORCED: Using full data due to DATA_SOURCE=full environment variable")
            else:
                # Default: prefer POC data for development
                data_dir = get_data_directory()
        except NameError:
            # Fallback if config not available
            data_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'poc_data'

    logger.info("=" * 50)
    logger.info("[STEP 1: LOAD DATA]")
    logger.info("=" * 50)
    logger.info(f"Data source: {data_dir}")

    if 'poc_data' in str(data_dir):
        logger.info("WARNING: Using POC data (1% sample) - for testing only")
        logger.info("TIP: For full production data, ensure files exist in data/2_raw/")
    else:
        logger.info("SUCCESS: Using full data from data/2_raw/")

    dataframes = {}

    if not data_dir.exists():
        logger.error(f"ERROR: Raw data directory not found: {data_dir}")
        logger.error("Please run: python scripts/create_sample_data.py")
        sys.exit(1)

    files = [f for f in data_dir.iterdir() if f.is_file() and f.suffix in ['.csv', '.parquet']]
    if not files:
        logger.warning(f"WARNING: No .csv or .parquet files found in {data_dir}")
        return {}

    for file_path in files:
        try:
            key = file_path.stem
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            dataframes[key] = df
            logger.info(f"SUCCESS: Loaded {file_path.name} (Shape: {df.shape}) -> key: '{key}'")
        except Exception as e:
            logger.error(f"ERROR loading file {file_path.name}: {e}")

    logger.info(f"SUCCESS: Loaded {len(dataframes)} data files")
    logger.info(f"Keys: {list(dataframes.keys())}")
    logger.info("=" * 50)
    return dataframes


if __name__ == "__main__":
    logging.info("Running _01_load_data.py in standalone test mode...")
    data = load_competition_data()
    if data:
        logging.info("Test data load successful.")