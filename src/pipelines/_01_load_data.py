"""
Data Loading Module
===================
Loads raw competition data from various sources (Dunnhumby, M5, etc.).
"""
import pandas as pd
import logging
from pathlib import Path
import sys
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Auto-detect data directory (configurable for different use cases)
# Priority: 2_raw (full data) > poc_data (sample data)
RAW_DATA_DIR: Optional[Path] = None
for possible_dir in [
    PROJECT_ROOT / 'data' / '2_raw',     # Full data - PRIORITY for production
    PROJECT_ROOT / 'data' / 'poc_data',  # POC data (1% sample) - for testing
]:
    if possible_dir.exists() and list(possible_dir.glob('*.csv')):
        RAW_DATA_DIR = possible_dir
        break

if RAW_DATA_DIR is None:
    RAW_DATA_DIR = PROJECT_ROOT / 'data' / '2_raw'  # Default fallback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        data_dir = RAW_DATA_DIR

    logging.info(f"========== [STEP 1: LOAD DATA] ==========")
    logging.info(f"Data source: {data_dir}")
    if 'poc_data' in str(data_dir):
        logging.info("⚠️  Using POC data (1% sample) - for testing only")
        logging.info("💡 For full production data, ensure files exist in data/2_raw/")
    else:
        logging.info("✅ Using full data from data/2_raw/")
    dataframes = {}
    
    if not data_dir.exists():
        logging.error(f"ERROR: Raw data directory not found: {data_dir}")
        logging.error(f"Please run: python scripts/create_sample_data.py")
        sys.exit(1)

    files = [f for f in data_dir.iterdir() if f.is_file() and (f.suffix in ['.csv', '.parquet'])]
    if not files:
        logging.warning(f"WARNING: No .csv or .parquet files found in {data_dir}")
        return {}

    for file_path in files:
        try:
            key = file_path.stem
            if file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            dataframes[key] = df
            logging.info(f"OK. Successfully loaded file: {file_path.name} (Shape: {df.shape}) -> saved to key: '{key}'")
        except Exception as e:
            logging.error(f"ERROR loading file {file_path.name}: {e}")

    logging.info(f"OK. Loaded {len(dataframes)} data files.")
    logging.info(f"Keys created: {list(dataframes.keys())}")
    logging.info(f"==========================================")
    return dataframes


if __name__ == "__main__":
    logging.info("Running _01_load_data.py in standalone test mode...")
    data = load_competition_data()
    if data:
        logging.info("Test data load successful.")