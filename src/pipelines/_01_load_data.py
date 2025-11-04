import pandas as pd
import logging
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
RAW_DATA_DIR = PROJECT_ROOT / 'data' / '2_raw'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_competition_data(data_dir=RAW_DATA_DIR):
    """
    Tai TAT CA du lieu tho (Dunnhumby, M5, v.v.) tu data/2_raw/.
    Tu dong doc cac file .csv hoac .parquet.
    """
    logging.info(f"========== [STEP 1: LOAD DATA] ==========")
    logging.info(f"Starting to load raw data from: {data_dir}")
    dataframes = {}
    if not data_dir.exists():
        logging.error(f"ERROR: Raw data directory not found: {data_dir}")
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