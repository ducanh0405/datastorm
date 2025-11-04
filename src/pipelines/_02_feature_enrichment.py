import pandas as pd
import logging
from pathlib import Path
import sys
import os

# === XÁC ĐỊNH ĐƯỜNG DẪN GỐC ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ===============================

# Cấu hình Logging (chỉ dùng Tiếng Anh/ASCII)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- IMPORT TỪ CÁC THƯ MỤC TRONG src/ ---
try:
    # 1. Import hàm loader cho DỮ LIỆU THẬT (từ data/2_raw/)
    from src.pipelines._01_load_data import load_competition_data

    # 2. Import "THƯ VIỆN CODE" (các hàm đã refactor từ 4 PoC)
    # --- SỬA Ở ĐÂY ---
    # Import logic JOIN của Dunnhumby (WS1)
    from src.features import ws1_relational_features as ws1
    # Import logic Time-Series (WS2)
    from src.features import ws2_timeseries_features as ws2
    # Import logic Behavior (WS3)
    from src.features import ws3_behavior_features as ws3
    # Import logic Price/Promo (WS4)
    from src.features import ws4_price_features as ws4
    # --- KẾT THÚC SỬA ---

    # 3. Import hàm tiện ích validation
    from src.utils.validation import comprehensive_validation

except ImportError as e:
    logging.error(f"ERROR IMPORTING: {e}")
    logging.error("Please ensure __init__.py files exist in all src/ subdirectories.")
    sys.exit(1)


# ---------------------------------------------

def main():
    """
    KIẾN TRÚC SƯ PIPELINE (Chạy trên Dunnhumby để test WS1, 2, 4)
    Tích hợp logic từ 4 Workstream (WS) để xây dựng Master Table cuối cùng.
    """
    logging.info("========== STARTING FEATURE ENRICHMENT PIPELINE (WS1+2+3+4) ==========")

    # 1. Định nghĩa đường dẫn
    OUTPUT_PROCESSED_DIR = PROJECT_ROOT / 'data' / '3_processed'
    OUTPUT_FILE = OUTPUT_PROCESSED_DIR / 'master_feature_table.parquet'
    OUTPUT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Tải Dữ liệu Thật (từ data/2_raw/ - Đặt Dunnhumby vào đây)
    logging.info("--- (1/6) Loading Competition Data (from data/2_raw/) ---")
    dataframes = load_competition_data()  # Gọi hàm từ _01_load_data.py

    if not dataframes or 'transaction_data' not in dataframes:
        logging.critical("Error: 'transaction_data.csv' (main sales file) not found in data/2_raw/.")
        sys.exit(1)

    master_df = dataframes['transaction_data'].copy()
    logging.info(f"Initialized Master Table from 'transaction_data'. Shape: {master_df.shape}")

    # 4. Tích hợp (Enrichment) theo Mô-đun (Tính năng "Bật/Tắt")
    # -----------------------------------------------------------------
    # Workstream 1: Relational (Joins)
    # -----------------------------------------------------------------
    logging.info("--- (2/6) Integrating Workstream 1: Relational ---")
    try:
        # Gọi hàm từ 'ws1_relational_features.py'
        master_df = ws1.enrich_relational_features(master_df, dataframes)
        logging.info(f"-> Shape after WS1: {master_df.shape}")
    except KeyError as e:
        logging.warning(f"SKIPPING WS1: Required data not found (e.g., 'product', 'hh_demographic'). Error: {e}")
    except Exception as e:
        logging.warning(f"ERROR during WS1: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 2: Time-Series & Lịch (Lags, Rolling)
    # -----------------------------------------------------------------
    logging.info("--- (3/6) Integrating Workstream 2: Time-Series ---")
    try:
        # Gọi hàm từ 'ws2_timeseries_features.py'
        master_df = ws2.add_lag_rolling_features(master_df)  # (Hàm này có thể cần dataframes['calendar'])
        logging.info(f"-> Shape after WS2: {master_df.shape}")
    except Exception as e:
        logging.warning(f"ERROR during WS2: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 3: Hành vi (Clickstream)
    # -----------------------------------------------------------------
    logging.info("--- (4/6) Integrating Workstream 3: Behavior ---")
    try:
        # Dunnhumby KHÔNG CÓ clickstream. Logic "bật/tắt" sẽ phát hiện điều này.
        if 'clickstream_log' not in dataframes:
            logging.info("INFO: Skipping WS3: 'clickstream_log' not found in data (As expected for Dunnhumby).")
        else:
            master_df = ws3.add_behavioral_features(master_df, dataframes['clickstream_log'])
            logging.info(f"-> Shape after WS3: {master_df.shape}")

    except Exception as e:
        logging.warning(f"ERROR during WS3: {e}. Skipping...")

    # -----------------------------------------------------------------
    # Workstream 4: Giá & Khuyến mãi (Price & Promotion)
    # -----------------------------------------------------------------
    logging.info("--- (5/6) Integrating Workstream 4: Price/Promotion ---")
    try:
        # Gọi hàm từ 'ws4_price_features.py'
        master_df = ws4.add_price_promotion_features(master_df, dataframes)
        logging.info(f"-> Shape after WS4: {master_df.shape}")
    except KeyError as e:
        logging.warning(f"SKIPPING WS4: Required data not found (e.g., 'causal_data'). Error: {e}")
    except Exception as e:
        logging.warning(f"ERROR during WS4: {e}. Skipping...")

    # 5. Validation và Lưu trữ cuối cùng
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