import pandas as pd
import logging
from pathlib import Path
import sys
import os

# === XÁC ĐỊNH ĐƯỜNG DẪN GỐC ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ===============================

# Đường dẫn tới thư mục dữ liệu thô (nơi bạn đặt 9 file Olist)
RAW_DATA_DIR = PROJECT_ROOT / 'data' / '2_raw'

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_competition_data(data_dir=RAW_DATA_DIR):
    """
    Tải TẤT CẢ dữ liệu từ thư mục data/2_raw/.

    Trả về:
        Một dictionary of DataFrames
    """
    logging.info(f"========== [BƯỚC 1: LOAD DATA] ==========")
    logging.info(f"Bắt đầu tải dữ liệu (Olist PoC) từ: {data_dir}")

    dataframes = {}

    if not data_dir.exists():
        logging.error(f"LỖI: Thư mục dữ liệu thô không tồn tại: {data_dir}")
        sys.exit(1)

    # Danh sách 9 file Olist mà chúng ta mong đợi
    files_to_keys = {
        'olist_orders_dataset.csv': 'orders',
        'olist_order_items_dataset.csv': 'items',
        'olist_products_dataset.csv': 'products',
        'olist_customers_dataset.csv': 'customers',
        'olist_order_reviews_dataset.csv': 'reviews',
        'olist_order_payments_dataset.csv': 'payments',
        'olist_sellers_dataset.csv': 'sellers',
        'olist_geolocation_dataset.csv': 'geolocation',
        'product_category_name_translation.csv': 'translation'
    }

    files_found = 0
    for file, key in files_to_keys.items():
        file_path = data_dir / file

        if not file_path.exists():
            logging.warning(f"⚠️ CẢNH BÁO: Không tìm thấy file {file} trong {data_dir}. Bỏ qua...")
            continue

        try:
            df = pd.read_csv(file_path)
            dataframes[key] = df
            logging.info(f"✓ Đã tải thành công file: {file} (Shape: {df.shape}) -> lưu vào key: '{key}'")
            files_found += 1

        except Exception as e:
            logging.error(f"🚨 LỖI khi tải file {file}: {e}")

    if files_found == 0:
        logging.critical(f"LỖI NGHIÊM TRỌNG: Không tìm thấy bất kỳ file Olist nào trong {data_dir}.")
        sys.exit(1)

    logging.info(f"✓ Tải xong {files_found} file dữ liệu Olist.")
    logging.info(f"Các khóa (keys) đã tạo: {list(dataframes.keys())}")
    logging.info(f"==========================================")
    return dataframes


if __name__ == "__main__":
    logging.info("Chạy _01_load_data.py ở chế độ test (standalone)...")
    data = load_competition_data()
    if data:
        logging.info("Tải dữ liệu test thành công.")