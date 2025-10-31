# Workstream 1: Olist Data Engineering Pipeline (PoC)
#Hợp nhất 9 tệp dữ liệu Olist thành một Bảng Dữ liệu Tổng thể (Master Table) duy nhất và trích xuất các đặc trưng nghiệp vụ (features) và Phân tích làm sạch dữ liệu với các quy trình tiêu chuẩn


"""
WORKSTREAM 1 (OLIST) - PIPELINE HOÀN CHỈNH (TỪ NOTEBOOKS)

Mục đích:
1.  Tải (Load) các tệp .csv của Olist.
2.  Hợp nhất (Merge) chúng một cách an toàn (xử lý bẫy 'payments').
3.  Tạo (Create) các đặc trưng nghiệp vụ (features) bao gồm geolocation/distance.
4.  Làm sạch (Clean) & Điền Nulls (Impute) tập trung ở cuối.
5.  Kiểm tra (Validate) chi tiết chất lượng dữ liệu cuối cùng.
6.  Xuất (Save) ra một file CSV cuối cùng đã làm sạch.

Cách chạy (từ Terminal):
> pip install pandas numpy haversine pyarrow
> python pipeline_ws1_final.py
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from haversine import haversine # Cần cài đặt: pip install haversine
import logging # Sử dụng logging thay cho print để quản lý tốt hơn
import json
import pprint

# Cấu hình Logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cấu hình Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None # Tắt cảnh báo CopyWarning (chỉ dùng nếu hiểu rõ code)

# --- 1. HÀM TẢI DỮ LIỆU ---

def load_data(data_dir='data/'):
    """Tải tất cả các tệp CSV cần thiết vào một dictionary của DataFrames."""
    logging.info(f"[Bước 1/7] Đang tải dữ liệu từ thư mục: {data_dir}...")
    files_to_keys = {
        'olist_orders_dataset.csv': 'orders', 'olist_order_items_dataset.csv': 'items',
        'olist_products_dataset.csv': 'products', 'olist_customers_dataset.csv': 'customers',
        'olist_order_reviews_dataset.csv': 'reviews', 'olist_order_payments_dataset.csv': 'payments',
        'olist_sellers_dataset.csv': 'sellers', 'olist_geolocation_dataset.csv': 'geolocation',
        'product_category_name_translation.csv': 'translation' # Thêm file translation
    }
    dataframes = {}
    try:
        for file, key in files_to_keys.items():
            file_path = os.path.join(data_dir, file)
            dataframes[key] = pd.read_csv(file_path)
        logging.info(f"-> Tải {len(dataframes)} tệp dữ liệu chính thành công.")
        logging.info(f"-> Các khóa (keys) đã tạo: {list(dataframes.keys())}")
        return dataframes
    except FileNotFoundError as e:
        logging.error(f"🚨 LỖI: Không tìm thấy file {e.filename}. Đảm bảo các tệp CSV nằm trong thư mục '{data_dir}'.")
        sys.exit(1)

def aggregate_payments(df_payments):
    """(QUAN TRỌNG) Xử lý "Bẫy Hợp nhất" 💣. Gộp bảng payments."""
    logging.info("[Bước 2/7] Đang gộp (Aggregate) bảng 'payments'...")
    df_payments_agg = df_payments.groupby('order_id').agg(
        payment_installments_total=('payment_installments', 'sum'),
        payment_value_total=('payment_value', 'sum'),
        payment_type_primary=('payment_type', 'first'),
        payment_sequential_count=('payment_sequential', 'max') # Thêm từ Notebook 1
    ).reset_index()
    logging.info(f"-> Đã gộp 'payments' từ {len(df_payments)} hàng xuống {len(df_payments_agg)} hàng.")
    return df_payments_agg

def aggregate_geolocation(df_geo):
    """Aggregate geolocation để tối ưu merge."""
    logging.info("[Bước 3/7] Đang gộp (Aggregate) bảng 'geolocation'...")
    # Lấy tọa độ trung bình cho mỗi zip code
    df_geo_agg = df_geo.groupby('geolocation_zip_code_prefix').agg(
        geo_lat=('geolocation_lat', 'mean'),
        geo_lng=('geolocation_lng', 'mean')
    ).reset_index()
    logging.info(f"-> Đã gộp 'geolocation' từ {len(df_geo)} hàng xuống {len(df_geo_agg)} hàng (zip codes duy nhất).")
    return df_geo_agg

# --- 2. HÀM HỢP NHẤT ---

def merge_tables(dataframes, df_payments_agg, df_geo_agg):
    """Thực thi pipeline hợp nhất (merge) các bảng."""
    logging.info("[Bước 4/7] Đang hợp nhất (Merge) các bảng...")
    df_master = dataframes['orders'].copy()

    # Merge bảng chính
    df_master = pd.merge(df_master, dataframes['customers'], on='customer_id', how='left')
    df_reviews_dedup = dataframes['reviews'].sort_values('review_creation_date', ascending=False).drop_duplicates('order_id', keep='first')
    df_master = pd.merge(df_master, df_reviews_dedup, on='order_id', how='left')
    df_master = pd.merge(df_master, df_payments_agg, on='order_id', how='left')
    df_master = pd.merge(df_master, dataframes['items'], on='order_id', how='left')
    df_master = pd.merge(df_master, dataframes['products'], on='product_id', how='left')
    df_master = pd.merge(df_master, dataframes['sellers'], on='seller_id', how='left')
    df_master = pd.merge(df_master, dataframes['translation'], on='product_category_name', how='left')

    # Merge Geolocation (2 lần, đã aggregate)
    # Lần 1: Customer
    df_master = pd.merge(df_master, df_geo_agg, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left')
    df_master.rename(columns={'geo_lat': 'customer_lat', 'geo_lng': 'customer_lng'}, inplace=True)
    df_master.drop(columns=['geolocation_zip_code_prefix'], inplace=True, errors='ignore')

    # Lần 2: Seller
    df_master = pd.merge(df_master, df_geo_agg, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix', how='left', suffixes=('', '_seller_geo'))
    df_master.rename(columns={'geo_lat': 'seller_lat', 'geo_lng': 'seller_lng'}, inplace=True)
    df_master.drop(columns=['geolocation_zip_code_prefix_seller_geo', 'geolocation_zip_code_prefix'], inplace=True, errors='ignore') # Xóa cả 2 cột zip thừa

    logging.info(f"-> Hợp nhất (Merge) thành công. Kích thước bảng tổng thể: {df_master.shape}")
    return df_master

# --- 3. HÀM TẠO ĐẶC TRƯNG (CHƯA CLEAN) ---

def create_features(df_merged):
    """Tạo tất cả các đặc trưng nghiệp vụ."""
    logging.info("[Bước 5/7] Đang tạo đặc trưng (Feature Engineering)...")
    df_featured = df_merged.copy()

    # 1. Chuyển đổi Thời gian
    time_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_carrier_date', 'order_delivered_customer_date',
                 'order_estimated_delivery_date', 'shipping_limit_date',
                 'review_creation_date', 'review_answer_timestamp']
    for col in time_cols:
        if col in df_featured.columns: # Kiểm tra trước khi chuyển đổi
            df_featured[col] = pd.to_datetime(df_featured[col], errors='coerce')

    # 2. Đặc trưng Vận hành
    if 'order_delivered_customer_date' in df_featured.columns and 'order_purchase_timestamp' in df_featured.columns:
        df_featured['delivery_time_days'] = (df_featured['order_delivered_customer_date'] - df_featured['order_purchase_timestamp']).dt.total_seconds() / (24 * 60 * 60)
    if 'order_estimated_delivery_date' in df_featured.columns and 'order_delivered_customer_date' in df_featured.columns:
        df_featured['delivery_vs_estimated_days'] = (df_featured['order_estimated_delivery_date'] - df_featured['order_delivered_customer_date']).dt.total_seconds() / (24 * 60 * 60)
    if 'order_delivered_carrier_date' in df_featured.columns and 'order_purchase_timestamp' in df_featured.columns:
        df_featured['order_processing_time_days'] = (df_featured['order_delivered_carrier_date'] - df_featured['order_purchase_timestamp']).dt.total_seconds() / (24 * 60 * 60)

    # 3. Đặc trưng Cyclical Time (Từ Notebook 1)
    if 'order_purchase_timestamp' in df_featured.columns:
        df_featured['purchase_year'] = df_featured['order_purchase_timestamp'].dt.year
        df_featured['purchase_month'] = df_featured['order_purchase_timestamp'].dt.month
        df_featured['purchase_day_of_week'] = df_featured['order_purchase_timestamp'].dt.dayofweek # 0=Mon, 6=Sun
        df_featured['purchase_hour'] = df_featured['order_purchase_timestamp'].dt.hour
        df_featured['is_weekend'] = df_featured['purchase_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # 4. Đặc trưng Địa lý (Khoảng cách)
    if 'customer_lat' in df_featured.columns and 'seller_lat' in df_featured.columns:
        locations_available = df_featured[['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']].notnull().all(axis=1)
        distances = df_featured[locations_available].apply(
            lambda row: haversine((row['customer_lat'], row['customer_lng']), (row['seller_lat'], row['seller_lng'])),
            axis=1
        )
        df_featured['dist_cust_seller_km'] = np.nan
        df_featured.loc[locations_available, 'dist_cust_seller_km'] = distances
        logging.info(" -> Đã tính 'dist_cust_seller_km'.")

    # 5. Đặc trưng Tài chính & Sản phẩm
    if 'price' in df_featured.columns and 'freight_value' in df_featured.columns:
        df_featured['freight_ratio'] = df_featured['freight_value'] / (df_featured['price'] + 1e-6)
        df_featured['freight_ratio'] = df_featured['freight_ratio'].replace([np.inf, -np.inf], 0) # Xử lý inf
    if 'product_length_cm' in df_featured.columns: # Kiểm tra tồn tại
        df_featured['product_volume_cm3'] = (
            df_featured['product_length_cm'] * df_featured['product_height_cm'] * df_featured['product_width_cm']
        )
        logging.info(" -> Đã tính 'freight_ratio' và 'product_volume_cm3'.")

    # 6. Đặc trưng Thanh toán (Cờ)
    if 'payment_type_primary' in df_featured.columns:
        df_featured['is_payment_credit_card'] = (df_featured['payment_type_primary'] == 'credit_card').astype(float)
        df_featured['is_payment_boleto'] = (df_featured['payment_type_primary'] == 'boleto').astype(float)
        df_featured['is_payment_voucher'] = (df_featured['payment_type_primary'] == 'voucher').astype(float)
    if 'payment_installments_total' in df_featured.columns:
        df_featured['is_payment_installments'] = (df_featured['payment_installments_total'] > 1).astype(float)

    # 7. (TÙY CHỌN) Đặc trưng Review Time-Safe (Tối ưu 3) - Có thể thêm hàm fix_review_leakage ở đây

    logging.info(f"-> Tạo đặc trưng hoàn tất. Shape hiện tại: {df_featured.shape}")
    return df_featured

# --- 4. HÀM LÀM SẠCH & ĐIỀN NULLS CUỐI CÙNG ---

def clean_and_impute(df_featured):
    """Làm sạch và điền TẤT CẢ nulls còn lại MỘT LẦN."""
    logging.info("[Bước 6/7] Đang thực hiện làm sạch cuối cùng và điền Nulls...")
    df_clean = df_featured.copy()

    # === 1. LÀM SẠCH (Cleaning) ===
    # 1.1 Cardinality (Category Name)
    cat_cols_to_clean = ['product_category_name', 'product_category_name_english']
    for col in cat_cols_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].str.lower().str.strip()

    # 1.2 Outliers & Logic (Dựa trên Notebook 2)
    # Cap Freight Ratio
    if 'freight_ratio' in df_clean.columns:
        df_clean['freight_ratio'] = df_clean['freight_ratio'].clip(upper=10)
    # Xử lý giá trị âm (trừ sentinel -999)
    cols_non_negative = ['price', 'freight_value', 'payment_value_total']
    for col in cols_non_negative:
        if col in df_clean.columns:
            df_clean.loc[df_clean[col] < 0, col] = 0
    delivery_negative_mask = (df_clean['delivery_time_days'] < 0) & (df_clean['delivery_time_days'] != -999)
    if delivery_negative_mask.any():
        df_clean.loc[delivery_negative_mask, 'delivery_time_days'] = 0

    # === 2. ĐIỀN NULLS (Imputation) ===
    # Chiến lược được định nghĩa ở đây

    # 2.1 Cột Review Score (0 = Chưa review)
    if 'review_score' in df_clean.columns:
        df_clean['review_score'] = df_clean['review_score'].fillna(0)

    # 2.2 Cột Vận hành (Chưa giao = -999)
    delivery_cols_to_flag = ['delivery_time_days', 'delivery_vs_estimated_days', 'order_processing_time_days']
    for col in delivery_cols_to_flag:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(-999)

    # 2.3 Cột Phân loại (Categorical = 'unknown')
    categorical_cols_to_unknown = [
        'product_category_name', 'product_category_name_english',
        'payment_type_primary',
        'customer_city', 'customer_state',
        'seller_city', 'seller_state'
    ]
    for col in categorical_cols_to_unknown:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('unknown')

    # 2.4 Cột Số học (Numeric = 0 hoặc median/mean nếu hợp lý)
    # Điền 0 cho các giá trị này
    numeric_cols_to_zero = [
        'payment_installments_total', 'payment_value_total',
        'price', 'freight_value', 'freight_ratio',
        'product_name_lenght', 'product_description_lenght', 'product_photos_qty',
        'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',
        'product_volume_cm3',
        'is_payment_credit_card', 'is_payment_boleto', 'is_payment_voucher',
        'is_payment_installments', 'payment_sequential_count' # Thêm payment_sequential_count
    ]
    for col in numeric_cols_to_zero:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    # Điền mean cho Khoảng cách (nếu thiếu lat/lng)
    if 'dist_cust_seller_km' in df_clean.columns:
         mean_dist = df_clean['dist_cust_seller_km'].mean() # Tính mean trên cột đã tính (loại bỏ NaN)
         df_clean['dist_cust_seller_km'] = df_clean['dist_cust_seller_km'].fillna(mean_dist if not pd.isna(mean_dist) else 0)

    # Điền 0 cho các cột Lat/Lng còn sót (sau merge geo)
    geo_coords = ['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng']
    for col in geo_coords:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    # Chuyển các cột cờ về int sau khi fillna
    flag_cols = ['is_payment_credit_card', 'is_payment_boleto', 'is_payment_voucher', 'is_payment_installments', 'is_weekend']
    for col in flag_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)

    # 2.5 Các cột khác (Review text/dates, approved_at, etc.) - Bỏ qua hoặc điền nếu cần
    # Ví dụ:
    # df_clean['review_comment_message'] = df_clean['review_comment_message'].fillna('none')

    # 3. Làm sạch cuối cùng (loại bỏ hàng thiếu khóa chính)
    df_clean.dropna(subset=['order_id', 'order_item_id'], inplace=True)

    logging.info("-> Làm sạch cuối cùng và điền Nulls hoàn tất.")
    return df_clean


# --- 5. HÀM KIỂM TRA (VALIDATION FUNCTION) ---
# Sử dụng hàm comprehensive_validation chi tiết từ Notebook 2
def comprehensive_validation(df, verbose=True):
    """Validation tổng hợp toàn diện (lấy từ Notebook 2)."""
    logging.info("[Bước 7/7] Đang kiểm tra (Validate) pipeline cuối cùng...")  # Cập nhật số bước
    validation_results = {}
    issues_found = False  # Cờ để theo dõi lỗi

    # 3.1: Thông tin cơ bản
    if verbose: logging.info("\n--- 3.1 Thông tin cơ bản DataFrame ---")
    validation_results['shape'] = df.shape
    validation_results['memory_mb'] = round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2)
    if verbose:
        logging.info(f"✓ Shape: {validation_results['shape']}")
        logging.info(f"✓ Memory: {validation_results['memory_mb']} MB")

    # 3.2: Kiểm tra Missing Values
    if verbose: logging.info("\n--- 3.2 Kiểm tra Missing Values ---")

    # Định nghĩa các cột được phép là Null/NaT (vì chúng mang ý nghĩa nghiệp vụ)
    cols_allowed_to_be_null = [
        'review_comment_title',
        'review_comment_message',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'shipping_limit_date',
        'review_creation_date',
        'review_answer_timestamp',
        'review_id'
    ]
    # Tạo danh sách các cột CẦN PHẢI SẠCH (không được Null)
    all_cols = df.columns.tolist()
    cols_to_validate = [col for col in all_cols if col not in cols_allowed_to_be_null]

    # Tạo một DataFrame tạm thời CHỈ chứa các cột cần kiểm tra
    df_validate = df[cols_to_validate]
    # --- [SỬA ĐỔI KẾT THÚC] ---

    # Sửa các dòng tính toán để dùng df_validate, không dùng df
    total_cells = df_validate.shape[0] * df_validate.shape[1]
    total_missing = df_validate.isna().sum().sum()  # SỬA Ở ĐÂY
    missing_pct = round(total_missing * 100.0 / total_cells, 2) if total_cells > 0 else 0
    validation_results['total_missing_values'] = total_missing
    validation_results['missing_pct'] = missing_pct
    validation_results['cols_with_missing'] = df_validate.isna().any().sum()  # SỬA Ở ĐÂY

    if total_missing > 0:
        # Cập nhật thông báo lỗi
        logging.warning(
            f"-> 🚨 KIỂM TRA MISSING THẤT BẠI: Vẫn còn {total_missing:,} giá trị Null ({missing_pct}%) TRONG CÁC CỘT QUAN TRỌNG.")
        issues_found = True
        if verbose:
            top_missing = df_validate.isna().sum().sort_values(ascending=False).head(5)  # SỬA Ở ĐÂY
            top_missing = top_missing[top_missing > 0]
            logging.warning("  Top cột (quan trọng) thiếu nhiều nhất:")
            for col, count in top_missing.items():
                pct = round(count * 100.0 / df_validate.shape[0], 2)  # SỬA Ở ĐÂY
                logging.warning(f"    - {col}: {count:,} ({pct}%)")
    elif verbose:
        logging.info("✓ Kiểm tra Missing: Đạt.")

    # 3.3: Kiểm tra Duplicates (Toàn bộ hàng)
    if verbose: logging.info("\n--- 3.3 Kiểm tra Duplicates ---")
    dup_rows = df.duplicated().sum()
    validation_results['duplicate_rows'] = dup_rows
    if dup_rows > 0:
        dup_pct = round(dup_rows * 100.0 / df.shape[0], 2)
        logging.warning(f"-> 🚨 KIỂM TRA DUPLICATES THẤT BẠI: Tìm thấy {dup_rows:,} hàng trùng lặp ({dup_pct}%).")
        issues_found = True
    elif verbose:
        logging.info("✓ Kiểm tra Duplicates: Đạt.")

    # 3.4: Kiểm tra Key Integrity (Granularity)
    if verbose: logging.info("\n--- 3.4 Kiểm tra Key Integrity (Granularity) ---")
    key_cols = ['order_id', 'order_item_id']
    if all(col in df.columns for col in key_cols):
        df_check = df.copy()
        df_check[key_cols[0]] = df_check[key_cols[0]].fillna('MISSING_ORDER')
        df_check[key_cols[1]] = df_check[key_cols[1]].fillna('MISSING_ITEM')
        dup_keys = df_check.duplicated(subset=key_cols).sum()
        validation_results['duplicate_keys'] = dup_keys
        if dup_keys > 0:
            logging.warning(f"-> 🚨 KIỂM TRA KEY THẤT BẠI: Tìm thấy {dup_keys:,} hàng trùng lặp theo khóa {key_cols}.")
            issues_found = True
        elif verbose:
            logging.info(f"✓ Kiểm tra Key Integrity {key_cols}: Đạt.")
    else:
        logging.error(f"-> 🚨 KIỂM TRA KEY THẤT BẠI: Thiếu cột khóa {key_cols}.")
        issues_found = True
        validation_results['duplicate_keys'] = -1  # Indicate check failed

    # 3.5: Kiểm tra Business Logic Constraints
    if verbose: logging.info("\n--- 3.5 Kiểm tra Business Logic Constraints ---")
    violations = {}
    if 'review_score' in df.columns:
        invalid_reviews = df[(df['review_score'] < 0) | (df['review_score'] > 5)]
        violations['invalid_review_score'] = len(invalid_reviews)
    if 'price' in df.columns: violations['negative_price'] = (df['price'] < 0).sum()  # Chỉ cần < 0 vì 0 có thể hợp lệ
    if 'freight_value' in df.columns: violations['negative_freight'] = (df['freight_value'] < 0).sum()
    if 'delivery_time_days' in df.columns:
        invalid_delivery = (df['delivery_time_days'] < -999).sum()  # Chỉ kiểm tra < -999
        violations['invalid_delivery_time'] = invalid_delivery
    total_violations = sum(violations.values())
    validation_results['business_logic_violations'] = violations
    if total_violations > 0:
        logging.warning(f"-> 🚨 KIỂM TRA LOGIC THẤT BẠI: Tìm thấy {total_violations} vi phạm logic nghiệp vụ.")
        issues_found = True
        if verbose: print(violations)
    elif verbose:
        logging.info("✓ Kiểm tra Business Logic: Đạt.")

    # 3.6: Validation Score
    score = 100.0
    score -= min(missing_pct * 5, 25)  # missing_pct bây giờ đã được tính đúng
    score -= min((dup_rows * 100.0 / df.shape[0]) * 5, 15) if df.shape[0] > 0 else 0
    violation_pct = total_violations * 100.0 / df.shape[0] if df.shape[0] > 0 else 0
    score -= min(violation_pct * 10, 20)
    validation_results['quality_score'] = round(max(score, 0), 2)

    if verbose:
        logging.info(f"\n--- 3.6 Overall Data Quality Score ---")
        logging.info(f"🎯 Quality Score: {validation_results['quality_score']}/100")
        if validation_results['quality_score'] >= 90:
            logging.info("✅ EXCELLENT")
        elif validation_results['quality_score'] >= 75:
            logging.info("✓ GOOD")
        else:
            logging.warning("⚠ FAIR/POOR")

    validation_results['passed'] = not issues_found
    return validation_results

# --- 6. HÀM CHÍNH (MAIN FUNCTION) ---

def main():
    """Điều phối toàn bộ pipeline."""
    start_time = time.time()
    DATA_DIR = 'data/'
    OUTPUT_FILE_CSV = 'olist_master_table_final.csv'  # Đổi tên file output cuối
    OUTPUT_FILE_PARQUET = 'olist_master_table_final.parquet'
    # --- Chạy Pipeline ---
    dataframes = load_data(DATA_DIR)
    df_payments_agg = aggregate_payments(dataframes['payments'])
    df_geo_agg = aggregate_geolocation(dataframes['geolocation']) # Thêm bước aggregate geo
    df_merged = merge_tables(dataframes, df_payments_agg, df_geo_agg) # Truyền df_geo_agg vào
    df_featured = create_features(df_merged)
    # df_featured = fix_review_leakage(df_featured) # Tạm thời comment Tối ưu 3 nếu chưa cần
    df_final = clean_and_impute(df_featured) # Bước làm sạch và impute cuối cùng

    # --- Kiểm tra & Lưu ---
    validation_report = comprehensive_validation(df_final, verbose=True)  # Chạy validation chi tiết

    # Chúng ta sẽ không chặn việc lưu file nếu validation thất bại,
    # nhưng chúng ta sẽ ghi lại cảnh báo và vẫn lưu file lỗi.

    if not validation_report['passed']:
        logging.warning("\n⚠️ CẢNH BÁO: Pipeline không vượt qua kiểm tra, nhưng VẪN TIẾN HÀNH LƯU FILE.")
        # Vẫn giữ lại phần lưu file JSON báo lỗi từ khối 'else' cũ
        try:
            validation_file_error = 'validation_report_FAILED.json'
            clean_report = {}
            for k, v in validation_report.items():
                try:
                    json.dumps({k: v})
                    clean_report[k] = v
                except TypeError:
                    clean_report[k] = str(v)

            with open(validation_file_error, 'w', encoding='utf-8') as f:
                json.dump(clean_report, f, indent=2, ensure_ascii=False)
            logging.info(f"-> Đã lưu chi tiết lỗi validation vào: {validation_file_error}")
        except Exception as json_e:
            logging.error(f"-> Không thể lưu validation report lỗi: {json_e}")

    # KHỐI LƯU FILE NÀY GIỜ SẼ LUÔN CHẠY (được đưa ra khỏi 'if')
    logging.info(f"\n[Bước 8/8] Đang lưu trữ file {OUTPUT_FILE_CSV}...")

    try:
        # Chọn các cột cuối cùng để lưu
        final_columns = [
            'order_id', 'order_item_id', 'product_id', 'customer_id', 'seller_id',
            'order_purchase_timestamp',
            'delivery_time_days', 'delivery_vs_estimated_days', 'order_processing_time_days',
            'price', 'freight_value', 'freight_ratio',
            'is_payment_credit_card', 'is_payment_boleto', 'is_payment_voucher', 'is_payment_installments',
            'payment_value_total', 'payment_installments_total', 'payment_sequential_count',
            'review_score',
            'dist_cust_seller_km',
            'product_category_name', 'product_category_name_english',
            'customer_state', 'seller_state',
            'customer_lat', 'customer_lng', 'seller_lat', 'seller_lng',
            'product_weight_g', 'product_volume_cm3',
            'purchase_year', 'purchase_month', 'purchase_day_of_week', 'purchase_hour', 'is_weekend'
        ]
        final_columns_exist = [col for col in final_columns if col in df_final.columns]
        df_final_output = df_final[final_columns_exist]

        df_final_output.to_csv(OUTPUT_FILE_CSV, index=False)
        logging.info(f"\n--- 🥳 HOÀN THÀNH WORKSTREAM 1 (FINAL VERSION) ---")
        logging.info(f"Output đã được lưu tại: {OUTPUT_FILE_CSV}")
        logging.info(f"Kích thước cuối cùng: {df_final_output.shape}")

    except Exception as e:
        logging.error(f"\n🚨 LỖI khi lưu file CSV: {e}")

    end_time = time.time()
    logging.info(f"\nTổng thời gian chạy pipeline: {end_time - start_time:.2f} giây.")

# --- ĐIỂM BẮT ĐẦU CHẠY SCRIPT ---
if __name__ == "__main__":
    main()
