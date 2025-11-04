import pandas as pd
import numpy as np
import logging



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _process_clickstream_logs(df_events):
    """
    Hàm nội bộ: Thực hiện logic từ 'EDA_Data_Preprocess.ipynb'.
    Làm sạch, xử lý thời gian, và chuẩn bị log.
    """
    logging.info("[WS3] Bắt đầu xử lý log hành vi (clickstream)...")
    
    # Giả sử df_events có các cột: 'timestamp', 'visitorid', 'event', 'itemid'
    if 'timestamp' in df_events.columns:
        df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='ms') # Giả sử timestamp là ms

    # Xử lý NaNs (nếu có)
    df_events = df_events.dropna(subset=['visitorid', 'itemid'])
    
    logging.info(f"[WS3] Xử lý log hoàn tất. Tổng số sự kiện: {len(df_events)}")
    return df_events

def _create_user_features(df_logs):
    """
    Hàm nội bộ: Thực hiện logic từ 'Feature_Engineering.ipynb'.
    Tạo bảng đặc trưng (feature table) ở cấp độ người dùng (user-level).
    """
    logging.info("[WS3] Bắt đầu tạo đặc trưng hành vi (Feature Engineering)...")
    
    # 1. Tạo các đặc trưng cơ bản (ví dụ từ PoC của bạn)
    # Đây là logic tính toán "phễu chuyển đổi"
    user_features = df_logs.pivot_table(
        index='visitorid', 
        columns='event', 
        aggfunc='size', 
        fill_value=0
    )
    
    # Đổi tên cột nếu cần (ví dụ: 'addtocart' -> 'total_addtocart')
    user_features = user_features.rename(columns={
        'view': 'total_views',
        'addtocart': 'total_addtocart',
        'transaction': 'total_transactions'
    })
    
    # 2. Tạo đặc trưng về tỷ lệ (Conversion Rates)
    # Tỷ lệ xem -> thêm vào giỏ
    user_features['rate_view_to_cart'] = user_features['total_addtocart'] / (user_features['total_views'] + 1e-6)
    
    # Tỷ lệ thêm vào giỏ -> mua
    user_features['rate_cart_to_buy'] = user_features['total_transactions'] / (user_features['total_addtocart'] + 1e-6)
    
    # Tỷ lệ xem -> mua (tỷ lệ chuyển đổi tổng thể)
    user_features['rate_view_to_buy'] = user_features['total_transactions'] / (user_features['total_views'] + 1e-6)

    # 3. Tạo các đặc trưng về thời gian (session-based)
    # (Đây là ví dụ, bạn sẽ thay thế bằng logic phức tạp hơn từ notebook của bạn)
    if 'timestamp' in df_logs.columns:
        time_stats = df_logs.groupby('visitorid')['timestamp'].agg(['min', 'max'])
        time_stats['session_duration_days'] = (time_stats['max'] - time_stats['min']).dt.total_seconds() / (60 * 60 * 24)
        user_features = user_features.join(time_stats['session_duration_days'], how='left')

    # 4. Tạo đặc trưng về thời gian kể từ lần tương tác cuối cùng
    # (Nếu có cột timestamp trong df_logs)
    if 'timestamp' in df_logs.columns:
        latest_timestamp = df_logs['timestamp'].max()
        last_interaction = df_logs.groupby('visitorid')['timestamp'].max()
        user_features['days_since_last_action'] = (latest_timestamp - last_interaction).dt.total_seconds() / (60 * 60 * 24)

    logging.info(f"[WS3] Tạo đặc trưng hoàn tất. Số user: {len(user_features)}")
    return user_features


# ===================================================================
# HÀM CHÍNH (SẼ ĐƯỢC GỌI BỞI _02_feature_enrichment.py)
# ===================================================================

def add_behavioral_features(master_df, dataframes_dict):
    """
    Hàm "chủ" (master function) cho Workstream 3.
    Nó nhận Master Table và dict của dữ liệu thô (đặc biệt là 'clickstream_log').

    Nó tạo ra các đặc trưng về hành vi người dùng và merge vào Master Table.
    """

    # 1. Kiểm tra xem dữ liệu behavior có tồn tại không
    required_keys = ['clickstream_log']  # Giả sử tên file là 'clickstream_log'
    if not all(key in dataframes_dict for key in required_keys):
        logging.warning("⚠️ Bỏ qua Workstream 3: Thiếu dữ liệu clickstream.")
        return master_df

    # 2. Xử lý dữ liệu clickstream
    df_events = dataframes_dict['clickstream_log'].copy()
    df_processed = _process_clickstream_logs(df_events)

    # 3. Tạo đặc trưng người dùng
    user_features = _create_user_features(df_processed)

    # 4. Merge vào Master Table
    # Giả sử master_df có cột 'visitorid' hoặc 'customer_id' để merge
    merge_keys = ['visitorid']  # Thay đổi nếu tên cột khác
    if all(key in master_df.columns for key in merge_keys):
        original_rows = master_df.shape[0]
        master_df = pd.merge(master_df, user_features, on=merge_keys, how='left')

        # Điền NaN cho các user không có dữ liệu behavior
        numeric_cols = user_features.select_dtypes(include=[np.number]).columns
        master_df[numeric_cols] = master_df[numeric_cols].fillna(0)

        logging.info("✓ Tích hợp Workstream 3 (Hành vi) thành công.")
    else:
        logging.warning("⚠️ Bỏ qua merge WS3: Không tìm thấy cột merge keys trong Master Table.")

    return master_df