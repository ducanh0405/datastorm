import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _clean_causal_data(df_causal):
    """
    Hàm nội bộ: Làm sạch file causal_data.csv.
    (Logic từ clean_w4.py)
    """
    logging.info("[WS4] Đang làm sạch dữ liệu 'causal' (khuyến mãi)...")
    
    # Chuyển đổi kiểu dữ liệu (ví dụ)
    # (Bạn sẽ thay thế bằng logic clean từ 'clean_w4.py')
    df_causal['DISPLAY'] = df_causal['DISPLAY'].astype(str)
    df_causal['MAILER'] = df_causal['MAILER'].astype(str)
    
    # Đổi tên cột để tránh trùng lặp khi merge (ví dụ)
    df_causal = df_causal.rename(columns={
        'DISPLAY': 'promo_display_type',
        'MAILER': 'promo_mailer_type'
    })
    
    # Tạo các cờ (flags) nhị phân
    df_causal['is_on_display'] = (df_causal['promo_display_type'] != '0').astype(int)
    df_causal['is_on_mailer'] = (df_causal['promo_mailer_type'] != '0').astype(int)
    
    # Chỉ giữ các cột cần thiết để merge
    # Khóa (key) của Dunnhumby causal là 'STORE_ID', 'PRODUCT_ID', 'WEEK_NO'
    causal_features = ['STORE_ID', 'PRODUCT_ID', 'WEEK_NO', 'is_on_display', 'is_on_mailer']
    
    # Loại bỏ trùng lặp (nếu có)
    df_causal_clean = df_causal[causal_features].drop_duplicates()
    
    return df_causal_clean

def _clean_transaction_data(master_df):
    """
    Hàm nội bộ: Làm sạch các cột giá/khuyến mãi trên bảng transaction.
    (Logic từ build_w4_features.py)
    """
    logging.info("[WS4] Đang tạo đặc trưng giá/khuyến mãi từ bảng transactions...")
    
    # Các cột này đến từ transaction_data.csv
    price_cols = ['SALES_VALUE', 'RETAIL_DISC', 'COUPON_DISC']
    
    # Điền NaNs (nếu có)
    master_df[price_cols] = master_df[price_cols].fillna(0)
    
    # 1. Tính toán giá gốc (Base Price)
    # Giá gốc = (Doanh thu - (tổng giảm giá))
    # (Lưu ý: Giảm giá của Dunnhumby là SỐ ÂM, nên ta phải cộng)
    master_df['base_price'] = master_df['SALES_VALUE'] - (master_df['RETAIL_DISC'] + master_df['COUPON_DISC'])
    
    # 2. Tạo đặc trưng Tỷ lệ % Giảm giá
    # (Tránh chia cho 0)
    master_df['total_discount'] = (master_df['RETAIL_DISC'] + master_df['COUPON_DISC']).abs()
    master_df['discount_pct'] = master_df['total_discount'] / (master_df['base_price'] + 1e-6)
    
    # 3. Tạo các cờ (flags) nhị phân
    master_df['is_on_retail_promo'] = (master_df['RETAIL_DISC'] < 0).astype(int)
    master_df['is_on_coupon_promo'] = (master_df['COUPON_DISC'] < 0).astype(int)
    
    return master_df


# ===================================================================
# HÀM CHÍNH (SẼ ĐƯỢC GỌI BỞI _02_feature_enrichment.py)
# ===================================================================

def add_price_promotion_features(master_df, dataframes_dict):
    """
    Hàm "chủ" (master function) cho Workstream 4.
    Nó nhận Master Table (từ 'transaction_data') và dict của 
    dữ liệu thô (đặc biệt là 'causal_data').
    
    Nó tạo ra các đặc trưng về giá và khuyến mãi.
    """
    
    # 1. Xử lý các đặc trưng trên Master Table (từ transaction_data)
    master_df = _clean_transaction_data(master_df)

    # 2. Xử lý và Merge dữ liệu Causal (Khuyến mãi) (Giải quyết Rủi ro 2)
    if 'causal_data' not in dataframes_dict:
        logging.warning("⚠️ Bỏ qua WS4 (Causal): Không tìm thấy 'causal_data' trong dữ liệu đầu vào.")
        # Nếu không có file causal, ít nhất chúng ta vẫn có các đặc trưng 
        # khuyến mãi từ file transaction (tính ở trên)
        return master_df
        
    df_causal_clean = _clean_causal_data(dataframes_dict['causal_data'])
    
    # 3. Tích hợp (Merge) vào Master Table
    # Khóa (key) của Dunnhumby là 'STORE_ID', 'PRODUCT_ID', 'WEEK_NO'
    # (Giả sử master_df đã có các cột này từ WS1/WS2)
    merge_keys = ['STORE_ID', 'PRODUCT_ID', 'WEEK_NO']
    
    if all(key in master_df.columns for key in merge_keys):
        original_rows = master_df.shape[0]
        master_df = pd.merge(master_df, df_causal_clean, on=merge_keys, how='left')
        
        if master_df.shape[0] != original_rows:
            logging.error("🚨 LỖI (WS4): Merge causal_data đã làm thay đổi số hàng (row explosion)!")
        
        # Điền 0 cho các sản phẩm/tuần không có trong file causal (nghĩa là không khuyến mãi)
        master_df['is_on_display'] = master_df['is_on_display'].fillna(0).astype(int)
        master_df['is_on_mailer'] = master_df['is_on_mailer'].fillna(0).astype(int)
        
        logging.info("✓ Tích hợp Workstream 4 (Giá/Khuyến mãi) thành công.")
        
    else:
        logging.warning("⚠️ Bỏ qua Merge (WS4): Không tìm thấy 'STORE_ID', 'PRODUCT_ID', 'WEEK_NO' để làm khóa (key) merge.")

    return master_df