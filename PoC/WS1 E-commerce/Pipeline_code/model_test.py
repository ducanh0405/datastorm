import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import time
import sys # MỚI: Thêm sys để kiểm tra lỗi

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 1. TẢI DỮ LIỆU SẠCH (TỪ PIPELINE)
# -----------------------------------------------------------------
# Đảm bảo tên file này khớp với file output của pipeline
# DATA_FILE = 'olist_master_table_CLEAN_FINAL_v1.parquet'
DATA_FILE = 'olist_master_table_completed.parquet' # MỚI: Đổi tên file cho khớp với log của bạn

print(f"Đang tải dữ liệu từ: {DATA_FILE}...")
start_time = time.time()
try:
    if DATA_FILE.endswith('.parquet'):
        df = pd.read_parquet(DATA_FILE)
    else:
        df = pd.read_csv(DATA_FILE)
    print(f"Tải xong. Shape ban đầu: {df.shape}. (Mất {time.time() - start_time:.2f}s)")
except FileNotFoundError:
    print(f"🚨 LỖI: Không tìm thấy file {DATA_FILE}. Hãy chạy pipeline trước.")
    sys.exit(1) # MỚI: Thoát script nếu lỗi
except Exception as e:
    print(f"🚨 LỖI: {e}")
    sys.exit(1)

# -----------------------------------------------------------------
# 2. CHUẨN BỊ DỮ LIỆU CHO MÔ HÌNH
# -----------------------------------------------------------------
print("\nĐang chuẩn bị dữ liệu cho mô hình...")

# Logic: Chúng ta chỉ có thể dự đoán review của các đơn "đã giao"
# và chỉ có thể huấn luyện trên các đơn "đã được review" (score > 0)
# MỚI: Thêm cột 'order_status' từ pipeline vào (nếu bạn đã lưu nó)
if 'order_status' in df.columns:
    df_model = df[
        (df['order_status'] == 'delivered') &
        (df['review_score'] > 0) # Lọc ra các đơn chưa review (score = 0)
    ].copy()
else:
    # Giả định nếu không có cột status, ta dùng các đơn đã giao (có delivery_time)
    df_model = df[
        (df['delivery_time_days'] > -999) & # Lọc các đơn chưa giao
        (df['review_score'] > 0) # Lọc các đơn chưa review
    ].copy()

if df_model.empty:
    print("🚨 LỖI: Không tìm thấy dữ liệu đã giao và đã review để huấn luyện.")
    sys.exit(1)

# Tạo biến mục tiêu (Y)
target_col = 'is_good_review'
df_model[target_col] = (df_model['review_score'] == 5).astype(int)
print(f"Phân bổ biến mục tiêu (Y):")
print(df_model[target_col].value_counts(normalize=True).apply(lambda x: f"{x:.1%}"))

# -----------------------------------------------------------------
# 3. CHỌN ĐẶC TRƯNG (FEATURE SELECTION)
# -----------------------------------------------------------------
numeric_features = [
    'delivery_time_days',
    'delivery_vs_estimated_days',
    'order_processing_time_days',
    'price',
    'freight_value',
    'freight_ratio',
    'payment_value_total',
    'payment_installments_total',
    'payment_sequential_count',
    'dist_cust_seller_km',
    'product_weight_g',
    'product_volume_cm3',
    'purchase_day_of_week',
    'purchase_hour'
]

categorical_features = [
    'product_category_name_english',
    'customer_state',
    'seller_state',
    'payment_type_primary',
    'is_weekend'
]

# MỚI: Kiểm tra xem các cột có tồn tại không (phòng trường hợp pipeline của bạn lưu khác)
features = [col for col in (numeric_features + categorical_features) if col in df.columns]
numeric_features = [col for col in numeric_features if col in features]
categorical_features = [col for col in categorical_features if col in features]

if not features:
    print("🚨 LỖI: Không tìm thấy bất kỳ đặc trưng nào trong file. Kiểm tra lại tên cột.")
    sys.exit(1)

X = df_model[features]
y = df_model[target_col]

print(f"\nĐang chuyển đổi {len(categorical_features)} cột sang 'category' dtype...")
for col in categorical_features:
    X[col] = X[col].astype('category')

print("Chuẩn bị dữ liệu (X, y) hoàn tất.")

# -----------------------------------------------------------------
# 4. HUẤN LUYỆN (TRAIN) & ĐÁNH GIÁ (EVALUATE) - NÂNG CẤP
# -----------------------------------------------------------------
print("\nĐang chia Train/Test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MỚI: TÍNH TOÁN TRỌNG SỐ CHO DỮ LIỆU MẤT CÂN BẰNG
# Công thức: (Số lượng class Âm tính) / (Số lượng class Dương tính)
# Class Âm (0 - Bad) / Class Dương (1 - Good)
try:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Mất cân bằng: Tỷ lệ (Xấu/Tốt) là {scale_pos_weight:.2f}")
    print("-> Mô hình sẽ 'phạt' gấp {scale_pos_weight:.2f} lần nếu dự đoán sai review Xấu.")
except ZeroDivisionError:
    print("LỖI: Không có review 'Tốt' (1) trong tập huấn luyện.")
    scale_pos_weight = 1 # Dùng giá trị mặc định

# MỚI: TINH CHỈNH HYPERPARAMETERS
model = lgb.LGBMClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,  # <-- Tham số chống mất cân bằng
    n_estimators=500,                 # <-- Tăng số lượng cây
    learning_rate=0.02,               # <-- Giảm tốc độ học
    n_jobs=-1                         # <-- Sử dụng tất cả CPU
)

print("\nBắt đầu huấn luyện mô hình (Training) NÂNG CẤP...")
start_train = time.time()

# MỚI: Thêm Early Stopping để tự động dừng khi mô hình hết tốt
model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features,
    eval_set=[(X_test, y_test)],      # <-- Dùng tập test để theo dõi
    eval_metric='logloss',
    callbacks=[lgb.early_stopping(100, verbose=False)] # <-- Dừng nếu 100 vòng không cải thiện
)
print(f"✓ Huấn luyện hoàn tất (Mất {time.time() - start_train:.2f}s)")
print(f"Số lượng cây (vòng lặp) tối ưu: {model.best_iteration_}")

# -----------------------------------------------------------------
# 5. XEM KẾT QUẢ
# -----------------------------------------------------------------
print("\nĐang dự đoán trên tập Test...")
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("           KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (V2)")
print("="*50)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy (Độ chính xác tổng thể): {accuracy:.2%}")
print("\n" + "-"*50)

print("📊 Báo cáo Phân loại (Classification Report):")
print(classification_report(y_test, y_pred, target_names=['Bad Review (0)', 'Good Review (1)']))
print("-"*50)

print("🔢 Ma trận nhầm lẫn (Confusion Matrix):")
print("(Hàng = Thực tế, Cột = Dự đoán)")
print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                   index=['Actual: Bad', 'Actual: Good'],
                   columns=['Predicted: Bad', 'Predicted: Good']))
print("="*50)

# -----------------------------------------------------------------
# MỚI: 6. PHÂN TÍCH ĐẶC TRƯNG QUAN TRỌNG
# -----------------------------------------------------------------
print("\n" + "="*50)
print("      TOP 10 ĐẶC TRƯNG QUAN TRỌNG NHẤT")
print("="*50)

# Tạo DataFrame từ độ quan trọng của đặc trưng
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# In top 10
print(feature_importance_df.head(10).to_string(index=False))
print("="*50)