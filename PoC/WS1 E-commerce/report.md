# 🏆 Báo cáo Dự án Olist: Từ Dữ liệu thô đến Mô hình Vận hành

**Trạng thái:** ✅ **Hoàn thành (Cả 2 Workstream)**

---

## 1. Tóm tắt Mục tiêu

Dự án này bao gồm hai luồng công việc (Workstream) chính, thể hiện năng lực end-to-end:
1.  **Workstream 1 (Data Engineering):** Xây dựng một pipeline kỹ thuật dữ liệu mạnh mẽ để xử lý 9 tệp CSV thô, làm sạch, làm giàu và xuất ra một Bảng Đặc trưng (Feature Table) tổng thể, sạch và sẵn sàng cho ML.
2.  **Workstream 2 (Machine Learning):** Xây dựng một pipeline huấn luyện (training script) chuyên nghiệp, sử dụng Bảng Đặc trưng từ WS1 để tự động tối ưu hóa (tuning) và đóng gói (save) một mô hình dự đoán (review 5 sao), sẵn sàng để triển khai (deploy).

## 2. WS1 - Giai đoạn 1: Lập Kế hoạch & Merge An toàn (V1 PoC)

Giai đoạn đầu tiên tập trung vào việc khám phá và hợp nhất 9 tệp CSV một cách an toàn bằng Jupyter Notebooks (`Merge and clean.ipynb`, `EDA_featureengineering.ipynb`).

* **Xác định "Bẫy Hợp nhất" 💣:** Chủ động phát hiện `order_payments` là quan hệ 1-Nhiều, có nguy cơ gây "nổ" dữ liệu.
* **Giải pháp cho Bẫy:** Đã thực hiện `aggregate` (gộp) bảng `order_payments` theo `order_id` **trước khi** merge, ngăn chặn hoàn toàn lỗi nhân dữ liệu.

## 3. WS1 - Giai đoạn 2: Tối ưu hóa Pipeline (V2 Production)

Giai đoạn này nâng cấp các khám phá từ V1 thành một pipeline "sạch" và mạnh mẽ (`Completed Pipeline.py`).

* **Tối ưu 1: Tái cấu trúc "Production-Ready" 🧩:** Toàn bộ logic đã được **tái cấu trúc (refactored)** thành một script Python dựa trên các **hàm (functions)** rõ ràng (ví dụ: `load_data()`, `aggregate_payments()`, `merge_tables()`, `clean_and_impute()`), có logging và validation.
* **Tối ưu 2: Làm giàu Đặc trưng (Geolocation) 🗺️:** Tích hợp thành công bảng `geolocation` (1 triệu hàng) bằng cách `aggregate` trước, sau đó merge 2 lần và tạo ra đặc trưng `dist_cust_seller_km` (Haversine).
* **Tối ưu 3: Sửa lỗi Rò rỉ Dữ liệu (Academic Rigor) 💧:** (Đã xác định trong kế hoạch) Triển khai các đặc trưng "an toàn theo thời gian" (time-safe), tránh sử dụng thông tin tương lai để dự đoán quá khứ.

---

## 4. 🎁 Sản phẩm Bàn giao (Artifacts Delivered)

Dự án đã bàn giao 2 bộ "artifact" rõ rệt cho :

### 4.1. Data Engineering Artifacts

Đây là các sản phẩm của pipeline xử lý dữ liệu thô.

| Tên File | Loại                 | Mục đích |
|:---|:---------------------|:---|
| `Completed Pipeline.py` | **Engine **          | Script Python "sạch", production-ready, thực thi toàn bộ logic của WS1. |
| `olist_master_table_final.csv` | **Data Output (V2)** | **Sản phẩm chính của WS1.** Bảng đặc trưng cuối cùng, đã làm giàu và sạch. |
| `Merge and clean.ipynb` | Code (V1 - PoC)      | Khám phá (PoC) cho việc merge và làm sạch ban đầu. |
| `EDA_featureengineering.ipynb` | Code (V1 - PoC)      | Khám phá (PoC) cho việc tạo đặc trưng. |
| `schema_planning/` | Planning             | Sơ đồ quan hệ và kế hoạch thực thi merge. |

### 4.2. Machine Learning Artifacts

Đây là các sản phẩm của pipeline huấn luyện mô hình.

| Tên File | Loại             | Mục đích |
|:---|:-----------------|:---|
| `train_model.py` | **Engine **      | Script Python "sạch", tự động tải dữ liệu WS1, chạy `RandomizedSearchCV`, và lưu kết quả. |
| `lgbm_review_model_v1.joblib` | **Model Output** | **Sản phẩm chính của WS2.** File mô hình LightGBM đã được huấn luyện và tối ưu, sẵn sàng để dự đoán. |
| `model_features_v1.json` | **Model Output** | File JSON chứa danh sách các đặc trưng (features) và đặc trưng `categorical` mà mô hình cần để dự đoán. |
| `model_metrics_v1.json` | **Model Output** | File JSON chứa kết quả (Accuracy, ROC AUC, Báo cáo Phân loại) của mô hình trên tập Test. |

---

## 5. 💡 Năng lực đã Chứng minh

Dự án đã chứng minh năng lực chuyên sâu trên cả hai lĩnh vực:

### 5.1. PoC (DATA ENGINEERING / ML )

* ✅ **Xử lý Dữ liệu Lớn:** Xử lý và aggregate hiệu quả bảng `geolocation` (1M+ records).
* ✅ **Pipeline Phức tạp:** Hợp nhất 9 bảng CSV, xử lý thành công bẫy merge 1-Nhiều.
* ✅ **Tạo Đặc trưng (Feature Engineering):** Tạo 20+ đặc trưng nghiệp vụ (Haversine distance, cyclical time features, v.v.).
* ✅ **Đảm bảo Chất lượng:** Tích hợp bước `comprehensive_validation` (kiểm tra toàn diện) vào pipeline.


* ✅ **Xử lý Mất cân bằng (Imbalance):** Áp dụng `scale_pos_weight` để mô hình tập trung vào class thiểu số (review "Xấu").
* ✅ **Tối ưu hóa Tự động (Tuning):** Sử dụng `RandomizedSearchCV` và `StratifiedKFold` (Cross-Validation) để tự động tìm ra siêu tham số (hyperparameters) tốt nhất, thay vì "đoán" thủ công.
* ✅ **Quản lý "Artifact" (ML-Ops):** Thiết kế pipeline huấn luyện để "xuất bản" (publish) các file cần thiết cho việc triển khai (`.joblib`, `features.json`, `metrics.json`).
* ✅ **Đánh giá Mô hình:** Sử dụng thước đo `roc_auc` (thay vì chỉ `accuracy`) để tối ưu hóa mô hình, phù hợp với bài toán mất cân bằng.