# 🏆 Workstream 1: Olist Data Engineering & Feature Enrichment

**Trạng thái:** ✅ **Hoàn thành (Đã Tối ưu hóa V2 & ML Testing)**

---

## 1. Tóm tắt Mục tiêu

Mục tiêu của Workstream 1 là chứng minh năng lực **Kỹ thuật Dữ liệu (Data Engineering)**. Cụ thể là khả năng:
1.  **Xử lý (Handle):** Một cơ sở dữ liệu quan hệ (relational) phức tạp (9 tệp CSV Olist).
2.  **Hợp nhất (Merge):** Hợp nhất các bảng một cách an toàn, đặc biệt là xử lý các "bẫy" (traps) 1-Nhiều.
3.  **Làm giàu (Enrich):** Tạo ra các đặc trưng (features) nghiệp vụ có giá trị cao từ dữ liệu thô.
4.  **Tối ưu hóa (Optimize):** Xây dựng một pipeline "sạch", hiệu quả, và chính xác về mặt học thuật.

Báo cáo này xác nhận chúng tôi đã hoàn thành và tối ưu hóa thành công pipeline này.

---

## 2. Giai đoạn 1: Lập Kế hoạch & Merge An toàn (V1)

Giai đoạn đầu tiên tập trung vào việc hợp nhất 9 tệp CSV một cách an toàn và chính xác.

### 2.1. Lập Kế hoạch Chiến lược (Schema Diagram)

Chúng tôi đã tạo một **Sơ đồ Quan hệ (Schema Diagram)** "tối ưu". Sơ đồ này không chỉ mô tả cấu trúc, mà còn là một **kế hoạch thực thi**:
* **Xác định "Bẫy Hợp nhất" 💣:** Chủ động phát hiện `order_payments` là quan hệ 1-Nhiều (1 đơn hàng, nhiều thanh toán), có nguy cơ gây "nổ" dữ liệu (data explosion) nếu merge trực tiếp.
* **"Cắt tỉa" (Prune) ✂️:** Chủ động xác định `geolocation` (1 triệu hàng) là "Ưu tiên thấp V1" do chi phí xử lý cao, và đề xuất dùng `customer_state` làm proxy.



### 2.2. Thực thi Merge & Kiểm tra (Validation)

* **Hành động:** Xây dựng `Merge and clean.ipynb` (hợp nhất và làm sạch) và `EDA_featureengineering.ipynb` (phân tích và tạo đặc trưng).
* **Giải pháp cho Bẫy:** Đã thực hiện `aggregate` (gộp) bảng `order_payments` theo `order_id` **trước khi** merge, ngăn chặn hoàn toàn lỗi nhân dữ liệu.
* **Kết quả Kiểm tra Toàn vẹn:**
    ```bash
    Số lượng hàng bị trùng lặp (duplicate) theo khóa [order_id, order_item_id]: 0
    -> ✅ TỐT! Pipeline hợp nhất (merge) an toàn.
    ```

---

## 3. Giai đoạn 2: Tối ưu hóa Pipeline (V2)

Sau khi có pipeline V1 cơ bản, chúng tôi đã thực hiện 3 cấp độ tối ưu hóa để nâng cấp PoC lên mức "xuất sắc".

### 3.1. Tối ưu 1: Tái cấu trúc "Production-Ready" 🧩

* **Vấn đề:** Code V1 là các script chạy tuần tự, khó tái sử dụng.
* **Giải pháp (V2):** Toàn bộ logic đã được **tái cấu trúc (refactored)** thành một pipeline "sạch" (`Completed Pipeline.py`), dựa trên các **hàm (functions)** Python rõ ràng (ví dụ: `load_data()`, `aggregate_payments()`, `merge_tables()`, `create_features()`, `clean_and_impute()`).
* **Giá trị:** Chứng minh năng lực xây dựng code **mô-đun (modular)** và **tái sử dụng (reusable)**.

### 3.2. Tối ưu 2: Làm giàu Đặc trưng Nghiệp vụ (Geolocation) 🗺️

* **Vấn đề:** Pipeline V1 đã "cắt tỉa" `geolocation`.
* **Giải pháp (V2):** Đã tích hợp thành công bảng `geolocation` (1 triệu hàng):
    1.  **Aggregate:** Gộp (groupby) 1 triệu hàng `geolocation` theo `zip_code_prefix` để lấy `lat`/`lng` trung bình.
    2.  **Merge:** Hợp nhất (merge) 2 lần vào Bảng Tổng thể (cho `customer` và `seller`).
    3.  **Feature Mới:** Tạo ra đặc trưng `distance_seller_customer` (khoảng cách người bán-người mua, tính bằng km) sử dụng công thức **Haversine**.
* **Giá trị:** Đặc trưng `distance` này là một yếu tố dự báo (predictor) nghiệp vụ cực kỳ mạnh mẽ.

### 3.3. Tối ưu 3: Sửa lỗi Rò rỉ Dữ liệu (Academic Rigor) 💧

* **Vấn đề:** Đặc trưng `avg_review_score_product` (V1) bị **rò rỉ dữ liệu (data leakage)**, vì nó dùng review của *tương lai* để tính trung bình cho đơn hàng *quá khứ*.
* **Giải pháp (V2):** Đã triển khai một đặc trưng **"an toàn theo thời gian" (time-safe)** bằng cách sử dụng `sort_values('timestamp')` -> `groupby().expanding().mean()` -> `shift(1)`.
* **Giá trị:** Chứng minh sự nghiêm túc về mặt học thuật và hiểu biết sâu sắc về **xác thực chuỗi thời gian (time-series validation)**.

---

## 4. 🎁 Sản phẩm Bàn giao (Deliverables)

| Tên File                                    | Mục đích |
|:--------------------------------------------| :--- |
| `Pipeline_code/Completed Pipeline.py`       | (Code) Pipeline hoàn chỉnh, production-ready với logging và validation. |
| `Pipeline_code/EDA_featureengineering.ipynb`| (Code) Phân tích khám phá dữ liệu và kỹ thuật đặc trưng. |
| `Pipeline_code/Merge and clean.ipynb`       | (Code) Hợp nhất và làm sạch dữ liệu ban đầu. |
| `Pipeline_code/model_test.py`               | (Code) Script test mô hình ML cơ bản trên dữ liệu đã xử lý. |
| `Pipeline_code/olist_master_table_final.csv`| (Data) Bảng đặc trưng cuối cùng, đã làm giàu và sạch. |
| `Pipeline_code/olist_master_table_final.parquet`| (Data) Dữ liệu định dạng Parquet hiệu quả cho phân tích lớn. |
| `Pipeline_code/data/`                        | (Data) Thư mục chứa tất cả 9 file CSV gốc của Olist. |
| `schema_planning/schema.sql`                | (Plan) Schema SQL tối ưu với ghi chú chiến thuật và xử lý bẫy. |
| `schema_planning/schemadiagram_olist.jpg`   | (Plan) Sơ đồ quan hệ database với ghi chú execution plan. |

## 5. 💡 Năng lực đã Chứng minh (Capabilities Demonstrated)

Workstream 1 đã chứng minh đội ngũ có năng lực chuyên sâu về:

* ✅ **Kỹ thuật Dữ liệu (Data Engineering):** Xử lý pipeline dữ liệu quan hệ phức tạp với 9 bảng CSV.
* ✅ **Xử lý Dữ liệu Lớn (Big Data):** Xử lý và aggregate các bảng lớn (geolocation 1M+ records).
* ✅ **Tạo Đặc trưng (Feature Engineering):** Tạo ra 23+ đặc trưng nghiệp vụ có giá trị cao (Haversine distance, time-series features).
* ✅ **Học thuật (Academic Rigor):** Phát hiện và sửa các lỗi tinh vi (Data Leakage trong time-series).
* ✅ **Mô hình hóa ML Cơ bản:** Áp dụng LightGBM cho bài toán phân loại review (accuracy ~74%) trên dữ liệu đã xử lý.