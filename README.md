# 🏆 E-GroceryForecaster: Động Cơ Dự Báo Tối Ưu Hóa Kệ Hàng Số tại Việt Nam

[![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Models-XGBoost%20%7C%20LightGBM-green.svg)](https://xgboost.readthedocs.io/en/stable/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

## 📋 Tổng quan Dự án

**E-GroceryForecaster** là một giải pháp khoa học dữ liệu toàn diện được thiết kế đặc biệt cho thị trường thương mại điện tử thực phẩm tại Việt Nam. Dự án tập trung vào việc giải quyết thách thức lớn nhất của ngành: **tối ưu hóa quản lý tồn kho hàng hóa dễ hỏng** thông qua việc dự báo nhu cầu chính xác và các chiến lược vận hành thông minh.

**E-GroceryForecaster** không chỉ là một mô hình dự báo đơn thuần, mà là một hệ thống tích hợp 3 mô-đun hoàn chỉnh:
- 🔮 **Dự báo Nhu cầu**: Dự đoán doanh số bán hàng chính xác cho 28 ngày tới
- 📦 **Tối ưu hóa Tồn kho**: Tính toán các chính sách tồn kho tối ưu
- 💰 **Định giá Động**: Tối đa hóa doanh thu từ hàng tồn sắp hết hạn

Dự án sử dụng các mô hình machine learning tiên tiến (XGBoost/LightGBM) kết hợp với dữ liệu lịch sử bán hàng để tạo ra các dự báo có độ chính xác cao, giúp doanh nghiệp giảm thiểu lãng phí từ hàng hỏng và tối ưu hóa lợi nhuận.

---

## 1. 🎯 Bối cảnh Vấn đề: Nghịch lý của Thị trường Tăng trưởng 76%

Thị trường E-Grocery Việt Nam là phân khúc tăng trưởng nhanh nhất (76,3%) trong toàn cảnh TMĐT, nhưng cũng là phân khúc phức tạp nhất về vận hành. Thách thức cốt lõi là **rủi ro hai mặt (dual risk)**:

1.  **Tồn kho Quá mức (Overstocking):** Dẫn đến **Hư hỏng (Spoilage)**, gây tổn thất tài chính trực tiếp (trung bình ngành ~2% doanh thu).
2.  **Tồn kho Không đủ (Understocking):** Dẫn đến **Hết hàng (Stockouts)**, làm mất doanh thu tức thì và suy giảm lòng trung thành của khách hàng (chuẩn ngành < 5%).

Các phương pháp truyền thống (EOQ, JIT) thất bại vì chúng dựa trên các giả định tĩnh, không thể xử lý sự biến động phức tạp của nhu cầu TMĐT.

---

## 2. 💡 Giải pháp Đề xuất: Động cơ Tích hợp 3 Mô-đun

Chúng tôi đề xuất một hệ thống khép kín, nơi dự báo chính xác sẽ kích hoạt các hành động vận hành thông minh.

1.  **Mô-đun 1: Lõi Dự báo Nhu cầu (Forecasting Core)**
    * **Chức năng:** Tạo ra dự báo doanh số chi tiết ở cấp độ SKU cho 28 ngày tới.
    * **Kỹ thuật:** Sử dụng **XGBoost/LightGBM** (xem Lý do Kỹ thuật bên dưới).
2.  **Mô-đun 2: Tối ưu hóa Tồn kho (Inventory Optimization)**
    * **Chức năng:** Tự động tính toán các chính sách tồn kho tối ưu từ kết quả dự báo.
    * **Đầu ra:** Tồn Kho An Toàn (Safety Stock) và Điểm Đặt Hàng Lại (Reorder Point).
3.  **Mô-đun 3: Định giá Động (Dynamic Pricing)**
    * **Chức năng:** Tối đa hóa doanh thu từ hàng tồn sắp hết hạn, chuyển đổi "lỗ 100%" (hủy hàng) thành "bán hàng giảm giá".
    * **Kỹ thuật:** Áp dụng logic dựa trên Heuristic (Giai đoạn 1) và Học Tăng Cường (Lộ trình Giai đoạn 2).

---

## 3. 🧪 Kiến trúc Kỹ thuật & Lý do (Rationale)

Lựa chọn kiến trúc của chúng tôi không dựa trên xu hướng, mà dựa trên bằng chứng thực nghiệm (empirical evidence) và sự phù hợp tuyệt đối với bài toán "E-Grocery" (dữ liệu dạng bảng, gián đoạn, và yêu cầu tối ưu hóa tồn kho).

### 1. Lựa chọn Mô hình Chủ lực: Gradient Boosted Decision Trees (LightGBM)

Chúng tôi chọn **LightGBM** (một triển khai GBDT) làm động cơ dự báo cốt lõi, thay vì các kiến trúc Deep Learning phức tạp.

**Bằng chứng 1 (Từ thực tiễn):** Trong cuộc thi dự báo bán lẻ M5 (Walmart) — cuộc thi benchmark quy mô lớn và gần nhất với bài toán này — các giải pháp chiến thắng áp đảo (cả về Độ chính xác và Độ không chắc chắn) đều dựa trên **LightGBM**.

**Bằng chứng 2 (Từ học thuật):** Các nghiên cứu so sánh (benchmarks) chỉ ra rằng GBDT thường xuyên vượt trội hơn các mô hình Deep Learning trên dữ liệu dạng bảng (tabular data).

**Bằng chứng 3 (Từ đặc tính dữ liệu):** Dữ liệu E-Grocery có tính gián đoạn cao (nhiều SKU có doanh số bằng 0), nhiều đặc trưng phân loại (category, brand), và bị ảnh hưởng bởi các sự kiện rời rạc (khuyến mãi, lễ). LightGBM được thiết kế để xử lý hiệu quả các đặc tính này một cách tự nhiên.

### 2. Kiến trúc Dự báo Xác suất (Probabilistic Forecasting Architecture)

Một dự báo điểm (point forecast - ví dụ: "dự báo bán 10 hộp") là vô dụng đối với bài toán E-Grocery, vì nó không trả lời được câu hỏi: *"Nhưng rủi ro bán được 15 hộp (hết hàng) hoặc 5 hộp (hư hỏng) là bao nhiêu?"*

Do đó, chúng tôi không xây dựng một mô hình, mà là một hệ thống dự báo xác suất sử dụng **Quantile Regression** của LightGBM (objective='quantile').

Pipeline của chúng tôi sẽ huấn luyện song song (ít nhất) ba mô hình để tạo ra một khoảng dự báo (prediction interval) cho mỗi SKU:

**Dự báo Trung vị (Q50 - alpha=0.5):**
- **Mục đích:** Cung cấp ước tính "thực tế" nhất về nhu cầu ($\mu_D$)
- **Ứng dụng:** Lập kế hoạch tài chính, dự báo doanh thu cơ sở

**Dự báo Ngưỡng An toàn (Q95 - alpha=0.95):**
- **Mục đích:** Cung cấp kịch bản nhu cầu cao (chỉ có 5% khả năng nhu cầu thực tế vượt qua mức này)
- **Ứng dụng (Mô-đun 2):** Đây là đầu vào cốt lõi để tính Tồn Kho An Toàn (Safety Stock) và Điểm Đặt Hàng Lại (Reorder Point)
- **Công thức:** $$\text{ROP} = \text{Dự báo Q95 Daily} \times \text{Lead Time (days)}$$

**Dự báo Rủi ro Tồn kho (Q05 - alpha=0.05):**
- **Mục đích:** Cung cấp kịch bản nhu cầu thấp (chỉ có 5% khả năng nhu cầu thực tế thấp hơn mức này)
- **Ứng dụng (Mô-đun 3):** Kích hoạt Định giá Động (Dynamic Pricing). Nếu Tồn kho hiện tại > Dự báo Q05 cho số ngày còn lại của hạn sử dụng, hệ thống sẽ tự động đề xuất giảm giá để tránh hư hỏng

### 3. Pipeline Kỹ thuật Đặc trưng (Feature Engineering)

Mô hình GBDT chỉ thực sự mạnh mẽ khi được cung cấp các đặc trưng chất lượng. Dựa trên 4 PoC (Olist, M5, RetailRocket, Dunnhumby), pipeline của chúng tôi sẽ tự động làm giàu (enrich) dữ liệu thô với các nhóm đặc trưng đã được kiểm chứng:

**Đặc trưng Chuỗi thời gian (Time-Series):**
- Giá trị trễ (Lags t-7, t-14, t-28)
- Cửa sổ trượt (Rolling means/std 7/14/28 ngày)

**Đặc trưng Lịch & Sự kiện (Calendar):**
- `day_of_week`, `is_holiday` (Tết)
- `is_event` (Sale 10/10)
- `days_to/from_holiday`

**Đặc trưng Khuyến mãi & Giá (Price/Promo):**
- `is_promotion`, `discount_percentage`
- `price_elasticity_proxy`

**Đặc trưng Sản phẩm & E-commerce:**
- `category`, `brand`, `shelf_life_days`
- `avg_review_score`, `freight_value` (phí ship)

**Đặc trưng Hành vi (Behavioral) - nếu có dữ liệu:**
- `add_to_cart_rate`, `view_to_purchase_ratio`

### 4. Ngăn xếp Công nghệ (Tech Stack)

**Ngôn ngữ & Xử lý Dữ liệu:**
- Python, Pandas, Polars (để tối ưu hóa xử lý dữ liệu lớn, như kinh nghiệm từ M5)

**Mô hình hóa (Modeling):**
- LightGBM, Scikit-learn (cho pipeline và đánh giá)

**Quản lý & Trình diễn:**
- Git, Jupyter Notebooks, Streamlit (cho dashboard demo chung kết)

---

## 4. 🚀 Cài đặt và Sử dụng (Installation & Usage)

1.  Clone repository này:
    ```bash
    git clone [https://github.com/ducanh0405/datastorm.git](https://github.com/ducanh0405/datastorm.git)
    cd E-GroceryForecaster
    ```

2.  (Khuyến nghị) Tạo một môi trường ảo (virtual environment):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên Windows: venv\Scripts\activate
    ```

3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

4.  Chạy các Notebooks chính trong thư mục `/notebooks`:
    ```bash
    jupyter-lab
    ```
    * `01_EDA_and_Feature_Engineering.ipynb`
    * `02_Model_Training_XGBoost.ipynb`
    * `03_Inventory_Logic_Simulation.ipynb`

---

## 5. 📁 Cấu trúc Thư mục (Repository Structure)
📁 E-Grocery_Forecaster/
│
├── 📄 .gitignore                 # File quan trọng: Bỏ qua data, models, venv
├── 📄 README.md                  # Hướng dẫn cài đặt, chạy pipeline và mô tả dự án
├── 📄 requirements.txt           # Danh sách thư viện (pandas, polars, lgbm...)
│
├── 📁 data/
│   │
│   ├── 📁 1_poc_data/            # Dữ liệu "kinh điển" cho 4 Workstream
│   │   ├── ws1_olist/
│   │   ├── ws2_m5/
│   │   ├── ws3_retailrocket/
│   │   └── ws4_dunnhumby/
│   │
│   ├── 📁 2_raw/                 # Nơi chứa DỮ LIỆU THẬT (của cuộc thi)
│   │   └── .gitkeep             # Placeholder để giữ thư mục này trên Git
│   │
│   └── 📁 3_processed/           # Đầu ra của pipeline: Bảng Master Table
│       └── master_feature_table.parquet
│
├── 📁 notebooks/                 # Sân chơi & Bản nháp (Nơi thực hiện PoC)
│   │
│   ├── 📁 archive/
│   │   ├── 01_ws1_olist_poc.ipynb           (Notebook PoC 1 của bạn)
│   │   ├── 02_ws2_m5_poc.ipynb              (Notebook PoC 2 của bạn)
│   │   ├── 03_ws3_retailrocket_poc.ipynb    (Notebook PoC 3 của bạn)
│   │   ├── 04_ws4_dunnhumby_poc.ipynb       (Notebook PoC 4 của bạn)
│   │
│   ├── 📄 05_competition_eda.ipynb    # (Quan trọng) EDA dữ liệu thật (Giảm thiểu Rủi ro 3)
│   └── 📄 06_baseline_model.ipynb     # (Quan trọng) Chạy baseline (Giảm thiểu Rủi ro 4)
│
├── 📁 src/                     # Code "sạch" (Production) của Giai đoạn 2
│   │
│   ├── 📁 features/            # 💡 THƯ VIỆN CODE (Giảm thiểu Rủi ro 1)
│   │   │   # Đây là nơi chứa các hàm "sạch" rút ra từ 4 PoC
│   │   ├── 📄 ws1_ecommerce_features.py    (Hàm tính review_score, freight_ratio...)
│   │   ├── 📄 ws2_timeseries_features.py   (Hàm tạo lag/rolling, event flags...)
│   │   ├── 📄 ws3_behavior_features.py     (Hàm tính add_to_cart_rate...)
│   │   └── 📄 ws4_price_features.py        (Hàm tính elasticity...)
│   │
│   ├── 📁 pipelines/           # 💡 KIẾN TRÚC SƯ PIPELINE (Giảm thiểu Rủi ro 1)
│   │   │   # Các script này "gọi" các hàm từ src/features/
│   │   ├── 📄 01_load_data.py            (Tải dữ liệu "raw" của cuộc thi)
│   │   ├── 📄 02_feature_enrichment.py   (Tích hợp 4 Workstream, Giảm thiểu Rủi ro 2)
│   │   ├── 📄 03_model_training.py       (Huấn luyện mô hình cuối cùng)
│   │   └── 📄 04_run_pipeline.py         (Script chính để chạy 1, 2, 3)
│   │
│   └── 📁 utils/
│       └── 📄 validation.py          (Chứa các hàm kiểm tra chất lượng dữ liệu)
│
├── 📁 models/                  # Nơi lưu các mô hình đã huấn luyện
│   ├── 📄 final_forecaster.joblib
│   └── 📄 model_features.json      (Lưu danh sách feature mô hình đã dùng)
│
├── 📁 planning/                # Nơi chứa các PoC/Demo của Workstream 1
│   ├── 📄 schema.sql
│   └── 📄 schemadiagram_olist.jpg
│
└── 📁 reports/
    ├── 📁 metrics/             # Nơi lưu kết quả benchmark
    │   ├── 📄 baseline_metrics.json
    │   └── 📄 final_model_metrics.json
    │
    └── 📄 final_presentation.md  (File .md hoặc .pptx cho Vòng Chung kết)
## 6. 📈 Đo lường Thành công (Measuring Success)

Thành công của dự án được đo lường trên cả hai mặt: Kỹ thuật và Kinh doanh.

### Chỉ số Kỹ thuật (Technical Metrics)

* **RMSE (Root Mean Squared Error):** Phạt nặng các lỗi dự báo lớn.
* **MAE (Mean Absolute Error):** Dễ diễn giải (sai lệch trung bình bao nhiêu đơn vị).
* **WAPE (Weighted Absolute Percentage Error):** Chỉ số chính từ M5, tập trung vào độ chính xác của các SKU quan trọng nhất.
