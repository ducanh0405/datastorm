# 🏆 SmartGrocy

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Models-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Interactive%20Plotly-red.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

**Giải pháp AI tiên tiến cho dự báo nhu cầu và tối ưu hóa tồn kho trong ngành thực phẩm tươi sống tại Việt Nam**

## 📋 Tổng quan

SmartGrocy là hệ thống dự báo nhu cầu thông minh, sử dụng machine learning để giải quyết vấn đề tồn kho trong ngành thương mại điện tử thực phẩm. Dự án kết hợp LightGBM với feature engineering chuyên sâu để tạo ra các dự báo chính xác, giúp doanh nghiệp giảm lãng phí và tối ưu hóa lợi nhuận.

### 🎯 Tính năng chính
- 🔮 **Dự báo xác suất**: Quantile regression với prediction intervals (Q05-Q95)
- 📦 **Pipeline hiện đại**: Prefect orchestration với data quality monitoring
- 📊 **Dashboard tương tác**: Visualization với Plotly
- 🚀 **Performance tối ưu**: Polars processing, 6-15x faster than pandas
- 🔄 **End-to-end workflow**: Từ raw data đến production predictions

## 🆕 Phiên bản mới: Modern Pipeline với Data Quality Monitoring

### ✨ Tính năng mới:
- 🔄 **Pipeline Orchestration**: Prefect-based DAG workflow
- 📊 **Data Quality Monitoring**: Great Expectations + custom validations
- 🚨 **Alerting System**: Tự động cảnh báo chất lượng dữ liệu
- 💾 **Intelligent Caching**: Tối ưu hóa hiệu năng
- 🔍 **Drift Detection**: Phát hiện thay đổi phân phối dữ liệu

## 🚀 Cài đặt và Chạy

### Yêu cầu hệ thống
- Python 3.10+
- 16GB+ RAM (32GB khuyến nghị)
- Windows/Linux/MacOS

### Cài đặt nhanh
```bash
# Clone repository
git clone https://github.com/ducanh0405/datastorm.git
cd E-Grocery_Forecaster

# Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Khởi tạo data quality monitoring
python scripts/setup_data_quality.py
```

### Chạy Pipeline
```bash
# Chạy pipeline hiện đại v2 với monitoring (khuyến nghị)
python run_modern_pipeline_v2.py --full-data

# Hoặc chạy pipeline hiện đại v1
python run_modern_pipeline.py --full-data

# Test với sample data (nhanh hơn)
python run_modern_pipeline_v2.py --full-data --sample 0.1

# Giám sát chất lượng dữ liệu
python scripts/monitor_data_quality.py
```

### Tạo Dashboard
```bash
# Dashboard được tạo tự động sau khi chạy prediction pipeline
# Hoặc chạy trực tiếp module dashboard
python -m src.pipelines._07_dashboard

# Mở dashboard (sau khi đã tạo)
start reports/dashboard/forecast_dashboard.html  # Windows
open reports/dashboard/forecast_dashboard.html   # Mac
```

## 📁 Cấu trúc dự án

```
E-Grocery_Forecaster/
├── src/
│   ├── pipelines/          # Pipeline modules (_01 đến _06)
│   ├── features/           # Feature engineering (WS0-WS6)
│   ├── utils/             # Utilities (caching, validation, etc.)
│   └── config.py          # Configuration
├── data/
│   ├── poc_data/         # POC test datasets (optional)
│   ├── 2_raw/            # Production raw data
│   └── 3_processed/      # Processed feature tables
├── models/               # Trained models
├── reports/              # Outputs & dashboard
├── scripts/              # Utility scripts
├── notebook/             # Jupyter notebooks
└── tests/               # Unit tests
```

## 📊 Kết quả & Metrics

- **Model Performance**: Q50 Pinball Loss = 0.0492, Coverage = 78.6%
- **Pipeline Speed**: 4.7x faster than baseline
- **Data Processing**: 2.6M+ transactions, 92K+ products
- **Features**: 53 engineered features across 6 workstreams

## 🔧 Tech Stack

- **ML**: LightGBM (default), Optuna, CatBoost (optional)
- **Data**: Pandas, Polars, PyArrow
- **Visualization**: Plotly, Matplotlib
- **Orchestration**: Prefect (optional server mode)
- **Quality**: Great Expectations

**Note**: CatBoost and Prefect server are optional. LightGBM is the default and recommended model.

## 📚 Documentation

- **[QUICKSTART.md](docs/QUICKSTART.md)** - Hướng dẫn setup nhanh
- **[OPERATIONS.md](docs/OPERATIONS.md)** - Vận hành production
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Đóng góp dự án

## 🤝 Liên hệ

**Email**: ducanh0405@gmail.com  
**License**: MIT

---

**🎯 Dự án đã sẵn sàng cho demo và production deployment!**