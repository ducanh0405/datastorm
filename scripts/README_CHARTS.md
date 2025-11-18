# 📊 Report Charts Generator

Script để tạo tất cả các charts cần thiết cho báo cáo dự án SmartGrocy.

## 🎯 Charts được tạo

### Bắt buộc (3 charts):

1. **Chart 1: Model Performance Metrics** (`chart1_model_performance.png`)
   - MAE, RMSE, Pinball Loss across quantiles (Q05-Q95)
   - R² Score và Coverage (90%)
   - Chứng minh model tốt

2. **Chart 2: Business Impact KPI Comparison** (`chart2_business_impact.png`)
   - Spoilage Rate: 6.8% → 4.18% (38.48% improvement)
   - Stockout Rate: 5.2% → 3.19% (38.48% improvement)
   - Profit Margin: 12.5% → 15.76% (25.85% improvement)
   - Chứng minh business value với baseline 2024

3. **Chart 3: Forecast Quality** (`chart3_forecast_quality.png`)
   - Prediction intervals (Q05-Q95)
   - Actual values overlay
   - Chứng minh forecast accurate

### Khuyến nghị (2 charts):

4. **Chart 4: Feature Importance (SHAP)** (`chart4_feature_importance.png`)
   - Top 10 features theo SHAP values
   - Chứng minh interpretability

5. **Chart 5: Market Context** (`chart5_market_context.png`)
   - Vietnam e-grocery market growth (2020-2025)
   - Justify problem importance

## 🚀 Cách sử dụng

```bash
# Chạy script để generate tất cả charts
python scripts/generate_report_charts.py

# Hoặc nếu dùng venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
python scripts/generate_report_charts.py
```

## 📁 Output

Tất cả charts sẽ được lưu vào thư mục:
```
reports/report_charts/
├── chart1_model_performance.png
├── chart2_business_impact.png
├── chart3_forecast_quality.png
├── chart4_feature_importance.png
└── chart5_market_context.png
```

## 📋 Requirements

Script sử dụng các thư viện:
- `matplotlib` - Vẽ charts
- `seaborn` - Styling
- `pandas` - Đọc dữ liệu
- `numpy` - Tính toán

Tất cả đã có trong `requirements.txt`.

## 🔧 Customization

Bạn có thể chỉnh sửa script để:
- Thay đổi màu sắc (biến `COLORS`)
- Thay đổi số lượng samples cho Chart 3 (mặc định 200)
- Thay đổi số lượng features cho Chart 4 (mặc định top 10)
- Điều chỉnh kích thước và DPI của charts

## 📊 Data Sources

Script đọc dữ liệu từ:
- `reports/metrics/model_metrics.json` - Model performance metrics
- `reports/backtesting/estimated_results.csv` - Business impact
- `reports/shap_values/feature_importance.csv` - Feature importance
- `reports/predictions_test_set.csv` - Forecast predictions (sample)

## ✅ Checklist

- [x] Chart 1: Model Performance Metrics
- [x] Chart 2: Business Impact KPI Comparison
- [x] Chart 3: Forecast Quality
- [x] Chart 4: Feature Importance (SHAP)
- [x] Chart 5: Market Context

