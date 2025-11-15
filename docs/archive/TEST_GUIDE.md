# Hướng Dẫn Test Pipeline với Sample Data

## Tổng quan

Pipeline đã được cập nhật với các tính năng sau:
1. ✅ Hỗ trợ nhiều models (LightGBM, CatBoost, Random Forest)
2. ✅ QuantileForecaster class đầy đủ với SHAP values
3. ✅ Metrics chi tiết (Pinball Loss, MAE, RMSE, Coverage, R2)
4. ✅ Dashboard visualization tối ưu UX/UI
5. ✅ SHAP values calculation và explanation

## Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

Dependencies chính:
- lightgbm
- catboost
- shap
- pandas
- numpy
- scikit-learn
- plotly

## Chạy Pipeline

### Bước 1: Tạo Master Feature Table

Nếu chưa có master feature table, chạy feature enrichment pipeline:

```bash
python src/pipelines/_02_feature_enrichment.py
```

### Bước 2: Train Models

Train models với mặc định (LightGBM):

```bash
python src/pipelines/_03_model_training.py
```

Hoặc train nhiều models:

```bash
python src/pipelines/_03_model_training.py --model-types lightgbm catboost
```

### Bước 3: Generate Predictions với SHAP

Generate predictions với SHAP values:

```bash
python src/pipelines/_05_prediction.py --shap --sample-size 1000
```

### Bước 4: Tạo Dashboard

Tạo dashboard HTML:

```bash
python src/pipelines/_07_dashboard.py
```

Dashboard sẽ được lưu tại: `reports/dashboard/forecast_dashboard.html`

## Test với Sample Data

Sử dụng script test đã tạo:

```bash
python test_pipeline_sample.py
```

Script này sẽ:
1. Kiểm tra data availability
2. Test training với sample data (10% data hoặc tối đa 10,000 rows)
3. Test prediction
4. Test SHAP values (optional)

## Cấu hình Models

Trong `src/config.py`, bạn có thể:

1. Chọn models để train:
```python
MODEL_TYPES = ['lightgbm', 'catboost', 'random_forest']
```

2. Cấu hình hyperparameters cho từng model:
- `LIGHTGBM_PARAMS`
- `CATBOOST_PARAMS`
- `RANDOM_FOREST_PARAMS`

3. Cấu hình SHAP:
```python
SHAP_CONFIG = {
    'enabled': True,
    'sample_size': 1000,
    'max_display_features': 20,
    'save_plots': True,
}
```

## Kết quả

Sau khi chạy pipeline, bạn sẽ có:

1. **Models**: `models/*.joblib`
   - `lightgbm_q05_forecaster.joblib`
   - `lightgbm_q25_forecaster.joblib`
   - `lightgbm_q50_forecaster.joblib`
   - `lightgbm_q75_forecaster.joblib`
   - `lightgbm_q95_forecaster.joblib`

2. **Predictions**: `reports/predictions_test_set.csv`

3. **Metrics**: `reports/metrics/model_metrics.json`

4. **SHAP Values**: 
   - `reports/shap_values/shap_values.csv`
   - `reports/shap_values/shap_summary.json`

5. **Dashboard**: `reports/dashboard/forecast_dashboard.html`

## Dashboard Features

Dashboard bao gồm:

1. **Statistics Cards**: Total predictions, Quantiles, Models trained, Top features
2. **Predictions Tab**: 
   - Prediction distribution histogram
   - Prediction intervals chart
3. **Metrics Tab**: 
   - Metrics table cho từng model
4. **Feature Importance Tab**: 
   - SHAP values bar chart
   - Top features list

## Troubleshooting

### Lỗi: "Master feature table not found"
- Chạy `_02_feature_enrichment.py` trước

### Lỗi: "No models found"
- Chạy `_03_model_training.py` trước

### Lỗi: "SHAP not available"
- Cài đặt: `pip install shap`

### Lỗi: "CatBoost not available"
- Cài đặt: `pip install catboost` hoặc `pip install catboost --no-build-isolation`

## Notes

- Mặc định chỉ train LightGBM để test nhanh
- Có thể train nhiều models nhưng sẽ mất nhiều thời gian hơn
- SHAP values calculation có thể chậm với dataset lớn, nên dùng `sample_size`
- Dashboard sử dụng Plotly.js CDN, cần internet để load

