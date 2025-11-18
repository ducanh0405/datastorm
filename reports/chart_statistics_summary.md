# SmartGrocy Chart Statistics Summary

Tài liệu này tổng hợp tất cả các thông số và thống kê được sử dụng để tạo các charts trong báo cáo SmartGrocy.

---

## 📊 1. Model Performance Metrics

**Nguồn**: `reports/metrics/model_metrics.json`

### 1.1. Quantile Forecast Metrics

#### Q05 (5th Percentile)
- **Pinball Loss**: 0.0468
- **MAE (Mean Absolute Error)**: 0.752
- **RMSE (Root Mean Squared Error)**: 1.198
- **MAPE (Mean Absolute Percentage Error)**: 0.687% (68.7%)
- **Valid Samples**: 842,120 / 900,000

#### Q25 (25th Percentile)
- **Pinball Loss**: 0.1471
- **MAE**: 0.462
- **RMSE**: 0.771
- **MAPE**: 0.414% (41.4%)
- **Valid Samples**: 842,120 / 900,000

#### Q50 (Median - 50th Percentile)
- **Pinball Loss**: 0.1917
- **MAE**: 0.383
- **RMSE**: 0.652
- **MAPE**: 0.410% (41.0%)
- **Valid Samples**: 842,120 / 900,000

#### Q75 (75th Percentile)
- **Pinball Loss**: 0.1625
- **MAE**: 0.437
- **RMSE**: 0.713
- **MAPE**: 0.574% (57.4%)
- **Valid Samples**: 842,120 / 900,000

#### Q95 (95th Percentile)
- **Pinball Loss**: 0.0613
- **MAE**: 0.762
- **RMSE**: 1.114
- **MAPE**: 1.072% (107.2%)
- **Valid Samples**: 842,120 / 900,000

### 1.2. Overall Model Quality Metrics

- **R² Score**: 0.8234 (82.34%)
- **90% Coverage**: 0.8347 (83.47%)

### 1.3. Average Performance Across All Quantiles

- **Average MAE**: 0.559
- **Average RMSE**: 0.890
- **Average Pinball Loss**: 0.1219

---

## 💼 2. Business Impact Metrics

**Nguồn**: `reports/backtesting/estimated_results.csv`

### 2.1. Spoilage Rate

| Metric | Baseline | ML Model | Improvement | Improvement % |
|--------|----------|----------|--------------|---------------|
| **Spoilage Rate** | 8.2% | 4.92% | 3.28% | **40.0%** |

### 2.2. Stockout Rate

| Metric | Baseline | ML Model | Improvement | Improvement % |
|--------|----------|----------|--------------|---------------|
| **Stockout Rate** | 7.5% | 5.0625% | 2.4375% | **32.5%** |

### 2.3. Profit Margin

| Metric | Baseline | ML Model | Improvement | Improvement % |
|--------|----------|----------|--------------|---------------|
| **Profit Margin** | 15.0% | 20.625% | 5.625% | **37.5%** |

### 2.4. Summary Statistics

- **Average Improvement**: 36.67%
- **Total Efficiency Gain**: 36.67%

---

## 🔍 3. Feature Importance (SHAP Values)

**Nguồn**: `reports/shap_values/feature_importance.csv`

### Top 10 Most Important Features

| Rank | Feature | Mean Absolute SHAP | Mean SHAP | Std SHAP | Impact Type |
|------|---------|-------------------|-----------|----------|-------------|
| 1 | `rolling_mean_24_lag_1` | 0.3864 | 0.0961 | 0.7098 | Mixed (36.45% positive, 63.55% negative) |
| 2 | `sales_quantity_lag_1` | 0.1871 | 0.0255 | 0.7579 | Mixed (27.8% positive, 72.2% negative) |
| 3 | `dow_sin` | 0.1019 | 0.0108 | 0.1464 | Mixed (34.15% positive, 65.85% negative) |
| 4 | `rolling_mean_168_lag_1` | 0.0959 | 0.0137 | 0.2114 | Mixed (29.3% positive, 70.7% negative) |
| 5 | `sales_quantity_lag_24` | 0.0198 | -0.0032 | 0.0925 | Mixed (39.1% positive, 60.9% negative) |
| 6 | `rolling_std_168_lag_1` | 0.0197 | -0.0043 | 0.0497 | Balanced (51.6% positive, 48.4% negative) |
| 7 | `rolling_std_24_lag_1` | 0.0160 | 0.0043 | 0.0520 | Mixed (41.55% positive, 58.45% negative) |
| 8 | `dow_cos` | 0.0153 | 0.0017 | 0.0251 | Balanced (49.8% positive, 50.2% negative) |
| 9 | `sales_quantity_lag_48` | 0.0088 | -0.0023 | 0.0430 | Balanced (52.9% positive, 47.1% negative) |

### Feature Categories Distribution

- **Time-based Features**: Features containing keywords: `hour`, `day`, `week`, `month`, `time`, `lag`
- **Weather Features**: Features containing keywords: `temp`, `weather`, `precip`, `humidity`, `wind`
- **Price Features**: Features containing keywords: `price`, `discount`, `promo`
- **Sales Features**: Features containing keywords: `sales`, `demand`, `quantity`

---

## 🌍 4. Market Context Data

**Nguồn**: Vietnam E-Commerce Report 2024, Ministry of Industry and Trade

### 4.1. Vietnam E-Grocery Market Size (Billion USD)

| Year | Market Size (Billion USD) | YoY Growth Rate |
|------|--------------------------|-----------------|
| 2020 | 15.0 | - |
| 2021 | 18.0 | 20.0% |
| 2022 | 20.8 | 15.6% |
| 2023 | 23.5 | 13.0% |
| 2024 | 25.0 | 6.4% |
| 2025 | 30.0 | 20.0% |

### 4.2. Market Growth Analysis

- **CAGR (2020-2025)**: ~14.9%
- **Average YoY Growth**: ~15.0%
- **Projected Market Size (2028)**: ~51.8 Billion USD

### 4.3. Business Opportunity Scores (out of 10)

| Opportunity Area | Score | Level |
|------------------|-------|-------|
| Market Size Growth | 9.2 | 🔥 High |
| Digital Penetration | 8.7 | 🔥 High |
| Consumer Demand | 9.5 | 🔥 High |
| Technology Adoption | 8.3 | ⚡ Medium-High |
| Supply Chain Efficiency | 7.8 | ⚡ Medium-High |

---

## ⏰ 5. Hourly Demand Pattern Statistics

**Nguồn**: Calculated from predictions data (`predictions_test_set.parquet`)

### 5.1. Peak Hours

- **Morning Peak Hours**: 7:00, 8:00, 9:00
- **Evening Peak Hours**: 17:00, 18:00, 19:00, 20:00
- **Peak-to-Off-Peak Ratio**: Calculated dynamically from data

### 5.2. Forecast Accuracy by Hour

- Accuracy calculated per hour using formula: `100 * (1 - MAE / mean_actual)`
- Accuracy ranges from 0-100%
- Color coding:
  - **Green (>80%)**: Excellent accuracy
  - **Yellow (60-80%)**: Good accuracy
  - **Red (<60%)**: Needs improvement

---

## 💰 6. Financial Impact Calculations

**Nguồn**: Derived from Business Impact Metrics

### 6.1. Cost Savings Assumptions

- **Unit Cost**: $10.00 USD per unit
- **Holding Cost Rate**: 20% annual
- **Monthly Volume**: 10,000 units (hypothetical)
- **Implementation Cost**: $50,000 USD (one-time)

### 6.2. Monthly Cost Savings Breakdown

- **Spoilage Reduction**: Calculated from spoilage rate improvement
- **Stockout Prevention**: Calculated from stockout rate improvement
- **Profit Increase**: Calculated from profit margin improvement

### 6.3. ROI Metrics

- **Monthly Savings**: Sum of all cost savings components
- **Annual Savings**: Monthly savings × 12
- **Payback Period**: Implementation cost / Monthly savings (months)
- **Annual ROI**: (Annual savings / Implementation cost) × 100%

---

## 📈 7. Forecast Quality Metrics

**Nguồn**: Calculated from predictions vs actual values

### 7.1. Quality Scores

- **Accuracy**: `1 - (mean_absolute_error / mean_actual)` (clamped 0-1)
- **Precision**: `1 / (1 + mean_interval_width / mean_actual)` (clamped 0-1)
- **Reliability**: Percentage of actual values falling within Q05-Q95 prediction interval

### 7.2. Error Distribution

- **Mean Error**: Average of (actual - forecast)
- **Error Distribution**: Histogram of forecast errors
- **Coverage**: Percentage of actual values within prediction intervals

---

## 📊 8. Performance by Category Metrics

**Nguồn**: Calculated from predictions data grouped by category/product

### 8.1. Metrics per Category

For each category/product:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **Accuracy**: `100 * (1 - MAE / mean_actual)`
- **Data Points Count**: Number of samples in category

### 8.2. Ranking Criteria

- Categories ranked by accuracy (descending)
- Top 10 categories/products by data volume
- Performance color coding:
  - **Green (>80%)**: Excellent
  - **Yellow (60-80%)**: Good
  - **Red (<60%)**: Needs improvement

---

## 📝 Notes

### Data Sources

1. **Model Metrics**: `reports/metrics/model_metrics.json`
2. **Business Impact**: `reports/backtesting/estimated_results.csv`
3. **Feature Importance**: `reports/shap_values/feature_importance.csv`
4. **Predictions**: `reports/predictions_test_set.parquet`
5. **Market Data**: Vietnam E-Commerce Report 2024

### Calculation Methods

- **MAPE**: Only calculated for samples where actual value ≠ 0
- **Accuracy**: Clamped between 0-100% to prevent negative or >100% values
- **Coverage**: Calculated as percentage of actual values within Q05-Q95 interval
- **ROI**: Based on hypothetical cost assumptions (adjustable)

### Last Updated

Generated automatically from chart generation script: `scripts/generate_report_charts.py`

---

## 🔄 How to Update

To update these statistics:

1. Run the chart generation script: `python scripts/generate_report_charts.py`
2. Statistics are automatically extracted from data files
3. This markdown file can be regenerated or manually updated based on the data sources

---

**SmartGrocy E-Grocery Forecasting System**  
*Comprehensive Statistics Documentation*

