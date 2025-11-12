# 🚀 Quick Start Guide - SmartGrocy

**Last Updated:** November 12, 2025
**Status:** Demo-Ready with Interactive Dashboard & Modern Pipeline
**Default Data:** POC (1% sample) with 100% PRODUCT_ID matching
**Dashboard:** Available at reports/dashboard/index.html

---

## ⚡ TL;DR (30-Second Setup)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Khởi tạo data quality monitoring
python scripts/setup_data_quality.py

# 3. Chạy pipeline hiện đại với monitoring
python run_modern_pipeline.py

# 4. Giám sát chất lượng dữ liệu
python scripts/monitor_data_quality.py

# 5. Tạo dashboard tương tác
python scripts/create_dashboard.py

# Mở dashboard: reports/dashboard/index.html
```

---

## 📦 SmartGrocy Features

### Modern Pipeline với Data Quality Monitoring

| Component | SmartGrocy ✅ |
|-----------|----------------|
| **Pipeline Orchestration** | Prefect-based DAG workflow |
| **Data Quality Monitoring** | Great Expectations + custom validations |
| **Alerting System** | Tự động cảnh báo chất lượng dữ liệu và lỗi pipeline |
| **Intelligent Caching** | Tối ưu hóa hiệu năng với disk-based caching |
| **Performance Monitoring** | Theo dõi bottleneck và tối hóa tài nguyên |
| **Drift Detection** | Phát hiện thay đổi phân phối dữ liệu theo thời gian |
| **Data Granularity** | Aggregated weekly `[PRODUCT_ID, STORE_ID, WEEK_NO]` |
| **Missing Periods** | Zero-filled complete grid |
| **Train/Test Split** | Time-based (weeks 1-83 vs 84-104) - leak-safe |
| **Lag Features** | 100% leak-safe (on lagged series) |
| **Model Type** | 7 quantile models (Q05/Q10/Q25/Q50/Q75/Q90/Q95) |
| **Evaluation Metric** | Pinball loss + Prediction Interval Coverage |
| **Tests** | 6 smoke tests + validation script + comprehensive testing |
| **Dev Tooling** | ruff, black, isort, mypy |
| **Dashboard** | Interactive HTML with Plotly charts |
| **License** | MIT License |

**Bottom Line:** SmartGrocy là **hệ thống dự báo hiện đại với monitoring chất lượng dữ liệu, leak-safe, probabilistic, và production-ready**.

---

## 📋 Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **RAM**: 4GB+ recommended (8GB+ for full dataset)
- **Disk Space**: 2GB+ for data and models

---

## 🎯 Step-by-Step Walkthrough

### Step 1: Install Dependencies (2 minutes)

```bash
# Core dependencies (pandas, lightgbm, prefect, great_expectations, etc.)
pip install -r requirements.txt
```

**Key Dependencies:**
- **Prefect**: Pipeline orchestration và workflow management
- **Great Expectations**: Data quality monitoring và validation
- **LightGBM/XGBoost**: Machine learning models
- **Polars/Pandas**: High-performance data processing
- **Plotly**: Interactive dashboards

### Step 2: Setup Data Quality Monitoring (1 minute)

Khởi tạo hệ thống monitoring chất lượng dữ liệu:

```bash
python scripts/setup_data_quality.py
```

**What happens:**
- Tạo Great Expectations configuration
- Setup data validation rules
- Khởi tạo alerting system
- Tạo data quality checkpoints

### Step 4: Validate Setup (10 seconds)

Run the validation script to ensure everything is configured correctly:

```bash
python scripts/validate_setup.py
```

**Expected Output:**
```
======================================================================
DATASTORM PIPELINE VALIDATION
======================================================================

[1/5] Testing module imports...
  ✓ All modules import successfully

[2/5] Testing directory structure...
  ✓ data\poc_data
  ✓ data\processed
  ✓ models
  ✓ reports\metrics
  ✓ tests

[3/5] Testing POC data...
  ✓ POC transaction data exists: 26,229 rows

[4/5] Testing configuration files...
  ✓ pyproject.toml
  ✓ requirements-dev.txt

[5/5] Testing WS0 aggregation (functional test)...
  ✓ Aggregation successful: 1,000 rows → 957 rows

VALIDATION COMPLETE
```

### Step 4: Run Smoke Tests (30 seconds)

Run the automated test suite to verify core functionality:

```bash
pytest tests/test_smoke.py -v -m smoke
```

**Expected Output:**
```
tests/test_smoke.py::test_data_loader PASSED
tests/test_smoke.py::test_ws0_aggregation PASSED
tests/test_smoke.py::test_ws2_timeseries_features PASSED
tests/test_smoke.py::test_time_based_split PASSED
tests/test_smoke.py::test_quantile_model_config PASSED
tests/test_smoke.py::test_directory_structure PASSED

============== 6 passed in 28.43s ==============
```

### Step 5: Run Modern Pipeline với Monitoring (2-5 minutes)

Chạy pipeline hiện đại với data quality monitoring:

```bash
python run_modern_pipeline.py --full-data
```

**What Happens:**
1. **Data Quality Check:** Validate data quality với Great Expectations
2. **Pipeline Orchestration:** Prefect manages workflow execution
3. **Data Loading:** Reads từ `data/2_raw/` hoặc `data/1_poc_data/`
4. **WS0 (Aggregation):** Aggregates transactions to weekly level, creates master grid
5. **WS1 (Relational):** Joins product và household demographics
6. **WS2 (Time-Series):** Creates leak-safe lag/rolling features
7. **WS4 (Price/Promo):** Adds promotion indicators và price features
8. **WS5-WS6:** Additional features (stockout recovery, weather)
9. **Quality Validation:** Continuous data quality checks
10. **Training:** Trains 7 quantile models (Q05/Q10/Q25/Q50/Q75/Q90/Q95)
11. **Saves:** Models + metrics + quality reports

**Expected Runtime:**
- POC data (1%): ~2 minutes
- Full data (100%): ~45 minutes với monitoring

---

## 🔬 Understanding the Pipeline Stages

### WS0: Aggregation & Master Grid (NEW!)

**What it does:**
```
Raw transactions (multiple rows per product-store-week)
    ↓
Aggregate to [PRODUCT_ID, STORE_ID, WEEK_NO]
    ↓
Create complete grid (all combinations)
    ↓
Zero-fill missing periods
    ↓
Sort by time (CRITICAL for leak-safe features)
```

**Why it matters:**
- **Consistent granularity:** One row per product-store-week
- **No gaps:** Missing weeks filled with 0 sales (correct for sparse retail data)
- **Leak-safe foundation:** Sorted by time enables proper lag calculations

**File:** `src/features/ws0_aggregation.py`

### WS1: Relational Features

**What it does:**
- Joins product metadata (department, commodity)
- Joins household demographics (if available)

**File:** `src/features/ws1_relational_features.py`

### WS2: Time-Series Features (LEAK-SAFE!)

**What it does:**
```python
# Lags (never include current row)
sales_value_lag_1    # t-1 week
sales_value_lag_4    # t-4 weeks (1 month)
sales_value_lag_8    # t-8 weeks (2 months)
sales_value_lag_12   # t-12 weeks (3 months)

# Rolling stats (calculated on lagged series)
rolling_mean_4_lag_1   # Mean of [t-1, t-2, t-3, t-4]
rolling_std_4_lag_1    # Std dev of same window
rolling_mean_8_lag_1   # Mean of [t-1 through t-8]
rolling_std_8_lag_1

# Calendar features
week_of_year  # 1-52
month_proxy   # 1-12
quarter       # 1-4
week_sin/cos  # Cyclical encoding
```

**Why it's leak-safe:**
- Lags start from `t-1` (never `t-0`)
- Rolling windows on `lag_1` column (window = `[t-1, t-2, ..., t-N]`)
- First week of each product-store has NaN for lag_1 (cannot compute)

**File:** `src/features/ws2_timeseries_features.py`

### WS4: Price & Promotion Features

**What it does:**
- Calculates base price, discounts, discount percentages
- Merges causal data (display, mailer, retail promos)
- Creates binary flags: `is_on_display`, `is_on_mailer`, etc.

**File:** `src/features/ws4_price_features.py`

### WS5: Stockout Recovery Features

**What it does:**
- Detects và handles stockout patterns
- Creates recovery indicators sau stockouts
- Models demand recovery behavior

**File:** `src/features/ws5_stockout_recovery.py`

### WS6: Weather Features (Optional)

**What it does:**
- Integrates weather data nếu available
- Creates weather impact features
- Seasonal weather patterns

**File:** `src/features/ws6_weather_features.py`

### Model Training: Enhanced Quantile Regression

**What it does:**
1. **Time-based split:** Train on weeks 1-83, test on weeks 84-104
2. **Train 7 models:**
   - Q05/Q10/Q25: Lower bounds (5th/10th/25th percentiles)
   - Q50: Median forecast (best estimate)
   - Q75/Q90/Q95: Upper bounds (75th/90th/95th percentiles)
3. **Evaluate:** Pinball loss + Prediction Interval Coverage
4. **Ensemble:** Weighted combination của multiple quantiles
5. **Save:** 7 model files + feature config + metrics

**Why 7 quantile models?**
- **Granular probabilistic forecasting:** Chi tiết hơn về uncertainty
- **Better inventory optimization:** Multiple safety stock levels
- **Enhanced dynamic pricing:** Flexible discounting strategies
- **Ensemble methods:** Combine predictions for better accuracy

**Files:** `src/pipelines/_03_model_training.py`, `src/pipelines/_06_ensemble.py`

---

## 📊 Interpreting the Results

After running the pipeline, check these files:

### 1. Feature Table: `data/processed/master_feature_table.parquet`

```python
import pandas as pd

df = pd.read_parquet('data/processed/master_feature_table.parquet')
print(df.shape)  # e.g., (1,234,567, 35) rows × columns
print(df.columns.tolist())  # All features from WS0-WS4
print(df.head())
```

**Key columns:**
- `PRODUCT_ID`, `STORE_ID`, `WEEK_NO` (grouping keys)
- `SALES_VALUE`, `QUANTITY` (aggregated targets)
- `sales_value_lag_1/4/8/12` (time-series lags)
- `rolling_mean/std_4/8/12_lag_1` (rolling features)
- `is_on_display`, `is_on_mailer` (promotion flags)
- `week_of_year`, `month_proxy` (calendar)

### 2. Trained Models: `models/q{05,10,25,50,75,90,95}_forecaster.joblib`

```python
import joblib

# Load 7 quantile models
models = {}
for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:
    models[q] = joblib.load(f'models/q{int(q*100):02d}_forecaster.joblib')

print(models[0.50])  # LGBMRegressor(objective='quantile', alpha=0.5, ...)
```

### 3. Metrics: `reports/metrics/quantile_model_metrics.json`

```json
{
  "q05_pinball_loss": 12.34,
  "q10_pinball_loss": 15.67,
  "q25_pinball_loss": 18.21,
  "q50_pinball_loss": 20.45,
  "q75_pinball_loss": 22.12,
  "q90_pinball_loss": 24.89,
  "q95_pinball_loss": 26.34,
  "prediction_interval_coverage_90": 0.897,
  "prediction_interval_coverage_80": 0.856,
  "ensemble_pinball_loss": 19.23
}
```

**How to interpret:**
- **Pinball loss:** Lower is better (measures quantile accuracy)
- **Coverage:** Should be ~80-90% for intervals
  - 90% interval (Q05-Q95): target ~90%
  - 80% interval (Q10-Q90): target ~80%
- **Ensemble:** Combined prediction quality

---

## 🧪 Testing & Development

### Run Individual Smoke Tests

```bash
# Test data loading
pytest tests/test_smoke.py::test_data_loader -v

# Test aggregation
pytest tests/test_smoke.py::test_ws0_aggregation -v

# Test time-series features
pytest tests/test_smoke.py::test_ws2_timeseries_features -v

# Test time-based split
pytest tests/test_smoke.py::test_time_based_split -v
```

### Run Linting

```bash
# Manual code formatting and linting
ruff check src/ tests/ --fix
black src/ tests/
isort src/ tests/
mypy src/
```

### Debug a Single Stage

```bash
# Run modern pipeline với monitoring
python run_modern_pipeline.py

# Hoặc run từng stage riêng lẻ:
# Data loading
python -c "from src.pipelines._01_load_data import load_data; dataframes, config = load_data(); print('Data loaded')"

# Feature enrichment
python -c "from src.pipelines._02_feature_enrichment import main; main()"

# Model training
python -c "from src.pipelines._03_model_training import main; main()"

# Ensemble predictions
python -c "from src.pipelines._06_ensemble import main; main()"
```

---

## 🎓 Common Questions

### Q: How do I use the trained models for forecasting?

```python
import joblib
import pandas as pd

# Load models
model_q05 = joblib.load('models/q05_forecaster.joblib')
model_q50 = joblib.load('models/q50_forecaster.joblib')
model_q95 = joblib.load('models/q95_forecaster.joblib')

# Load new data (must have same features as training data)
X_new = pd.read_parquet('data/processed/master_feature_table.parquet')
X_future = X_new[X_new['WEEK_NO'] > 104]  # Weeks beyond training data

# Make predictions
forecast_q05 = model_q05.predict(X_future)  # Lower bound
forecast_q50 = model_q50.predict(X_future)  # Best estimate
forecast_q95 = model_q95.predict(X_future)  # Upper bound

# Calculate safety stock for inventory optimization
safety_stock = forecast_q95 - forecast_q50
```

### Q: How do I run on the full dataset (not POC)?

Modern pipeline tự động detect và sử dụng full dataset:

```bash
# Chạy trên full dataset từ data/2_raw/
python run_modern_pipeline.py --full-data

# Hoặc chỉ định rõ:
export DATA_SOURCE=full
python run_modern_pipeline.py --full-data
```

**Note:** Full dataset requires 16GB+ RAM và ~45 minutes runtime với monitoring.

### Q: What if I get import errors?

Make sure you're running from the project root:
```bash
cd c:\Users\Admin\.vscode\datastorm
python src/pipelines/_04_run_pipeline.py
```

If still failing, add project to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### Q: How do I know if there's data leakage?

Run the smoke test:
```bash
pytest tests/test_smoke.py::test_time_based_split -v
pytest tests/test_smoke.py::test_ws2_timeseries_features -v
```

Both tests explicitly check for leakage:
- Time split: `assert train['WEEK_NO'].max() < test['WEEK_NO'].min()`
- Lag features: `assert first_weeks['sales_value_lag_1'].isna().all()`

---

## 📚 Further Reading

- **QA Fix Log:** `reports/QA_FIXLOG.md` - Detailed root cause analysis of all issues fixed
- **Refactoring Summary:** `reports/REFACTORING_SUMMARY.md` - Executive summary
- **Original README:** `README.md` - Business context and project overview

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: transaction_data.csv` | Run `python scripts/create_sample_data.py` first |
| `Import errors` | Ensure you're in project root: `cd c:\Users\Admin\.vscode\datastorm` |
| `Pytest not found` | Install dev dependencies: `pip install -r requirements-dev.txt` |
| `Models not saved` | Check `models/` directory exists (created by validation script) |
| `Slow pipeline` | Use POC data (1%) for testing: `data/poc_data/` instead of `data/raw/` |
| `Pre-commit failing` | Run manually: `ruff check --fix; black .` |

---

## ✅ Success Checklist

Before considering SmartGrocy "working", verify:

- [ ] ✅ Data quality setup: `python scripts/setup_data_quality.py`
- [ ] ✅ Validation script passes: `python scripts/validate_setup.py`
- [ ] ✅ All smoke tests pass: `pytest tests/test_smoke.py -v -m smoke`
- [ ] ✅ Modern pipeline runs: `python run_modern_pipeline.py --full-data`
- [ ] ✅ 7 model files saved: `models/q{05,10,25,50,75,90,95}_forecaster.joblib`
- [ ] ✅ Metrics file created: `reports/metrics/quantile_model_metrics.json`
- [ ] ✅ Ensemble predictions: `reports/ensemble_predictions.csv`
- [ ] ✅ Dashboard created: `reports/dashboard/index.html`
- [ ] ✅ Data quality reports: Check alerting logs

---

## 📚 Documentation

For detailed information about SmartGrocy:

- **[OPERATIONS.md](./OPERATIONS.md)**: Hướng dẫn deployment và vận hành production
- **[CONTRIBUTING.md](./CONTRIBUTING.md)**: Hướng dẫn đóng góp cho dự án
- **[CHANGELOG.md](./CHANGELOG.md)**: Lịch sử thay đổi và updates
- **[TEST_README.md](./TEST_README.md)**: Tài liệu về testing và validation

---

**Happy Forecasting! 🚀**
