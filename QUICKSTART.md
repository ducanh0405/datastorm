# 🚀 Quick Start Guide - DataStorm Refactored Pipeline

**Last Updated:** November 6, 2025  
**Status:** Production-Ready

---

## ⚡ TL;DR (30-Second Setup)

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 2. Create sample data (1% for testing)
python scripts/create_sample_data.py

# 3. Validate setup
python scripts/validate_setup.py

# 4. Run smoke tests
pytest tests/test_smoke.py -v -m smoke

# 5. Run full pipeline
python src/pipelines/_04_run_pipeline.py
```

---

## 📦 What's New in This Refactoring?

### Before (Prototype) → After (Production-Ready)

| Component | Before ❌ | After ✅ |
|-----------|----------|---------|
| **Data Granularity** | Raw transactions (inconsistent) | Aggregated weekly `[PRODUCT_ID, STORE_ID, WEEK_NO]` |
| **Missing Periods** | Ignored (gaps in time series) | Zero-filled complete grid |
| **Train/Test Split** | Random shuffle 🚨 **LEAKAGE** | Time-based (weeks 1-83 vs 84-104) |
| **Lag Features** | Subtle leakage in rollings | 100% leak-safe (on lagged series) |
| **Model Type** | Single regression | 3 quantile models (Q05/Q50/Q95) |
| **Evaluation Metric** | RMSE (wrong for quantile) | Pinball loss (correct) |
| **Tests** | None | 6 smoke tests + validation script |
| **Dev Tooling** | None | ruff, black, isort, mypy, pre-commit |

**Bottom Line:** The refactored pipeline is now **leak-safe, probabilistic, and production-ready**.

---

## 📋 Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **RAM**: 4GB+ recommended (8GB+ for full dataset)
- **Disk Space**: 2GB+ for data and models

---

## 🎯 Step-by-Step Walkthrough

### Step 1: Install Dependencies (2 minutes)

```bash
# Core dependencies (pandas, lightgbm, etc.)
pip install -r requirements.txt

# Dev tools (pytest, ruff, black, mypy)
pip install -r requirements-dev.txt

# Optional: Set up pre-commit hooks for auto-formatting
pre-commit install
```

### Step 2: Create Sample Data (30 seconds)

The full Dunnhumby dataset has 2.6M rows. For quick testing, create a 1% sample:

```bash
python scripts/create_sample_data.py
```

**Output:**
```
Saved transaction_data.csv: 2,567,940 -> 25,679 rows (1.00%)
Saved product.csv: 92,353 -> 924 rows (1.00%)
Saved causal_data.csv: 36,786,524 -> 367,865 rows (1.00%)
...
Sample data created in: data/poc_data/
```

This creates `data/poc_data/` with small samples for fast iteration.

### Step 3: Validate Setup (10 seconds)

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
  ✓ .pre-commit-config.yaml
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

### Step 5: Run Full Pipeline (1-2 minutes on POC data)

Now run the entire pipeline end-to-end:

```bash
python src/pipelines/_04_run_pipeline.py
```

**What Happens:**
1. **Data Loading:** Reads CSVs from `data/poc_data/` (or `data/raw/` if using full dataset)
2. **WS0 (Aggregation):** Aggregates transactions to weekly level, creates master grid
3. **WS1 (Relational):** Joins product and household demographics
4. **WS2 (Time-Series):** Creates leak-safe lag/rolling features
5. **WS4 (Price/Promo):** Adds promotion indicators and price features
6. **Saves:** `data/processed/master_feature_table.parquet`
7. **Training:** Trains Q05/Q50/Q95 quantile models
8. **Saves:** `models/q{05,50,95}_forecaster.joblib` + metrics

**Expected Runtime:**
- POC data (1%): ~60 seconds
- Full data (100%): ~45 minutes

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

### Model Training: Quantile Regression

**What it does:**
1. **Time-based split:** Train on weeks 1-83, test on weeks 84-104
2. **Train 3 models:**
   - Q05: Lower bound (5th percentile)
   - Q50: Median forecast
   - Q95: Upper bound (95th percentile)
3. **Evaluate:** Pinball loss (correct metric for quantile regression)
4. **Save:** 3 model files + feature config + metrics

**Why quantile models?**
- **Probabilistic forecasting:** Get prediction intervals, not just point estimates
- **Inventory optimization:** Safety stock = Q95 - Q50
- **Dynamic pricing:** If current stock > Q05 × days_to_expiry, trigger discount

**File:** `src/pipelines/_03_model_training.py`

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

### 2. Trained Models: `models/q{05,50,95}_forecaster.joblib`

```python
import joblib

model_q50 = joblib.load('models/q50_forecaster.joblib')
print(model_q50)  # LGBMRegressor(objective='quantile', alpha=0.5, ...)
```

### 3. Metrics: `reports/metrics/quantile_model_metrics.json`

```json
{
  "q05_pinball_loss": 12.34,
  "q05_rmse": 45.67,
  "q50_pinball_loss": 18.76,
  "q50_rmse": 38.21,
  "q95_pinball_loss": 14.52,
  "q95_rmse": 51.34,
  "prediction_interval_coverage": 0.897
}
```

**How to interpret:**
- **Pinball loss:** Lower is better (measures quantile accuracy)
- **RMSE:** For reference (Q50 should have lowest RMSE)
- **Coverage:** Should be ~90% for P90 interval (Q05-Q95)
  - If 89.7%, model is well-calibrated ✅
  - If <80% or >95%, model is mis-calibrated ⚠️

---

## 🧪 Testing & Development

### Run Individual Smoke Tests

```bash
# Test just the aggregation
pytest tests/test_smoke.py::test_ws0_aggregation -v

# Test just the time-series features
pytest tests/test_smoke.py::test_ws2_timeseries_features -v

# Test time-based split
pytest tests/test_smoke.py::test_time_based_split -v
```

### Run Linting (Pre-Commit)

```bash
# Auto-format code
pre-commit run --all-files

# Or manually:
ruff check src/ tests/ --fix
black src/ tests/
isort src/ tests/
mypy src/
```

### Debug a Single Stage

```bash
# Run only feature enrichment (skips training)
python src/pipelines/_02_feature_enrichment.py

# Run only model training (requires master_feature_table.parquet to exist)
python src/pipelines/_03_model_training.py
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

Just change the data source in `_01_load_data.py`:

```python
# Edit src/pipelines/_01_load_data.py
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'Dunnhumby'  # Full dataset
# (instead of 'data' / 'poc_data')
```

Or set an environment variable:
```bash
export DATA_SOURCE=full  # Linux/Mac
set DATA_SOURCE=full     # Windows
python src/pipelines/_04_run_pipeline.py
```

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

Before considering the pipeline "working", verify:

- [ ] ✅ Validation script passes: `python scripts/validate_setup.py`
- [ ] ✅ All smoke tests pass: `pytest tests/test_smoke.py -v -m smoke`
- [ ] ✅ Pipeline runs successfully: `python src/pipelines/_04_run_pipeline.py`
- [ ] ✅ 3 model files saved: `models/q{05,50,95}_forecaster.joblib`
- [ ] ✅ Metrics file created: `reports/metrics/quantile_model_metrics.json`
- [ ] ✅ Prediction interval coverage ~90%: Check `prediction_interval_coverage` in metrics

---

**Happy Forecasting! 🚀**
