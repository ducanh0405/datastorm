# MODEL RETRAINING GUIDE
**For Non-Technical Users**

## WHEN TO RETRAIN

Retrain your model when:
- ✅ New sales data available (monthly recommended)
- ✅ Significant market changes (promotions, holidays)
- ✅ Model performance drops (>10% error increase)
- ✅ New products added to inventory
- ✅ Business rules change (new pricing strategy)

## PREPARATION CHECKLIST

Before retraining:
- [ ] Backup current model (`models/model.pkl` → `models/model_backup_YYYYMMDD.pkl`)
- [ ] Backup current reports (`reports/` → `reports_backup_YYYYMMDD/`)
- [ ] Prepare new data file (CSV format)
- [ ] Verify data quality (no missing product IDs, valid dates)
- [ ] Allocate 30-60 minutes for process

---

## STEP-BY-STEP RETRAIN PROCESS

### Step 1: Backup Current System (5 min)

```bash
# 1.1. Backup model
cp models/model.pkl models/model_backup_$(date +%Y%m%d).pkl

# 1.2. Backup reports
cp -r reports reports_backup_$(date +%Y%m%d)

echo "✅ Backup complete"
```

### Step 2: Prepare New Data (10 min)

**2.1. Data Requirements:**
- File format: CSV
- Required columns:
  - `date` (YYYY-MM-DD format)
  - `product_id`
  - `sales_quantity`
  - `price`
  - `store_id` (optional)

**2.2. Place file in correct location:**
```bash
# Copy new data to data directory
cp ~/Downloads/new_sales_data.csv data/raw/

echo "✅ Data copied"
```

**2.3. Validate data:**
```bash
# Quick validation
python scripts/validation/validate_input_data.py data/raw/new_sales_data.csv

# Expected output:
# ✅ Date format valid
# ✅ No missing product IDs
# ✅ Price values reasonable
# ✅ Data ready for processing
```

### Step 3: Run Retraining (20-30 min)

```bash
# 3.1. Activate environment
source venv/bin/activate  # or: conda activate smartgrocy

# 3.2. Run retraining pipeline
python scripts/pipeline/run_complete_retrain.py \
  --data data/raw/new_sales_data.csv \
  --output models/model_new.pkl

# Monitor output:
# [1/5] Loading data...       ✅
# [2/5] Feature engineering... ✅
# [3/5] Training model...      ✅ (takes 10-15 min)
# [4/5] Validation...          ✅
# [5/5] Saving model...        ✅
```

### Step 4: Validate New Model (5 min)

```bash
# 4.1. Compare performance
python scripts/validation/compare_models.py \
  --old models/model.pkl \
  --new models/model_new.pkl

# Expected output:
#               Old Model    New Model    Change
# MAE           0.384        0.350        ✅ -8.9%
# RMSE          0.653        0.621        ✅ -4.9%
# R²            0.891        0.903        ✅ +1.3%

# 4.2. Generate test predictions
python scripts/validation/test_predictions.py \
  --model models/model_new.pkl \
  --output reports/test_predictions.csv

echo "✅ Validation complete"
```

### Step 5: Deploy New Model (2 min)

**If validation passes:**
```bash
# 5.1. Replace old model
mv models/model.pkl models/model_old.pkl
mv models/model_new.pkl models/model.pkl

echo "✅ New model deployed"

# 5.2. Regenerate reports
python scripts/reporting/generate_all_reports.py

echo "✅ Reports updated"
```

**If validation fails:**
```bash
# Rollback to backup
rm models/model_new.pkl
echo "⚠️ New model rejected, keeping old model"
```

### Step 6: Verify System (3 min)

```bash
# 6.1. Test end-to-end
python run_complete_validation.py

# Expected: All tests PASS

# 6.2. Check reports
ls -lh reports/
# Should see updated timestamps

echo "✅ System verified"
```

---

## AUTOMATED RETRAIN (Scheduled)

### Weekly Auto-Retrain

**Setup cron job:**
```bash
# Edit crontab
crontab -e

# Add weekly retrain (every Sunday at 2 AM)
0 2 * * 0 cd /path/to/datastorm && ./scripts/automated_retrain.sh
```

**Script: `scripts/automated_retrain.sh`**
```bash
#!/bin/bash
set -e

# Backup
cp models/model.pkl models/model_backup_$(date +%Y%m%d).pkl

# Retrain
python scripts/pipeline/run_complete_retrain.py \
  --data data/raw/latest.csv \
  --output models/model_new.pkl

# Validate
python scripts/validation/compare_models.py \
  --old models/model.pkl \
  --new models/model_new.pkl \
  --auto-deploy

# Send notification
python scripts/utils/send_notification.py \
  --message "Weekly retrain complete"
```

---

## TROUBLESHOOTING

### Error: "Data validation failed"

**Cause:** Missing or invalid data

**Solution:**
```bash
# Check data quality
python scripts/validation/validate_input_data.py \
  data/raw/new_sales_data.csv \
  --verbose

# Fix common issues:
# - Remove duplicate rows
# - Fill missing dates
# - Correct data types
```

### Error: "Model training failed"

**Cause:** Insufficient data or feature engineering issues

**Solution:**
```bash
# Check data size
python -c "import pandas as pd; df = pd.read_csv('data/raw/new_sales_data.csv'); print(f'Rows: {len(df)}')"

# Minimum: 10,000 rows recommended

# If too small, use more historical data
```

### Error: "New model performs worse"

**Cause:** Data quality issues or concept drift

**Solution:**
```bash
# Keep old model
rm models/model_new.pkl

# Investigate data
python scripts/analysis/analyze_data_drift.py

# May need to:
# 1. Add more features
# 2. Adjust hyperparameters
# 3. Use longer training period
```

---

## PERFORMANCE MONITORING

### Daily Checks

```bash
# Check prediction quality
python scripts/monitoring/daily_check.py

# Output:
# ✅ Predictions within expected range
# ✅ No anomalies detected
# ⚠️ Warning: 3 products with high error
```

### Weekly Reports

```bash
# Generate performance report
python scripts/monitoring/weekly_report.py

# Output: reports/weekly_performance.pdf
# - Forecast accuracy trends
# - Feature importance changes
# - Model drift indicators
```

---

## EMERGENCY ROLLBACK

**If system breaks after update:**

```bash
# 1. Stop all running processes
pkill -f "python scripts"

# 2. Restore backup model
cp models/model_backup_YYYYMMDD.pkl models/model.pkl

# 3. Restore backup reports
rm -rf reports
cp -r reports_backup_YYYYMMDD reports

# 4. Verify system
python run_complete_validation.py

echo "✅ Rollback complete"
```

---

## GETTING HELP

**Support Contacts:**
- Technical Team: tech@smartgrocy.com
- Documentation: https://docs.smartgrocy.com
- Emergency Hotline: +84-xxx-xxx-xxxx

**Log Files:**
- Training logs: `logs/training_YYYYMMDD.log`
- Error logs: `logs/errors.log`
- System logs: `logs/system.log`

**Include in support request:**
1. Error message (copy from terminal)
2. Log file (`logs/errors.log`)
3. Data summary (row count, date range)
4. Steps you attempted
