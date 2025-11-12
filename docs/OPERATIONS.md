# 🚀 Operations & Deployment Guide - SmartGrocy

This guide covers operational aspects of deploying and maintaining the SmartGrocy system in production environments.

## 📋 Table of Contents
- [System Requirements](#system-requirements)
- [Local Deployment](#local-deployment)
- [Production Deployment](#production-deployment)
- [Model Serving](#model-serving)
- [Monitoring & Alerting](#monitoring--alerting)
- [Performance Optimization](#performance-optimization)
- [Backup & Recovery](#backup--recovery)
- [Troubleshooting](#troubleshooting)
- [Maintenance Schedule](#maintenance-schedule)

## 🖥️ System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **RAM:** 8GB minimum, 16GB recommended
- **Disk:** 50GB free space
- **Python:** 3.10, 3.11, or 3.12

### Production Requirements
- **RAM:** 32GB+ for full dataset processing
- **CPU:** 8+ cores for parallel processing
- **Storage:** 500GB+ SSD for data and models
- **Network:** Stable internet for data downloads

## 🏠 Local Deployment

### Quick Start (Development)
```bash
# Clone repository
git clone https://github.com/ducanh0405/datastorm.git
cd SmartGrocy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup data quality monitoring
python scripts/setup_data_quality.py

# Run modern pipeline with monitoring
python run_modern_pipeline.py --full-data

# Monitor data quality
python scripts/monitor_data_quality.py

# Generate dashboard
python scripts/create_dashboard.py
```

### Development Environment Setup
```bash
# Install development tools
pip install -r requirements-dev.txt

# Run code quality checks
ruff check src/ tests/ --fix
black src/ tests/
isort src/ tests/
mypy src/

# Run tests
pytest tests/test_smoke.py -v -m smoke
python test_project_comprehensive.py --no-end-to-end
```

## 🏭 Production Deployment

### 1. Environment Setup
```bash
# Create production environment
python -m venv prod_env
source prod_env/bin/activate

# Install production dependencies only
pip install -r requirements.txt

# Optional: Install monitoring tools
pip install prometheus-client
pip install datadog  # If using DataDog monitoring
```

### 2. Data Preparation
```bash
# Use full dataset instead of POC data
# Set environment variable to force full data
export DATA_SOURCE=full  # Linux/Mac
# or
$env:DATA_SOURCE="full"   # Windows PowerShell

# Create necessary directories
mkdir -p data/2_raw
mkdir -p models
mkdir -p reports/metrics
mkdir -p reports/dashboard

# Place your data files in data/2_raw/
# Required files: transaction_data.csv, product.csv, etc.
```

### 3. Initial Pipeline Run
```bash
# Setup data quality monitoring for production
python scripts/setup_data_quality.py

# Run modern pipeline with full monitoring
python run_modern_pipeline.py --full-data

# This will:
# - Validate data quality with Great Expectations
# - Orchestrate pipeline with Prefect
# - Load data from data/2_raw/
# - Process 2.6M+ transactions with quality checks
# - Train 7 quantile models (may take 45-60 minutes)
# - Generate feature tables, models, and quality reports
```

### 4. Model Validation
```bash
# Run predictions on test set
python -c "from src.pipelines._05_prediction import main; main()"

# Generate ensemble predictions
python -c "from src.pipelines._06_ensemble import main; main()"

# Check model metrics
cat reports/metrics/quantile_model_metrics.json

# Expected metrics:
# - Q50 Pinball Loss: < 0.05
# - Prediction Interval Coverage (80%): ~80%
# - Prediction Interval Coverage (90%): ~90%
# - Ensemble Pinball Loss: < 0.06
```

## 🌐 Model Serving

### Basic Prediction API
```python
from src.pipelines._05_prediction import QuantileForecaster

# Load trained models
forecaster = QuantileForecaster()

# Make predictions
result = forecaster.predict_single(
    product_id="P123",
    store_id="S456",
    week_no=100,
    features={
        'sales_value_lag_1': 50.0,
        'rolling_mean_4_lag_1': 45.0,
        'week_of_year': 15,
        # ... other features
    }
)

print(f"Q50 Forecast: {result['forecast_q50']:.2f}")
print(f"Prediction Interval: {result['forecast_q05']:.2f} - {result['forecast_q95']:.2f}")
```

### Flask/FastAPI Serving (Recommended for Production)
```python
# app.py
from flask import Flask, request, jsonify
from src.pipelines._05_prediction import QuantileForecaster

app = Flask(__name__)
forecaster = QuantileForecaster()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = forecaster.predict_single(**data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 📊 Monitoring & Alerting

### Key Metrics to Monitor

#### Model Performance
- **Prediction Accuracy**: Q50 RMSE and Pinball Loss
- **Coverage Rate**: Prediction interval coverage (target: 80-95%)
- **Feature Drift**: Monitor feature distributions over time

#### System Performance
- **Memory Usage**: Track RAM consumption during training/inference
- **CPU Usage**: Monitor parallel processing efficiency
- **Disk I/O**: Track data loading and saving operations

#### Business Metrics
- **Inventory Turnover**: Monitor actual vs predicted inventory levels
- **Stockout Rate**: Track out-of-stock incidents
- **Waste Reduction**: Monitor spoilage reduction

### Logging Setup
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### Alert Configuration
```python
# Alert thresholds
ALERT_THRESHOLDS = {
    'q50_pinball_loss': 0.10,  # Alert if > 0.10
    'coverage_rate': 0.75,     # Alert if < 75%
    'memory_usage_gb': 28,     # Alert if > 28GB
}

def check_alerts(metrics):
    alerts = []
    if metrics.get('q50_pinball_loss', 0) > ALERT_THRESHOLDS['q50_pinball_loss']:
        alerts.append("High prediction error detected")
    if metrics.get('prediction_interval_coverage', 1) < ALERT_THRESHOLDS['coverage_rate']:
        alerts.append("Low coverage rate detected")
    return alerts
```

## ⚡ Performance Optimization

### Memory Optimization
```python
# Use Polars for large datasets (automatically enabled)
import polars as pl

# Enable lazy evaluation for very large datasets
PERFORMANCE_CONFIG = {
    'use_polars': True,
    'lazy_evaluation': True,
    'memory_limit_gb': 16,
    'parallel_threads': -1,
}
```

### Training Optimization
```python
# LightGBM parameters for production
PRODUCTION_LGBM_PARAMS = {
    'n_estimators': 1000,      # More trees for better accuracy
    'learning_rate': 0.05,     # Conservative learning rate
    'num_leaves': 63,          # Balanced complexity
    'max_depth': -1,           # No depth limit
    'colsample_bytree': 0.8,   # Feature sampling
    'subsample': 0.8,          # Row sampling
    'n_jobs': -1,              # Use all cores
}
```

### Inference Optimization
```python
# Batch predictions for better throughput
def batch_predict(forecaster, batch_data):
    """Process multiple predictions efficiently"""
    results = []
    batch_size = 1000

    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i+batch_size]
        batch_results = forecaster.predict(batch)
        results.extend(batch_results)

    return results
```

## 💾 Backup & Recovery

### Data Backup Strategy
```bash
# Create backup script
#!/bin/bash
BACKUP_DIR="/path/to/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup models
tar -czf $BACKUP_DIR/models_$TIMESTAMP.tar.gz models/

# Backup feature tables
tar -czf $BACKUP_DIR/features_$TIMESTAMP.tar.gz data/3_processed/

# Backup configuration
cp src/config.py $BACKUP_DIR/config_$TIMESTAMP.py

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Model Versioning
```python
import joblib
import os
from datetime import datetime

def save_model_version(model, model_name, metrics):
    """Save model with version and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"{model_name}_{timestamp}"

    # Save model
    model_path = f"models/{version}.joblib"
    joblib.dump(model, model_path)

    # Save metadata
    metadata = {
        'version': version,
        'timestamp': timestamp,
        'metrics': metrics,
        'model_path': model_path
    }

    with open(f"models/{version}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return version
```

### Recovery Procedures
```python
def load_latest_model(model_name):
    """Load the most recent model version"""
    model_files = [f for f in os.listdir('models/') if f.startswith(model_name) and f.endswith('.joblib')]
    if not model_files:
        raise FileNotFoundError(f"No {model_name} models found")

    latest_model = max(model_files, key=lambda x: os.path.getctime(f"models/{x}"))
    return joblib.load(f"models/{latest_model}")
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Memory Issues
**Problem:** `MemoryError` during data processing
```python
# Solution: Enable chunked processing
PERFORMANCE_CONFIG = {
    'chunk_size_mb': 50,  # Reduce chunk size
    'lazy_evaluation': True,
    'use_polars': True,  # Polars is more memory efficient
}
```

#### Slow Training
**Problem:** Model training takes too long
```python
# Solution: Optimize LightGBM parameters
FAST_LGBM_PARAMS = {
    'n_estimators': 100,      # Reduce trees
    'learning_rate': 0.1,     # Higher learning rate
    'num_leaves': 31,         # Reduce complexity
    'max_depth': 6,           # Limit depth
}
```

#### Data Quality Issues
**Problem:** Missing or invalid PRODUCT_ID relationships
```bash
# Solution: Recreate POC data with proper relationships
python recreate_poc_data.py
```

#### Model Performance Degradation
**Problem:** Predictions becoming less accurate over time
```python
# Solution: Check for concept drift
def detect_drift(new_data, reference_data):
    """Monitor feature distribution changes"""
    for col in numeric_features:
        ref_mean = reference_data[col].mean()
        new_mean = new_data[col].mean()
        if abs(new_mean - ref_mean) / ref_mean > 0.2:  # 20% change
            logger.warning(f"Drift detected in {col}")
```

### Debug Commands
```bash
# Check data integrity
python -c "import pandas as pd; df = pd.read_parquet('data/3_processed/master_feature_table.parquet'); print(df.shape); print(df.dtypes)"

# Validate models
python -c "import joblib; model = joblib.load('models/q50_forecaster.joblib'); print('Model loaded successfully')"

# Test predictions
python src/pipelines/_05_prediction.py
```

## 📅 Maintenance Schedule

### Daily Tasks
- [ ] Monitor prediction accuracy metrics
- [ ] Check system resource usage
- [ ] Review error logs
- [ ] Validate data pipeline health

### Weekly Tasks
- [ ] Run comprehensive test suite
- [ ] Update dependencies (security patches)
- [ ] Review model performance trends
- [ ] Clean temporary files and logs

### Monthly Tasks
- [ ] Retrain models with new data
- [ ] Performance benchmarking
- [ ] Backup validation
- [ ] Documentation updates

### Quarterly Tasks
- [ ] Major version updates
- [ ] Security audits
- [ ] Architecture reviews
- [ ] Stakeholder reporting

## 📞 Support & Escalation

### Emergency Contacts
- **Critical Issues:** Immediate response required (< 1 hour)
- **High Priority:** Response within 4 hours
- **Normal Issues:** Response within 24 hours

### Escalation Path
1. **Developer Support**: Check logs and run diagnostics
2. **System Admin**: For infrastructure issues
3. **Data Science Team**: For model performance issues
4. **Management**: For business impact issues

---

## 🎯 Production Readiness Checklist

- [ ] Environment setup completed
- [ ] Full dataset processing tested
- [ ] Model performance validated
- [ ] Monitoring and alerting configured
- [ ] Backup procedures documented
- [ ] Recovery procedures tested
- [ ] Team trained on operations
- [ ] Runbook documentation complete
- [ ] Emergency contacts documented

**🚀 System ready for production deployment!**
