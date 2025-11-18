# 🏆 SmartGrocy - E-Grocery Demand Forecasting & Optimization

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![ML](https://img.shields.io/badge/ML-LightGBM-green.svg)](https://lightgbm.readthedocs.io/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](IMPROVEMENTS_SUMMARY.md)

**Production-ready AI solution for demand forecasting, inventory optimization, and dynamic pricing in Vietnamese e-grocery market**

---

## 🎯 Overview

SmartGrocy delivers **5 integrated modules** solving core e-grocery challenges:

1. 📈 **Demand Forecasting** - LightGBM quantile regression (85.68% R²)
2. 📦 **Inventory Optimization** - Statistical models with risk assessment
3. 💰 **Dynamic Pricing** - Profit maximization with 14 optimization metrics
4. 🧠 **LLM Insights** - Risk-based business recommendations (392 insights)
5. 📊 **Visualization** - Interactive charts and dashboards

### ⚡ Key Achievements

| Metric | Value | Impact |
|--------|-------|--------|
| **Forecast R² Score** | 85.68% | Accurate demand planning |
| **Coverage (90%)** | 87.03% | Reliable prediction intervals |
| **Spoilage Reduction** | 40.0% | Cost savings (8.2% → 4.92%) |
| **Stockout Reduction** | 32.5% | Service improvement (7.5% → 5.06%) |
| **Profit Margin Increase** | 37.5% | Revenue optimization (15% → 20.6%) |
| **LLM Insights** | 392 | Actionable intelligence |

**Business Impact**: $290,000+ annual savings | 2-4 months ROI payback

---

## 🚀 Quick Start

### Step 1: Setup Environment (2 minutes)

```bash
# Clone repository
git clone https://github.com/ducanh0405/datastorm.git
cd datastorm

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install streamlit plotly matplotlib seaborn
```

### Step 2: Verify Data & Configuration

```bash
# Check data availability
python scripts/run_data_quality_check.py

# Validate configuration
python run_complete_validation.py
# Expected output: 5/5 tests passed
```

### Step 3: Run Complete Pipeline

#### Option A: Full Pipeline (Recommended for Production)

```bash
# Run end-to-end pipeline (ML + Business Modules)
python run_end_to_end.py --full-data

# This will:
# 1. Load and process data
# 2. Engineer 66 features
# 3. Train LightGBM models (5 quantiles)
# 4. Generate predictions
# 5. Run inventory optimization
# 6. Run dynamic pricing
# 7. Generate LLM insights
```

#### Option B: Quick Test with Sample Data

```bash
# Run with 10% sample (faster, for testing)
python run_end_to_end.py --full-data --sample 0.1
```

#### Option C: Step-by-Step Pipeline

```bash
# run individual steps:
python -m src.pipelines._01_load_data
python -m src.pipelines._02_feature_enrichment
python -m src.pipelines._03_model_training
python -m src.pipelines._05_prediction
python run_business_modules.py --forecasts reports/predictions_test_set.parquet
```

### Step 4: Generate Reports & Charts

```bash
# Generate all 8 professional dashboard charts
python scripts/generate_report_charts.py

# Charts will be saved to: reports/report_charts/
# - chart1_model_performance.png
# - chart2_business_impact.png
# - chart3_forecast_quality.png
# - chart4_feature_importance.png
# - chart5_market_context.png
# - chart6_hourly_demand_pattern.png
# - chart7_profit_margin_improvement.png
# - chart8_performance_by_category.png
```

### Step 5: Launch Interactive Dashboard

```bash
streamlit run dashboard/streamlit_app.py

# Dashboard opens at: http://localhost:8501
# Features:
# - Real-time forecast filtering
# - Interactive drill-down analysis
# - Export capabilities
# - Business metrics visualization
```

### Quick Reference Commands

```bash
# Full pipeline (production)
python run_end_to_end.py --full-data

# Quick test (10% sample)
python run_end_to_end.py --full-data --sample 0.1

# Business modules only
python run_business_modules.py

# Generate charts
python scripts/generate_report_charts.py

# Launch dashboard
streamlit run dashboard/streamlit_app.py

# Run tests
python run_all_tests.py
```

### Expected Output Files

After running the pipeline, you should have:

```
reports/
├── predictions_test_set.parquet      # Forecast predictions
├── model_metrics.json                 # Model performance metrics
├── inventory_recommendations.csv      # Inventory optimization results
├── pricing_recommendations.csv        # Dynamic pricing results
├── llm_insights.csv                   # Business insights
├── business_report_summary.csv        # Business impact summary
├── estimated_results.csv              # Backtesting results
└── report_charts/                    # 8 PNG chart files
    ├── chart1_model_performance.png
    ├── chart2_business_impact.png
    └── ...
```

## 📊 System Architecture

```
FreshRetail-50K Dataset
        ↓
    ┌─────────────────────┐
    │  Data Processing    │ ← Robust imputation (90%+ missing resolved)
    │  & Quality Checks  │ ← Data quality validation
    └─────────────────────┘
            ↓
    ┌─────────────────────┐
    │ Feature Engineering │ ← 66 features across 7 workstreams (WS0-WS6)
    │  (50+ features)     │ ← Lag, rolling, time, categorical features
    └─────────────────────┘
            ↓
    ┌─────────────────────┐
    │   LightGBM Models   │ ← Quantile regression (Q05-Q95)
    │ 5 Quantile Forecasts│ ← R² score and coverage metrics
    └─────────────────────┘
            ↓
    ┌───────────┬───────────┬───────────┬───────────┐
    │           │           │           │           │
Module 1      Module 2    Module 3    Module 4    Analytics
Forecasting   Inventory   Pricing     Insights    Dashboard
(309K preds)  (18 metrics)(14 metrics)(392 ins)  (Interactive)
    │           │           │           │           │
    └───────────┴───────────┴───────────┴───────────┘
                        ↓
            📈 Actionable Business Intelligence
            • Risk-based recommendations
            • Profit optimization
            • Inventory efficiency 
            • Dynamic pricing more suitable price
```

---

## 🎨 Key Features

### Module 1: Forecasting
- LightGBM Quantile Regression (5 quantiles)
- 66 engineered features (lag, rolling, time, categorical)
- SHAP explainability
- 85.68% R² score

### Module 2: Inventory Optimization
- 18 optimization metrics
- Risk categorization (4 levels: LOW/MEDIUM/HIGH/CRITICAL)
- Stockout & overstock risk analysis
- Inventory turnover tracking
- Daily cost analysis

### Module 3: Dynamic Pricing
- 14 optimization metrics
- Revenue & profit impact analysis
- Price elasticity by category
- Priority scoring (4 levels)
- Competitive positioning

### Module 4: LLM Insights
- Risk-based insights generation
- Stockout risk (0-31%) + Overstock risk (5-15%)
- Actionable recommendations with priority levels
- Multi-product analysis
- Confidence scoring

### Module 5: Visualization
- Interactive Streamlit dashboard
- 8 professional chart dashboards
- Jupyter notebook for custom chart generation
- High-resolution exports (300 DPI)

---

## 📁 Project Structure

```
SmartGrocy/
├── src/
│   ├── core/              # Core business logic
│   ├── modules/           # 5 main modules
│   ├── pipelines/         # ML pipeline
│   ├── features/          # Feature engineering (66 features)
│   └── utils/             # Utilities
├── scripts/
│   ├── generate_report_charts.py    # Chart generation
│   ├── run_backtesting_analysis.py  # Backtesting
│   └── analysis/          # Sensitivity analysis
├── dashboard/
│   └── streamlit_app.py   # Interactive dashboard
├── tests/                 # 15+ unit tests
├── reports/
│   ├── report_charts/     # Generated charts (8 PNG files)
│   ├── metrics/           # Model metrics
│   └── backtesting/      # Business impact results
└── docs/                  # Complete documentation
```

---

## 📚 Documentation

### Quick Guides
- 🚀 [Quick Start](docs/guides/QUICK_START_VALIDATION.md)
- 🔄 [Retraining Guide](docs/guides/retraining_guide.md) - Non-tech friendly
- ☁️ [Cloud Deployment](docs/guides/deployment_cloud.md) - GCP/AWS/Azure

### Technical Docs
- 📊 [Technical Report](TECHNICAL_REPORT.md) - Complete system overview
- 📋 [Improvements Summary](IMPROVEMENTS_SUMMARY.md) - Latest status
- 🎨 [Chart Generation Guide](REGENERATE_REPORTS_GUIDE.md)
- 📊 [Statistics Summary](reports/chart_statistics_summary.md)

**Full Index**: [docs/README.md](docs/README.md)

---

## 📈 Performance Metrics

### Model Performance

| Quantile | MAE | RMSE | Pinball Loss |
|----------|-----|------|--------------|
| **Q50 (Median)** | 0.384 | 0.652 | 0.192 |
| Q05 | 0.752 | 1.198 | 0.047 |
| Q95 | 0.762 | 1.114 | 0.061 |

**Overall**: R² = 85.68% | 90% Coverage = 87.03%

### Business KPIs

| Module | Key Metric | Improvement |
|--------|------------|-------------|
| **Inventory** | Stockout reduction | 32.5% |
| **Inventory** | Spoilage reduction | 40.0% |
| **Pricing** | Profit margin | +37.5% |
| **Insights** | Generation rate | 100% (392 insights) |

---

## 🛠️ Tech Stack

**Core ML**: LightGBM 4.5.0 | Pandas 2.3.3 | NumPy | Scikit-learn  
**Visualization**: Streamlit | Plotly | Matplotlib  
**Quality**: Pytest | Black 24.8.0 | Pre-commit

---

## 🔧 Configuration

Key settings in `src/config.py`:

```python
ACTIVE_DATASET = 'freshretail'
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_MIN_MARGIN = 0.15
```

---

## 🚀 Deployment

### Docker

```bash
docker build -t smartgrocy:latest .
docker run -p 8501:8501 smartgrocy:latest
```

### Cloud Options
- **GCP**: Cloud Run + BigQuery ([Guide](docs/guides/deployment_cloud.md#gcp))
- **AWS**: ECS Fargate + RDS ([Guide](docs/guides/deployment_cloud.md#aws))
- **Azure**: Container Instances ([Guide](docs/guides/deployment_cloud.md#azure))

---

## 📊 Charts & Reports

### Generate All Charts

```bash
python scripts/generate_report_charts.py
# Generates 8 professional dashboard charts in reports/report_charts/
```

### Available Charts
1. **Model Performance** - 6-panel dashboard with AI insights
2. **Business Impact** - Spoilage, stockout, profit improvements
3. **Forecast Quality** - Forecast vs actual with intervals
4. **Feature Importance** - SHAP values visualization
5. **Market Context** - Vietnam e-grocery growth
6. **Hourly Demand** - Peak hours and patterns
7. **Profit Margin** - ROI and savings projections
8. **Category Performance** - Performance by product category

### Interactive Notebook

```bash
jupyter notebook reports/charts_notebook.ipynb
# Customize individual charts with interactive controls
```

---

## 🎯 Project Status

| Component | Status | Quality |
|-----------|--------|---------|
| Forecasting | ✅ Stable | High |
| Inventory | ✅ Enhanced | High |
| Pricing | ✅ Enhanced | High |
| Insights | ✅ Complete | High |
| Visualization | ✅ Complete | High |
| Testing | ✅ Comprehensive | High |
| Documentation | ✅ Complete | High |

**Overall: PRODUCTION READY** 🚀

---

## 👥 Team

**Lunous Team**  
Datastorm 2025  
Contact Email: ITDSIU24003@student.hcmiu.edu.vn

---

## 📝 License

MIT License - See [LICENSE](LICENSE)

---

**Last Updated**: 18/11/2025  
**Version**: 4.0.2  
**Status**: Production Ready
