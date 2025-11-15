# SmartGrocy Project Structure

## 📁 Cấu trúc thư mục

```
E-Grocery_Forecaster/
├── main.py                      # Main entry point (CLI)
├── run_pipeline.py              # Consolidated pipeline runner
├── run_business_modules.py      # Business modules runner
├── run_all_tests.py             # Test runner
│
├── README.md                    # Main documentation
├── PROJECT_STRUCTURE.md         # This file
├── REFACTORING_PLAN.md          # Refactoring documentation
│
├── config/                      # Configuration files
│   └── pipeline_config.json
│
├── data/                        # Data directories
│   ├── 2_raw/                   # Raw input data
│   └── 3_processed/             # Processed data
│
├── docs/                        # Documentation
│   ├── QUICKSTART.md
│   ├── CONTRIBUTING.md
│   ├── CHANGELOG.md
│   ├── TEST_README.md
│   ├── MEMORY_OPTIMIZATION.md
│   ├── OPERATIONS.md
│   └── archive/                 # Archived docs/reports
│
├── src/                         # Source code
│   ├── config.py                # Central configuration
│   ├── features/                # Feature engineering modules
│   │   ├── ws0_aggregation.py
│   │   ├── ws1_relational_features.py
│   │   ├── ws2_timeseries_features.py
│   │   ├── ws3_behavior_features.py
│   │   ├── ws4_price_features.py
│   │   ├── ws5_stockout_recovery.py
│   │   ├── ws6_weather_features.py
│   │   └── feature_selection.py
│   ├── modules/                 # Business logic modules
│   │   ├── inventory_optimization.py
│   │   ├── inventory_backtesting.py
│   │   ├── dynamic_pricing.py
│   │   └── llm_insights.py
│   ├── pipelines/               # Pipeline stages
│   │   ├── _01_load_data.py
│   │   ├── _02_feature_enrichment.py
│   │   ├── _03_model_training.py
│   │   ├── _04_run_pipeline.py
│   │   ├── _05_prediction.py
│   │   ├── _06_ensemble.py
│   │   ├── _07_dashboard.py
│   │   ├── _00_modern_orchestrator.py
│   │   └── _00_modern_orchestrator_v2.py
│   └── utils/                   # Utility modules
│       ├── alerting.py
│       ├── caching.py
│       ├── data_lineage.py
│       ├── data_quality.py
│       ├── data_quality_gx.py
│       ├── parallel_processing.py
│       ├── performance_monitor.py
│       ├── validation.py
│       └── visualization.py
│
├── tests/                       # All tests
│   ├── test_config_import.py
│   ├── test_config_validation.py
│   ├── test_features.py
│   ├── test_integration.py
│   ├── test_modules.py
│   ├── test_pipeline_quick.py
│   ├── test_pipeline_sample.py
│   ├── test_refactoring.py
│   ├── test_smoke.py
│   └── ...
│
├── scripts/                      # Utility scripts
│   ├── setup_great_expectations.py
│   ├── run_data_quality_check.py
│   ├── run_feature_selection.py
│   ├── run_backtesting_analysis.py
│   └── ...
│
├── models/                       # Trained models
│   ├── lightgbm_q05_forecaster.joblib
│   ├── lightgbm_q25_forecaster.joblib
│   ├── lightgbm_q50_forecaster.joblib
│   ├── lightgbm_q75_forecaster.joblib
│   ├── lightgbm_q95_forecaster.joblib
│   └── model_features.json
│
├── reports/                      # Output reports
│   ├── predictions_test_set.csv
│   ├── inventory_recommendations.csv
│   ├── pricing_recommendations.csv
│   ├── metrics/
│   ├── shap_values/
│   └── ...
│
└── logs/                        # Log files
    ├── pipeline.log
    └── alerts/
```

## 🚀 Entry Points

### 1. Main Entry Point (`main.py`)
```bash
# Run pipeline
python main.py pipeline --full-data

# Run business modules
python main.py business

# Run tests
python main.py test
```

### 2. Pipeline Runner (`run_pipeline.py`)
```bash
# Full pipeline
python run_pipeline.py --full-data --use-v2

# With sampling
python run_pipeline.py --full-data --sample 0.1
```

### 3. Business Modules (`run_business_modules.py`)
```bash
# Run all business modules
python run_business_modules.py

# Only inventory
python run_business_modules.py --inventory-only
```

## 📝 Key Files

### Configuration
- `src/config.py` - Central configuration
- `config/pipeline_config.json` - Pipeline settings

### Pipeline Stages
- `src/pipelines/_01_load_data.py` - Data loading
- `src/pipelines/_02_feature_enrichment.py` - Feature engineering
- `src/pipelines/_03_model_training.py` - Model training
- `src/pipelines/_05_prediction.py` - Prediction/forecasting

### Business Modules
- `src/modules/inventory_optimization.py` - Inventory optimization
- `src/modules/dynamic_pricing.py` - Dynamic pricing
- `src/modules/llm_insights.py` - LLM insights

## 🧹 Cleanup Notes

### Removed Files
- Duplicate test files (moved to `tests/`)
- Duplicate model files (`q*.joblib` → kept `lightgbm_*`)
- Duplicate documentation (moved to `docs/archive/`)
- `scripts/demo_modern_pipeline.py` (duplicate)

### Consolidated
- `run_modern_pipeline.py` + `run_modern_pipeline_v2.py` → `run_pipeline.py`
- Test files → All in `tests/` directory

## 📚 Documentation

- `README.md` - Main documentation
- `docs/QUICKSTART.md` - Quick start guide
- `docs/TEST_README.md` - Testing guide
- `docs/MEMORY_OPTIMIZATION.md` - Memory optimization guide
- `docs/OPERATIONS.md` - Operations guide

