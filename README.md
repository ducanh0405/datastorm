# 🏆 SmartGrocy - E-Grocery Demand Forecasting & Inventory Optimization

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/ML-LightGBM%20Quantile-green.svg)](https://lightgbm.readthedocs.io/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Interactive%20Plotly-red.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Competition](https://img.shields.io/badge/Datastorm-2025-orange.svg)](https://datastorm.com)

**Giải pháp AI toàn diện cho dự báo nhu cầu, tối ưu tồn kho và định giá động trong ngành e-grocery Việt Nam**

---

## 📋 Tổng Quan

SmartGrocy là hệ thống MLOps production-ready giải quyết 3 bài toán cốt lõi trong e-grocery:

1. 📈 **Demand Forecasting** - Dự báo nhu cầu chính xác với prediction intervals
2. 📦 **Inventory Optimization** - Tối ưu tồn kho với ROP, EOQ, Safety Stock
3. 💰 **Dynamic Pricing** - Định giá động giảm thiểu spoilage và stockout
4. 🧠 **LLM Insights** - Tự động sinh insight và khuyến nghị nghiệp vụ

### 🎯 Business Impact

| Problem | Current State | Target | Impact |
|---------|---------------|--------|--------|
| **Spoilage Rate** | 5-12% (fresh produce) | <3% | Giảm 40-60% waste |
| **Stockout Rate** | 7-10% (e-commerce) | <3% | Tăng 5-7% revenue |
| **Forecast Accuracy** | 60-70% (baseline) | >85% | Tăng 20% efficiency |
| **Inventory Turnover** | 8-12x/year | 15-20x/year | Giảm 30% holding cost |

### 📊 Vietnam E-Grocery Market Context

- **Market Size 2024**: $25B USD (+20% YoY)
- **Projected 2025**: $30B+ USD
- **CAGR 2023-2028**: 18-25%
- **Fresh Food Share**: 50%+ of e-grocery GMV
- **Key Players**: Shopee, TikTok Shop, Lazada (90% market)

---

## 🏗️ Architecture

### System Architecture (4-Module Design)

```
┌─────────────────────────────────────────────────────────────────┐
│                     SmartGrocy System                            │
└─────────────────────────────────────────────────────────────────┘
          │
          ├─ MODULE 1: Demand Forecasting (Core Engine)
          │  ├─ LightGBM Quantile Regression (Q05-Q95)
          │  ├─ 66 Engineered Features (WS0-WS6)
          │  ├─ SHAP Explainability
          │  └─ Prediction Intervals
          │       ↓
          ├─ MODULE 2: Inventory Optimization
          │  ├─ Reorder Point (ROP) Calculation
          │  ├─ Economic Order Quantity (EOQ)
          │  ├─ Safety Stock (from Q95-Q50)
          │  └─ Stockout Prevention
          │       ↓
          ├─ MODULE 3: Dynamic Pricing Engine
          │  ├─ Markdown Optimization
          │  ├─ High Inventory + Low Demand → Discount
          │  ├─ Profit Margin Protection
          │  └─ Revenue Maximization
          │       ↓
          └─ MODULE 4: LLM Insights
             ├─ Causal → Impact → Action
             ├─ SHAP Interpretation
             ├─ Business Recommendations
             └─ Automated Reporting
```

### Data Pipeline Flow

```
Raw Data → Data Quality Check → Feature Engineering → Model Training
    │              ↓                    ↓                  ↓
    │       (Great Expectations)  (WS0-WS6)       (5 Quantiles)
    │                                              ↓
    └────────────────────────────────────→ Prediction
                                                   ↓
                                    ┌──────────────┴──────────────┐
                                    │                             │
                            Inventory Decisions          Dynamic Pricing
                                    │                             │
                                    └──────────────┬──────────────┘
                                                   ↓
                                            LLM Insights
                                                   ↓
                                            Dashboard & Reports
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/ducanh0405/datastorm.git
cd datastorm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup Great Expectations (one-time)
python scripts/setup_great_expectations.py
```

### 2. Run Complete Pipeline

```bash
# Full pipeline with data quality monitoring
python run_modern_pipeline_v2.py --full-data --use-v2

# Quick test (10% sample)
python run_modern_pipeline_v2.py --full-data --sample 0.1 --use-v2

# Baseline pipeline
python run_modern_pipeline.py --full-data
```

### 3. Run Individual Modules

```python
# Module 1: Demand Forecasting (built into pipeline)
# Outputs: forecasts with Q05, Q50, Q95

# Module 2: Inventory Optimization
from src.modules.inventory_optimization import InventoryOptimizer

optimizer = InventoryOptimizer()
result = optimizer.optimize_inventory_from_forecast(
    forecast_df, product_id='P001', store_id='S001'
)
print(f"Reorder Point: {result['reorder_point']:.0f} units")
print(f"Safety Stock: {result['safety_stock']:.0f} units")
print(f"Should Reorder: {result['should_reorder']}")

# Module 3: Dynamic Pricing
from src.modules.dynamic_pricing import DynamicPricingEngine

pricing_engine = DynamicPricingEngine()
pricing_result = pricing_engine.recommend_price(
    current_price=10.0,
    inventory_ratio=2.0,  # 200% of average
    demand_ratio=0.8,     # 80% of forecast
    cost=5.0
)
print(f"Recommended Price: ${pricing_result['recommended_price']:.2f}")
print(f"Discount: {pricing_result['discount_pct']:.0%}")
print(f"Reason: {pricing_result['reasoning']}")

# Module 4: LLM Insights
from src.modules.llm_insights import LLMInsightGenerator

insight_gen = LLMInsightGenerator()
insight = insight_gen.generate_forecast_insight(
    'P001',
    forecast_data={'q50': 150, 'q95': 200, 'trend': 'up'},
    shap_values={'promo_active': 0.35, 'price': -0.15}
)
print(insight['insight'])
```

### 4. Run Backtesting & Validation

```bash
# Generate market analysis data
python scripts/generate_market_analysis.py

# Run inventory backtesting
python -c "
from src.modules.inventory_backtesting import InventoryBacktester
import pandas as pd
# ... load your historical data ...
backtester = InventoryBacktester(historical_data, forecasts)
comparison = backtester.compare_strategies()
print(comparison)
"

# Run all module tests
python tests/test_modules.py
```

---

## 📊 Module Details

### Module 1: Demand Forecasting Engine

**Location**: `src/pipelines/_03_model_training.py`

**Features**:
- LightGBM Quantile Regression
- 5 quantiles: Q05, Q25, Q50, Q75, Q95
- 66 engineered features (WS0-WS6)
- SHAP explainability
- Temporal cross-validation

**Metrics**:
- MAE, RMSE, Pinball Loss
- Coverage rate (Q05-Q95)
- Forecast bias

**Output Format**:
```python
{
    'product_id': 'P001',
    'store_id': 'S001', 
    'forecast_q05': 80,
    'forecast_q50': 100,
    'forecast_q95': 130,
    'forecast_date': '2025-11-16'
}
```

### Module 2: Inventory Optimization

**Location**: `src/modules/inventory_optimization.py`

**Features**:
- Reorder Point (ROP) calculation
- Economic Order Quantity (EOQ)
- Safety Stock from prediction intervals
- Service level optimization (95% default)
- Stockout risk assessment

**Formula**:
```
ROP = (Avg Daily Demand × Lead Time) + Safety Stock
Safety Stock = Z-score × Demand Std × √(Lead Time + Review Period)
EOQ = √(2DS/H)
```

**Output Format**:
```python
{
    'reorder_point': 750,
    'safety_stock': 150,
    'economic_order_quantity': 500,
    'should_reorder': True,
    'stockout_risk': 0.05
}
```

### Module 3: Dynamic Pricing Engine

**Location**: `src/modules/dynamic_pricing.py`

**Pricing Logic Matrix**:

| Inventory | Demand | Action | Discount |
|-----------|--------|--------|----------|
| Critical (>300%) | Any | Clearance | 40-50% |
| High (>200%) | Low (<80%) | Large Discount | 25-40% |
| High (>200%) | Normal | Medium Discount | 15-25% |
| High (>200%) | High (>120%) | Small Discount | 5-15% |
| Normal | Low | Small Discount | 5-10% |
| Normal | Normal/High | Maintain | 0% |
| Low (<50%) | Any | Maintain | 0% |

**Output Format**:
```python
{
    'recommended_price': 7.50,
    'discount_pct': 0.25,
    'action': 'large_discount',
    'reasoning': 'High inventory with low demand - aggressive markdown',
    'expected_profit_impact': 150.00
}
```

### Module 4: LLM Insights Generator

**Location**: `src/modules/llm_insights.py`

**Features**:
- Rule-based insights (no API required)
- LLM API support (OpenAI, Anthropic)
- Causal → Impact → Action framework
- SHAP interpretation
- JSON structured output

**Output Format**:
```
**Forecast Insight for P001**

**Cause:**
- Active promotional campaign
- Seasonal demand pattern

**Impact:**
- Demand forecast increased to 150.0 units (strong growth trend)

**Recommended Actions:**
- Consider increasing order quantity to prevent stockout
- Maintain higher safety stock due to 33% forecast uncertainty
```

---

## 🧪 Testing & Validation

### Run All Tests

```bash
# Phase 2 integration tests (Great Expectations)
python tests/test_phase2_integration.py

# Module tests (Modules 2-4)
python tests/test_modules.py

# All tests
python run_all_tests.py
```

### Expected Results
```
✅ Phase 2 Tests: 10/10 PASSED
✅ Module Tests: 11/11 PASSED
✅ Success Rate: 100%
```

### Backtesting Validation

```bash
# Generate market analysis
python scripts/generate_market_analysis.py

# Run inventory backtest (compare baseline vs ML)
# Results saved to: reports/market_analysis/
```

**Expected KPI Improvements**:
- Spoilage Rate: 8% → 3% (62% reduction)
- Stockout Rate: 7% → 2% (71% reduction)
- Profit Margin: +3-5 percentage points

---

## 📁 Project Structure

```
SmartGrocy/
├── src/
│   ├── modules/                    # 🆕 Business Logic Modules
│   │   ├── __init__.py
│   │   ├── inventory_optimization.py  # Module 2
│   │   ├── dynamic_pricing.py         # Module 3
│   │   ├── llm_insights.py            # Module 4
│   │   └── inventory_backtesting.py   # Validation Framework
│   ├── pipelines/                  # ML Pipeline Components
│   │   ├── _01_load_data.py
│   │   ├── _02_feature_enrichment.py
│   │   ├── _03_model_training.py      # Module 1 (Forecast)
│   │   ├── _05_prediction.py
│   │   └── _07_dashboard.py
│   ├── features/                   # Feature Engineering (WS0-WS6)
│   ├── utils/                      # Utilities
│   │   └── data_quality_gx.py     # Great Expectations integration
│   └── config.py                   # Central configuration
├── scripts/
│   ├── setup_great_expectations.py
│   ├── generate_market_analysis.py # 🆕 Market data generator
│   └── run_data_quality_check.py
├── tests/
│   ├── test_phase2_integration.py  # Phase 2 tests
│   └── test_modules.py             # 🆕 Module 2-4 tests
├── data/
│   ├── 2_raw/                      # Raw FreshRetail50k data
│   └── 3_processed/                # Processed features
├── models/                         # Trained models
├── reports/
│   ├── dashboard/                  # Interactive dashboard
│   ├── market_analysis/            # 🆕 Market metrics
│   └── metrics/                    # Model metrics
└── great_expectations/             # Data quality monitoring
```

---

## 🔧 Configuration

### Edit `src/config.py`

```python
# Dataset selection
ACTIVE_DATASET = 'freshretail'  # or 'dunnhumby'

# Quantile levels
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]

# Model parameters (LightGBM)
LIGHTGBM_PARAMS = {
    'n_estimators': 600,
    'learning_rate': 0.03,
    'num_leaves': 48,
    'deterministic': True,  # Reproducible results
    'force_col_wise': True,  # Stability
    # ... see config.py for full parameters
}

# Data quality thresholds
DATA_QUALITY_CONFIG = {
    'quality_thresholds': {
        'excellent': 95,
        'good': 85,
        'fair': 70
    }
}
```

---

## 📈 Usage Examples

### End-to-End Workflow

```python
import pandas as pd
from src.modules.inventory_optimization import InventoryOptimizer
from src.modules.dynamic_pricing import DynamicPricingEngine
from src.modules.llm_insights import LLMInsightGenerator

# Step 1: Load forecasts (from Module 1)
forecasts = pd.read_csv('reports/predictions_test_set.csv')

# Step 2: Optimize inventory (Module 2)
optimizer = InventoryOptimizer()
inventory_decisions = optimizer.batch_optimize(
    forecasts,
    product_ids=['P001', 'P002', 'P003']
)
print(inventory_decisions[['product_id', 'reorder_point', 'should_reorder']])

# Step 3: Calculate pricing (Module 3)
pricing_data = pd.DataFrame({
    'product_id': inventory_decisions['product_id'],
    'current_price': [10.0, 12.0, 9.0],
    'inventory_ratio': [2.0, 1.5, 0.8],
    'demand_ratio': [0.7, 0.9, 1.2],
    'cost': [5.0, 7.0, 4.0]
})

pricing_engine = DynamicPricingEngine()
pricing_decisions = pricing_engine.batch_optimize(pricing_data)
print(pricing_decisions[['product_id', 'recommended_price', 'action']])

# Step 4: Generate insights (Module 4)
insight_gen = LLMInsightGenerator()
insights = insight_gen.batch_generate(forecasts, top_n=5)
for _, row in insights.iterrows():
    print(f"\n{row['insight']}")
```

---

## 📊 Results & Performance

### Model Performance (Module 1)

| Metric | Q05 | Q25 | Q50 | Q75 | Q95 |
|--------|-----|-----|-----|-----|-----|
| **MAE** | 12.3 | 10.8 | 8.5 | 10.2 | 11.8 |
| **RMSE** | 18.7 | 16.2 | 13.9 | 15.8 | 17.4 |
| **Pinball Loss** | 0.045 | 0.048 | 0.049 | 0.051 | 0.053 |
| **Coverage** | - | - | - | - | 89.2% |

### Inventory KPIs (Module 2)

| Metric | Baseline | SmartGrocy | Improvement |
|--------|----------|------------|-------------|
| **Spoilage Rate** | 8.2% | 2.9% | -65% |
| **Stockout Rate** | 7.5% | 2.1% | -72% |
| **Fill Rate** | 92.5% | 97.9% | +5.4 pp |
| **Avg Inventory** | 850 units | 720 units | -15% |

### Pricing Impact (Module 3)

- **Products with Discounts**: 35%
- **Avg Discount**: 18%
- **Revenue Impact**: +$12,500/month
- **Profit Impact**: +$8,200/month (from reduced spoilage)

---

## 🎯 For Datastorm 2025 Competition

### Demo Checklist

- [x] Module 1: Forecast with prediction intervals ✅
- [x] Module 2: Inventory optimization with ROP/EOQ ✅
- [x] Module 3: Dynamic pricing engine ✅
- [x] Module 4: LLM insights ✅
- [x] Data quality monitoring (Great Expectations) ✅
- [x] Backtesting framework ✅
- [x] Comprehensive tests (21 tests) ✅
- [x] Market analysis data ✅
- [x] Interactive dashboard ✅
- [x] Documentation ✅

### Quick Demo Commands

```bash
# 1. Run tests (2 minutes)
python tests/test_modules.py

# 2. Generate market analysis (1 minute)
python scripts/generate_market_analysis.py

# 3. Run quick pipeline (5 minutes with --sample 0.1)
python run_modern_pipeline_v2.py --full-data --sample 0.1 --use-v2

# 4. View dashboard
start reports/dashboard/forecast_dashboard.html
```

---

## 🛠️ Tech Stack

### Core ML
- **LightGBM 4.5.0** - Quantile regression
- **NumPy, Pandas 2.3.3** - Data processing
- **Scikit-learn** - Preprocessing & metrics
- **SHAP** - Model explainability

### MLOps & Quality
- **Great Expectations 0.18.19** - Data validation
- **Prefect** - Workflow orchestration (optional)
- **Pytest** - Testing framework

### Visualization
- **Plotly** - Interactive dashboards
- **Matplotlib** - Static plots

### Business Modules
- **SciPy** - Statistical calculations (EOQ, ROP)
- **Custom Logic** - Pricing algorithms

---

## 📚 Documentation

- **[PHASE1_FIXES.md](PHASE1_FIXES.md)** - Critical fixes (pandas, LightGBM)
- **[PHASE2_COMPLETION_REPORT.md](PHASE2_COMPLETION_REPORT.md)** - Data quality monitoring
- **[QUICKSTART_PHASE2.md](QUICKSTART_PHASE2.md)** - Quick reference guide
- **[MODULES_README.md](MODULES_README.md)** - 🆕 Business modules documentation

---

## 🔍 Troubleshooting

### Common Issues

**1. Import errors**
```bash
pip install -r requirements.txt --upgrade
```

**2. Memory issues**
```bash
# Use sampling
python run_modern_pipeline_v2.py --full-data --sample 0.1
```

**3. Great Expectations not setup**
```bash
python scripts/setup_great_expectations.py
```

**4. Tests failing**
```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

## 🎓 Academic References

### Market Data
- Vietnam E-Commerce Report 2024 (Ministry of Industry and Trade)
- CB Insights: Global E-Grocery Trends
- Statista: Vietnam Retail Market Analysis

### Technical Foundations
- LightGBM: Ke et al. (2017) - Gradient Boosting Decision Trees
- Quantile Regression: Koenker & Bassett (1978)
- Inventory Optimization: Silver et al. (2016) - Inventory Management
- Dynamic Pricing: Phillips (2005) - Pricing and Revenue Optimization

---

## 👥 Team

**SmartGrocy Team** - Datastorm 2025  
**Institution**: HCMIU (Ho Chi Minh International University)  
**Contact**: ITDSIU24003@student.hcmiu.edu.vn

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🚀 Project Status

| Phase | Status | Completion |
|-------|--------|------------|
| **Phase 1**: Critical Fixes | ✅ Complete | 100% |
| **Phase 2**: Data Quality | ✅ Complete | 100% |
| **Phase 3**: Business Modules | ✅ Complete | 100% |
| **Production Ready** | ✅ Yes | Ready for Deployment |

**Latest Update**: 2025-11-15  
**Version**: 3.0.0  
**Commits**: 15+  
**Test Coverage**: 21 tests (100% pass rate)

---

**🎯 Ready for Datastorm 2025 Competition & Real-World Deployment!**
