# 📋 Changelog - SmartGrocy

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 📝 Changed
- **Project Rename**: Renamed from "E-Grocery Forecaster" to "SmartGrocy" across all documentation and code
- **Updated Documentation**: Refreshed all docs to reflect modern pipeline and current project structure

## [2.0.0] - 2025-11-09

### 🎉 Major Release - Complete Refactoring

#### Added
- **Interactive Dashboard**: Complete HTML dashboard with Plotly charts
  - Prediction accuracy visualizations
  - Quantile comparison plots
  - Individual product forecasts
  - Feature importance analysis
- **Quantile Regression Models**: Probabilistic forecasting with Q05/Q50/Q95
  - Prediction intervals for inventory optimization
  - Pinball loss evaluation metrics
  - 78.6% coverage accuracy
- **Prediction Pipeline**: Full inference and prediction API
  - QuantileForecaster class for real-time predictions
  - Batch prediction capabilities
  - Model serialization with joblib
- **Comprehensive Testing Suite**:
  - Smoke tests for core functionality
  - Validation scripts
  - Comprehensive project testing
- **Documentation Updates**:
  - Updated README with current status
  - CONTRIBUTING.md guidelines
  - MIT License file
  - Enhanced quickstart guide

#### Changed
- **Pipeline Architecture**: Complete modular refactoring
  - Leak-safe time-series features
  - Optimized data aggregation (6-15x faster)
  - Production-ready error handling
- **Development Tools**: Added modern Python tooling
  - ruff, black, isort, mypy for code quality
  - Removed pre-commit hooks for demo focus
- **Data Processing**: Enhanced POC data creation
  - Fixed PRODUCT_ID matching issues
  - 100% relational data integrity
  - Proper sampling with referential integrity

#### Removed
- **CI/CD Components**: Removed GitHub Actions and pre-commit hooks
- **Legacy Features**: Old dashboard files and deprecated code
- **Unused Dependencies**: Cleaned up requirements

#### Performance
- **WS0 Aggregation**: 6-15x speedup with Polars optimization
- **WS2 Features**: 10x faster vectorized operations
- **Pipeline**: 4.7x overall performance improvement

## [1.0.0] - 2025-10-01

### Added
- Initial SmartGrocy implementation
- Basic time-series forecasting pipeline
- Feature engineering workstreams (WS0-WS4)
- LightGBM regression models
- Basic evaluation metrics

### Changed
- Migrated from prototype to production-ready code
- Enhanced data processing capabilities
- Improved model training pipeline

---

## 📝 Types of Changes
- `Added` for new features
- `Changed` for changes in existing functionality
- `Deprecated` for soon-to-be removed features
- `Removed` for now removed features
- `Fixed` for any bug fixes
- `Security` in case of vulnerabilities

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

---

**Legend:**
- 🚀 Major feature addition
- 🔧 Enhancement/improvement
- 🐛 Bug fix
- 📚 Documentation update
- 🔒 Security fix
