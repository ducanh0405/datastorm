# 🤝 Contributing to E-Grocery Forecaster

Thank you for your interest in contributing to E-Grocery Forecaster! This document provides guidelines and information for contributors.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Contributing Guidelines](#contributing-guidelines)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## 🤝 Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. Please be respectful and constructive in all interactions.

## 🚀 Getting Started

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- Basic knowledge of machine learning and time series forecasting

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/ducanh0405/datastorm.git
cd E-Grocery_Forecaster

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create sample data for testing
python scripts/create_sample_data.py

# Run validation
python scripts/validate_setup.py
```

## 🛠️ Development Setup

### Development Dependencies
```bash
# Install all development tools
pip install -r requirements-dev.txt

# Manual code formatting and linting
ruff check src/ tests/ --fix
black src/ tests/
isort src/ tests/
mypy src/
```

### IDE Setup
- **VS Code/Cursor**: Recommended with Python extension
- **PyCharm**: Professional edition recommended for advanced debugging
- Configure Python interpreter to use the virtual environment

## 🏗️ Project Structure

```
E-Grocery_Forecaster/
├── src/                          # Production code
│   ├── pipelines/               # ML pipeline stages
│   │   ├── _01_load_data.py     # Data loading
│   │   ├── _02_feature_enrichment.py  # Feature engineering
│   │   ├── _03_model_training.py      # Model training
│   │   ├── _04_run_pipeline.py        # Pipeline orchestration
│   │   └── _05_prediction.py          # Inference & prediction
│   ├── features/                # Feature engineering modules
│   │   ├── ws0_aggregation.py         # Data aggregation
│   │   ├── ws1_relational_features.py # Relational features
│   │   ├── ws2_timeseries_features.py # Time-series features
│   │   ├── ws3_behavior_features.py   # Behavioral features
│   │   └── ws4_price_features.py      # Price/promo features
│   └── utils/                   # Utilities
│       ├── validation.py        # Data validation
│       └── visualization.py     # Dashboard & charts
├── scripts/                     # Utility scripts
├── tests/                       # Test suite
├── data/                        # Data directory (ignored)
├── models/                      # Trained models (ignored)
├── reports/                     # Reports and outputs
└── notebooks/                   # Jupyter notebooks
```

## 📝 Contributing Guidelines

### Code Style
- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Write descriptive variable and function names
- Add docstrings to all functions and classes

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (add, fix, refactor, docs, etc.)
- Keep first line under 50 characters
- Add detailed description if needed

Example:
```
feat: Add quantile regression models for probabilistic forecasting

- Implement Q05/Q50/Q95 quantile models
- Add pinball loss evaluation metric
- Update prediction intervals calculation
```

### Branch Naming
- Use descriptive branch names
- Prefix with type: `feature/`, `bugfix/`, `docs/`, `refactor/`
- Example: `feature/add-behavioral-features`

## 🧪 Testing

### Running Tests
```bash
# Run smoke tests (quick)
pytest tests/test_smoke.py -v -m smoke

# Run all tests
pytest tests/ -v

# Run comprehensive test suite
python test_project_comprehensive.py
```

### Test Coverage
- Aim for high test coverage on critical components
- Test edge cases and error conditions
- Use descriptive test names and assertions

### Adding New Tests
- Add tests for new features in `tests/` directory
- Follow naming convention: `test_*.py`
- Use pytest fixtures for common test data

## 📤 Submitting Changes

### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes with tests
4. **Test** your changes thoroughly
5. **Commit** with clear messages
6. **Push** to your fork
7. **Create** a Pull Request

### PR Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] Commit messages are clear
- [ ] No sensitive data included
- [ ] Changes don't break existing functionality

### Review Process
- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

## 🐛 Reporting Issues

### Bug Reports
When reporting bugs, please include:
- **Description**: Clear description of the issue
- **Steps to reproduce**: Detailed steps to reproduce the bug
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, dependencies
- **Logs/Error messages**: Any relevant error output

### Feature Requests
For new features, please include:
- **Description**: What feature you want to add
- **Use case**: Why this feature would be useful
- **Implementation ideas**: Any thoughts on how to implement it
- **Alternatives**: Other approaches you've considered

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation improvements
- `question`: Questions or discussions
- `help wanted`: Extra attention needed

## 📚 Additional Resources

- [README.md](README.md) - Project overview and setup
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [TEST_README.md](TEST_README.md) - Testing documentation
- [Project Wiki](https://github.com/ducanh0405/datastorm/wiki) - Additional documentation

## 🙏 Acknowledgments

Thank you to all contributors who help make E-Grocery Forecaster better! Your contributions, whether code, documentation, or feedback, are greatly appreciated.

---

**Happy contributing! 🚀**
