# Refactoring Plan - SmartGrocy Project Cleanup

## 🎯 Mục tiêu
1. Loại bỏ file duplicate/thừa
2. Tổ chức lại cấu trúc thư mục rõ ràng
3. Consolidate code để dễ maintain
4. Tạo entry point chính

## 📋 Các file cần xử lý

### 1. Test Files (di chuyển vào tests/)
- `test_import_config.py` → `tests/test_config_import.py`
- `test_pipeline_quick.py` → `tests/test_pipeline_quick.py`
- `test_pipeline_sample.py` → `tests/test_pipeline_sample.py`
- `test_refactoring_validation.py` → `tests/test_refactoring.py`

### 2. Documentation (consolidate)
- `QUICKSTART_PHASE2.md` → Merge vào `docs/QUICKSTART.md`
- `QUICKSTART_OPTIMIZED.md` → Merge vào `docs/QUICKSTART.md`
- `PHASE1_FIXES.md` → Move to `docs/archive/`
- `PHASE2_COMPLETION_REPORT.md` → Move to `docs/archive/`
- `REFACTORING_REPORT.md` → Move to `docs/archive/`
- `TEST_FINAL_RESULTS.md` → Merge vào `docs/TEST_README.md`
- `TEST_RESULTS.md` → Merge vào `docs/TEST_README.md`
- `TEST_GUIDE.md` → Merge vào `docs/TEST_README.md`
- `ISSUES_REPORT.md` → Move to `docs/archive/`

### 3. Scripts (consolidate)
- `run_modern_pipeline.py` và `run_modern_pipeline_v2.py` → Merge thành `run_pipeline.py`
- `scripts/demo_modern_pipeline.py` → Remove (duplicate)

### 4. Models (cleanup)
- Xóa duplicate: `q05_forecaster.joblib`, `q25_forecaster.joblib`, etc. (giữ `lightgbm_*`)

### 5. Entry Points (tạo mới)
- Tạo `main.py` làm entry point chính
- Tạo `cli.py` cho CLI commands

## 📁 Cấu trúc mới

```
E-Grocery_Forecaster/
├── main.py                 # Main entry point
├── cli.py                  # CLI commands
├── run_pipeline.py         # Consolidated pipeline runner
├── run_business_modules.py # Business modules runner
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── config/                 # Configuration files
│   └── pipeline_config.json
│
├── data/                   # Data directories
│   ├── 2_raw/
│   └── 3_processed/
│
├── docs/                   # Documentation
│   ├── QUICKSTART.md
│   ├── CONTRIBUTING.md
│   ├── CHANGELOG.md
│   └── archive/           # Old reports/docs
│
├── src/                    # Source code
│   ├── config.py
│   ├── features/
│   ├── modules/
│   ├── pipelines/
│   └── utils/
│
├── tests/                  # All tests
│   ├── test_*.py
│   └── ...
│
├── scripts/                # Utility scripts
│   └── ...
│
├── models/                 # Trained models
├── reports/                # Output reports
└── logs/                   # Log files
```

