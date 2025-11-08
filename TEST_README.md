# 🧪 Comprehensive Project Testing Script

## Tổng quan

Script `test_project_comprehensive.py` là công cụ test toàn diện cho dự án E-Grocery Forecaster, kiểm tra tất cả các thành phần quan trọng của hệ thống.

## Cách sử dụng

### 1. Test cơ bản (nhanh - 2-3 phút)
```bash
python test_project_comprehensive.py
```

### 2. Test đầy đủ (bao gồm test chậm - 5-10 phút)
```bash
python test_project_comprehensive.py --full
```

### 3. Test mà không có end-to-end (nhanh nhất - 1-2 phút)
```bash
python test_project_comprehensive.py --no-end-to-end
```

## Các test được thực hiện

### 1. ✅ Setup Validation
- Kiểm tra cấu trúc thư mục
- Validate imports cơ bản
- Kiểm tra POC data

### 2. ✅ Smoke Tests
- Test smoke với pytest
- Validate core functionality
- Quick pipeline validation

### 3. ✅ Unit Tests
- Test các feature engineering modules
- WS1, WS2, WS4 validation
- Time-series logic tests

### 4. ✅ Optimized Pipeline Tests
- Test pipeline được tối ưu hóa
- WS2 speed improvements
- Optuna integration

### 5. ✅ Code Quality
- Ruff linting (nếu có)
- Import validation
- Module structure checks

### 6. ✅ Data Integrity
- Data directory structure
- Model file existence
- Feature file validation

### 7. ✅ Model Validation
- Model loading tests
- Prediction method checks
- Feature compatibility

### 8. ✅ End-to-End Pipeline (tùy chọn)
- Full pipeline execution
- WS0 → WS2 → Features
- Functional validation

### 9. ✅ Performance Benchmarks
- Speed comparisons
- Memory usage checks

## Output

Script tạo ra:
- **Console output**: Real-time progress và kết quả
- **test_results.log**: Chi tiết logs để debug
- **Exit code**: 0 (thành công) hoặc 1 (có lỗi)

## Kết quả mẫu

```
🚀 STARTING COMPREHENSIVE PROJECT TESTING
======================================================================
TEST 1: SETUP VALIDATION
======================================================================
✓ Setup validation - PASSED

======================================================================
TEST 2: SMOKE TESTS
======================================================================
✓ Smoke tests - PASSED

... [các test khác] ...

======================================================================
TEST SUMMARY
======================================================================
Total tests run: 9
Tests passed: 9
Tests failed: 0

  Setup Validation       : ✓ PASS
  Smoke Tests           : ✓ PASS
  Unit Tests            : ✓ PASS
  Optimized Pipeline Tests: ✓ PASS
  Code Quality          : ✓ PASS
  Data Integrity        : ✓ PASS
  Model Validation      : ✓ PASS
  End To End            : ✓ PASS
  Performance           : ✓ PASS

Total time: 45.67 seconds

🎉 ALL TESTS PASSED! Project is ready for production.
```

## Troubleshooting

### Nếu test thất bại:

1. **Check dependencies**: `pip install -r requirements.txt`
2. **Check data**: Chạy `python scripts/create_sample_data.py`
3. **Check logs**: Xem `test_results.log` chi tiết
4. **Run individual tests**: Chạy từng script riêng lẻ

### Common issues:

- **Import errors**: Cài đặt requirements
- **Data missing**: Tạo POC data
- **Model missing**: Chạy training pipeline
- **Timeout**: Sử dụng `--no-end-to-end`

## Dependencies

Script yêu cầu:
- Python 3.10+
- pytest
- joblib
- pandas
- numpy
- (tùy chọn) ruff cho code quality

## Integration với CI/CD

Có thể sử dụng trong GitHub Actions hoặc CI pipeline:

```yaml
- name: Run comprehensive tests
  run: python test_project_comprehensive.py --no-end-to-end
```
