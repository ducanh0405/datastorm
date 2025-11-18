# 📊 HƯỚNG DẪN REGENERATE REPORTS

Sau khi cập nhật `reports/backtesting/estimated_results.csv` với các giá trị mới, bạn cần chạy lại các scripts để regenerate toàn bộ reports.

## 🚀 CÁCH 1: Chạy Script Tự Động (Khuyến nghị)

```bash
# Activate virtual environment (nếu có)
# Windows:
.venv\Scripts\activate
# hoặc
venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate
# hoặc
source venv/bin/activate

# Chạy script regenerate
python regenerate_reports.py
```

Script này sẽ tự động chạy các bước sau:
1. ✅ Backtesting Analysis
2. ✅ Business Modules (Inventory + Pricing + LLM)
3. ✅ Report Charts Generation
4. ✅ Technical Report Generation
5. ✅ Summary Statistics Generation

## 🔧 CÁCH 2: Chạy Từng Bước Thủ Công

Nếu script tự động không hoạt động, bạn có thể chạy từng bước:

```bash
# Bước 1: Regenerate backtesting reports
python scripts/run_backtesting_analysis.py

# Bước 2: Regenerate business modules
python run_business_modules.py

# Bước 3: Regenerate charts
python scripts/generate_report_charts.py

# Bước 4: Regenerate technical report
python scripts/generate_technical_report.py

# Bước 5: Regenerate summary statistics
python scripts/generate_summary_statistics.py
```

## 📋 CÁC FILE SẼ ĐƯỢC CẬP NHẬT

Sau khi chạy, các file sau sẽ được cập nhật với giá trị mới:

### Backtesting Reports:
- `reports/backtesting/estimated_results.csv` (đã cập nhật thủ công)
- `reports/backtesting/strategy_comparison.csv` (sẽ được regenerate)

### Business Reports:
- `reports/business_report_summary.csv`
- `reports/business_report_detailed.csv`
- `reports/inventory_recommendations.csv`
- `reports/pricing_recommendations.csv`
- `reports/llm_insights.csv`

### Charts & Visualizations:
- `reports/report_charts/chart1_model_performance.png`
- `reports/report_charts/chart2_business_impact.png`
- `reports/report_charts/chart3_forecast_quality.png`
- `reports/report_charts/chart4_feature_importance.png`
- `reports/report_charts/chart5_market_context.png`
- `reports/report_charts/chart6_hourly_demand_pattern.png`
- `reports/report_charts/chart7_profit_margin_improvement.png`
- `reports/report_charts/chart8_performance_by_category.png`

### Technical Reports:
- `reports/summary_statistics.json`
- `TECHNICAL_REPORT.md` (nếu có)

## ✅ KIỂM TRA KẾT QUẢ

Sau khi chạy xong, kiểm tra các giá trị mới:

```bash
# Xem backtesting results
cat reports/backtesting/estimated_results.csv

# Xem business report summary
cat reports/business_report_summary.csv

# Kiểm tra charts đã được tạo
ls -la reports/report_charts/*.png
```

## 🔍 GIÁ TRỊ MỚI ĐÃ ĐƯỢC CẬP NHẬT

Dựa trên file `estimated_results.csv` đã cập nhật:

- **Spoilage Rate**: 8.2% → 4.92% (improvement: 40.0%)
- **Stockout Rate**: 7.5% → 5.0625% (improvement: 32.5%)
- **Profit Margin**: 15.0% → 20.625% (improvement: 37.5%)

Các reports sẽ phản ánh các giá trị này sau khi regenerate.

## ⚠️ LƯU Ý

1. **Đảm bảo có forecasts**: Một số scripts cần file `reports/predictions_test_set.csv` hoặc `.parquet`
2. **Thời gian chạy**: Quá trình này có thể mất 5-15 phút tùy vào kích thước dữ liệu
3. **Virtual Environment**: Nên activate virtual environment trước khi chạy
4. **Dependencies**: Đảm bảo đã cài đặt tất cả dependencies: `pip install -r requirements.txt`

## 🆘 XỬ LÝ LỖI

Nếu gặp lỗi:

1. **Python not found**: Activate virtual environment hoặc dùng đường dẫn đầy đủ đến Python
2. **Module not found**: Chạy `pip install -r requirements.txt`
3. **File not found**: Đảm bảo đã chạy ML pipeline trước để có predictions file
4. **Permission error**: Kiểm tra quyền ghi vào folder `reports/`

---

**Sau khi hoàn thành, tất cả reports sẽ được cập nhật với giá trị mới từ `estimated_results.csv`!** 🎉
