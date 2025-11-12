# FreshRetailNet-50K Dataset

Download and manage the FreshRetailNet-50K dataset for e-commerce forecasting.

## 🚀 Quick Start

```bash
# Install dependencies
pip install datasets

# Load full dataset (4.85M rows)
python scripts/load_freshretail_datasets.py

# Load sample for testing (10K rows)
python scripts/load_freshretail_datasets.py --sample

# Export to other formats
python scripts/export_freshretail_data.py
```

## 📊 Dataset Info

- **Source**: Dingdong-Inc/FreshRetailNet-50K
- **Size**: 4,850,000 rows (4.5M train + 350K eval)
- **Columns**: 19 features including sales, weather, holidays
- **Timeframe**: Hourly sales data with weather integration

## 📁 Output Files

Script creates these files in `data/2_raw/`:
- `freshretail_train.csv/parquet` - Training data
- `freshretail_eval.csv/parquet` - Evaluation data
- `freshretail_datasets_metadata.json` - Dataset metadata

## 🛠️ Command Options

```bash
# Full dataset
python scripts/load_freshretail_datasets.py

# Sample for testing
python scripts/load_freshretail_datasets.py --sample

# Limit rows per split
python scripts/load_freshretail_datasets.py --max-rows 100000
```

## 📊 Schema

| Column | Type | Description |
|--------|------|-------------|
| city_id | int | City identifier |
| store_id | int | Store identifier |
| product_id | int | Product identifier |
| dt | string | Date (YYYY-MM-DD) |
| sale_amount | float | Sales amount |
| hours_sale | array | Hourly sales (24 elements) |
| discount | float | Discount amount |
| holiday_flag | int | Holiday indicator |
| precpt | float | Precipitation |
| avg_temperature | float | Average temperature |

## 🛠️ Troubleshooting

- **No datasets library**: `pip install datasets`
- **Timeout**: Check internet, use `--sample`
- **Memory**: Use `--sample` or `--max-rows`