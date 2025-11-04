# QA Report: Weekly Store-SKU Features

## 1. Basic Counts
- Total rows: 2,370,784
- Distinct stores: 582
- Distinct products: 92,339
- Distinct weeks: 102

## 2. Duplicate Key Check
- Rows with duplicate (store_id, product_id, week_no): 0

## 3. Value Range Checks
- Rows with negative units: 0
- Rows with negative baskets: 0
- Rows with non-positive avg_net_price: 4,036
- Rows with non-positive avg_gross_price: 620

## 4. Discount Rate Check
For rows with units > 0 and avg_gross_price > 0:
- Rows with avg_discount_rate outside [0,1]: 0

## 5. Promotion Distribution
- Rows with promo_display=1: 10.00%
- Rows with promo_mailer=1: 14.77%

## 6. Non-null Rates for Categorical Columns
- department: 100.00%
- brand: 100.00%
- commodity: 100.00%
- sub_commodity: 100.00%