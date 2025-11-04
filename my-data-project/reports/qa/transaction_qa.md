# QA Report: transaction
_Generated: 2025-10-29 14:43:59_

## Basic Stats
- Row count: 2,595,732
- Column count: 16

## Schema
- household_key: BIGINT
- basket_id: BIGINT
- day: INTEGER
- week_no: INTEGER
- store_id: INTEGER
- product_id: BIGINT
- quantity: INTEGER
- sales_value: DOUBLE
- total_discount: DOUBLE
- gross_sales: DOUBLE
- unit_price_net: DOUBLE
- unit_price_gross: DOUBLE
- is_return: BOOLEAN
- date_synth: TIMESTAMP
- hour: INTEGER
- minute: INTEGER

## NULL Percentages
- household_key: 0.00%
- basket_id: 0.00%
- day: 0.00%
- week_no: 0.00%
- store_id: 0.00%
- product_id: 0.00%
- quantity: 0.00%
- sales_value: 0.00%
- total_discount: 0.00%
- gross_sales: 0.00%
- unit_price_net: 0.56%
- unit_price_gross: 0.56%
- is_return: 0.00%
- date_synth: 0.00%
- hour: 0.16%
- minute: 0.16%

## Sample Rows
```
   household_key    basket_id  day  week_no  store_id  product_id  quantity  sales_value  total_discount  gross_sales  unit_price_net  unit_price_gross  is_return date_synth  hour  minute
0           2375  26984851472    1        1       364     1004906         1         1.39           -0.60         1.99            1.39              1.99      False 2015-01-01    16      31
1           2375  26984851472    1        1       364     1033142         1         0.82            0.00         0.82            0.82              0.82      False 2015-01-01    16      31
2           2375  26984851472    1        1       364     1036325         1         0.99           -0.30         1.29            0.99              1.29      False 2015-01-01    16      31
3           2375  26984851472    1        1       364     1082185         1         1.21            0.00         1.21            1.21              1.21      False 2015-01-01    16      31
4           2375  26984851472    1        1       364     8160430         1         1.50           -0.39         1.89            1.50              1.89      False 2015-01-01    16      31
```
