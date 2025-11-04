# Dunnhumby — Data Dictionary (tối thiểu)

_Generated: 2025-10-28 21:49:29_  

_Project root:_ `C:\Users\Admin\Desktop\datastorm\my-data-project`  

_RAW dir:_ `C:\Users\Admin\Desktop\datastorm\my-data-project\data\raw\Dunnhumby`


## campaign_desc.csv

- **Kích thước:** 0.00 MB

| column_name | column_type |
|---|---|
| DESCRIPTION | VARCHAR |
| CAMPAIGN | BIGINT |
| START_DAY | BIGINT |
| END_DAY | BIGINT |

**Ví dụ 3 dòng đầu:**
```
DESCRIPTION  CAMPAIGN  START_DAY  END_DAY
      TypeB        24        659      719
      TypeC        15        547      708
      TypeB        25        659      691
```

## campaign_table.csv

- **Kích thước:** 0.09 MB

| column_name | column_type |
|---|---|
| DESCRIPTION | VARCHAR |
| household_key | BIGINT |
| CAMPAIGN | BIGINT |

**Ví dụ 3 dòng đầu:**
```
DESCRIPTION  household_key  CAMPAIGN
      TypeA             17        26
      TypeA             27        26
      TypeA            212        26
```

## coupon.csv

- **Kích thước:** 2.69 MB

| column_name | column_type |
|---|---|
| COUPON_UPC | BIGINT |
| PRODUCT_ID | BIGINT |
| CAMPAIGN | BIGINT |

**Ví dụ 3 dòng đầu:**
```
 COUPON_UPC  PRODUCT_ID  CAMPAIGN
10000089061       27160         4
10000089064       27754         9
10000089073       28897        12
```

## coupon_redempt.csv

- **Kích thước:** 0.05 MB

| column_name | column_type |
|---|---|
| household_key | BIGINT |
| DAY | BIGINT |
| COUPON_UPC | BIGINT |
| CAMPAIGN | BIGINT |

**Ví dụ 3 dòng đầu:**
```
 household_key  DAY  COUPON_UPC  CAMPAIGN
             1  421 10000085364         8
             1  421 51700010076         8
             1  427 54200000033         8
```

## hh_demographic.csv

- **Kích thước:** 0.04 MB

| column_name | column_type |
|---|---|
| AGE_DESC | VARCHAR |
| MARITAL_STATUS_CODE | VARCHAR |
| INCOME_DESC | VARCHAR |
| HOMEOWNER_DESC | VARCHAR |
| HH_COMP_DESC | VARCHAR |
| HOUSEHOLD_SIZE_DESC | VARCHAR |
| KID_CATEGORY_DESC | VARCHAR |
| household_key | BIGINT |

**Ví dụ 3 dòng đầu:**
```
AGE_DESC MARITAL_STATUS_CODE INCOME_DESC HOMEOWNER_DESC     HH_COMP_DESC HOUSEHOLD_SIZE_DESC KID_CATEGORY_DESC  household_key
     65+                   A      35-49K      Homeowner 2 Adults No Kids                   2      None/Unknown              1
   45-54                   A      50-74K      Homeowner 2 Adults No Kids                   2      None/Unknown              7
   25-34                   U      25-34K        Unknown    2 Adults Kids                   3                 1              8
```

## product.csv

- **Kích thước:** 6.13 MB

| column_name | column_type |
|---|---|
| PRODUCT_ID | BIGINT |
| MANUFACTURER | BIGINT |
| DEPARTMENT | VARCHAR |
| BRAND | VARCHAR |
| COMMODITY_DESC | VARCHAR |
| SUB_COMMODITY_DESC | VARCHAR |
| CURR_SIZE_OF_PRODUCT | VARCHAR |

**Ví dụ 3 dòng đầu:**
```
 PRODUCT_ID  MANUFACTURER   DEPARTMENT    BRAND           COMMODITY_DESC          SUB_COMMODITY_DESC CURR_SIZE_OF_PRODUCT
      25671             2      GROCERY National                 FRZN ICE         ICE - CRUSHED/CUBED                22 LB
      26081             2 MISC. TRANS. National NO COMMODITY DESCRIPTION NO SUBCOMMODITY DESCRIPTION                     
      26093            69       PASTRY  Private                    BREAD        BREAD:ITALIAN/FRENCH                     
```

## transaction_data.csv

- **Kích thước:** 135.18 MB

| column_name | column_type |
|---|---|
| household_key | BIGINT |
| BASKET_ID | BIGINT |
| DAY | BIGINT |
| PRODUCT_ID | BIGINT |
| QUANTITY | BIGINT |
| SALES_VALUE | DOUBLE |
| STORE_ID | BIGINT |
| RETAIL_DISC | DOUBLE |
| TRANS_TIME | VARCHAR |
| WEEK_NO | BIGINT |
| COUPON_DISC | DOUBLE |
| COUPON_MATCH_DISC | DOUBLE |

**Ví dụ 3 dòng đầu:**
```
 household_key   BASKET_ID  DAY  PRODUCT_ID  QUANTITY  SALES_VALUE  STORE_ID  RETAIL_DISC TRANS_TIME  WEEK_NO  COUPON_DISC  COUPON_MATCH_DISC
          2375 26984851472    1     1004906         1         1.39       364         -0.6       1631        1          0.0                0.0
          2375 26984851472    1     1033142         1         0.82       364          0.0       1631        1          0.0                0.0
          2375 26984851472    1     1036325         1         0.99       364         -0.3       1631        1          0.0                0.0
```

## causal_data.csv

- **Kích thước:** 663.62 MB

| column_name | column_type |
|---|---|
| PRODUCT_ID | BIGINT |
| STORE_ID | BIGINT |
| WEEK_NO | BIGINT |
| display | VARCHAR |
| mailer | VARCHAR |

**Ví dụ 3 dòng đầu:**
```
 PRODUCT_ID  STORE_ID  WEEK_NO display mailer
      26190       286       70       0      A
      26190       288       70       0      A
      26190       289       70       0      A
```

---
## Gợi ý làm sạch mặc định

- Chuẩn hoá text: trim khoảng trắng; thống nhất UPPER/Title; điền 'UNKNOWN' cho danh mục trống.
- Số tiền/giảm giá: ép float; thay NULL bằng 0.0 khi phù hợp.
- Ngày/giờ: `DAY` (ordinal) → thêm cột `date`; `TRANS_TIME` (hhmm) → `time` nếu cần; giữ `WEEK_NO` dạng int.
- Khóa: kiểm tra trùng `PRODUCT_ID` (product); kiểm tra trùng `(household_key,BASKET_ID,PRODUCT_ID,DAY)` (transaction).
- Join keys: `PRODUCT_ID`, `household_key`, `STORE_ID`, `WEEK_NO`, `CAMPAIGN`, `COUPON_UPC`.
