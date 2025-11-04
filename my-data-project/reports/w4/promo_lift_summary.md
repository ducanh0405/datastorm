# Promotion Lift — DID with Store & Week Fixed Effects

- Response: `log1p(units)`
- Regressors: `promo_display`, `promo_mailer`
- Fixed effects: `C(store_id)`, `C(week_no)`
- Robust SE: HC1


## Coefficients

| variable      |      coef |   pct_lift |   std_err |   ci_low |   ci_high |
|:--------------|----------:|-----------:|----------:|---------:|----------:|
| promo_display | 0.0165906 |     1.6729 |       nan |      nan |       nan |
| promo_mailer  | 0.0581095 |     5.9831 |       nan |      nan |       nan |


**Interpretation**: %lift ≈ (exp(beta) − 1) × 100.


### Notes
- FE theo store & week hấp thụ khác biệt nền giữa cửa hàng và mùa vụ.
- Có thể chạy theo department để granular hơn hoặc thêm kiểm soát khác.
