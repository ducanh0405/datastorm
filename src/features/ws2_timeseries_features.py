# (Ben trong file ws2_timeseries_features.py)
# ... (Giu nguyen cac ham _clean, _create_features, _create_lag_rolling, ...)
import logging


def add_timeseries_features(master_df, dataframes_dict):
    """
    Ham "chu" (master function) cho Workstream 2
    (Da refactor de chay tren Dunnhumby)
    """

    # 1. Kiem tra du lieu can thiet
    if 'SALES_VALUE' not in master_df.columns or 'DAY' not in master_df.columns:
        logging.warning("SKIPPING WS2: Khong tim thay cot 'SALES_VALUE' hoac 'DAY'.")
        return master_df

    # 2. Refactor logic M5 cho Dunnhumby
    # (Day la mot vi du don gian hoa. Ban se ap dung logic Polars o day)
    logging.info("[WS2] Creating lag/rolling features for Dunnhumby...")

    # Doi ten cot de tai su dung code Polars (neu ban muon)
    df_temp = master_df.rename(columns={'SALES_VALUE': 'sales', 'PRODUCT_ID': 'id'})

    # (Goi cac ham _create_lag_rolling_features (Polars) o day...)
    # df_temp = _create_lag_rolling_features(df_temp) 

    # Vi du don gian (Pandas):
    df_temp['sales_lag_7'] = df_temp.groupby('id')['sales'].shift(7)
    df_temp['rolling_mean_7_lag_7'] = df_temp.groupby('id')['sales_lag_7'].transform(lambda x: x.rolling(7).mean())

    # Chi giu lai cac cot moi
    new_cols = ['sales_lag_7', 'rolling_mean_7_lag_7']  # va cac cot khac ban tao ra

    # Merge tro lai master_df (can mot khoa duy nhat, vi du: index)
    # Day la phan phuc tap nhat (Integration Hell)
    # Giai phap don gian la tra ve df_temp

    logging.info("OK. WS2 (Time-Series) integration complete.")
    return df_temp  # Tra ve bang da duoc lam giau