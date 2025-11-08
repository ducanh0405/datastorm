"""
WS0: Aggregation & Grid Creation (Polars + Pandas)
===================================================
Auto-selects fastest implementation. Returns pandas DataFrame for compatibility.
"""
import pandas as pd
import numpy as np
import logging

# Import dependencies
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Import centralized config
try:
    from ..config import AGGREGATION_CONFIG, PERFORMANCE_CONFIG, setup_logging
    setup_logging()
    logger = logging.getLogger(__name__)
except ImportError:
    # Fallback config
    AGGREGATION_CONFIG = {
        'groupby_keys': ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
        'aggregation_rules': {'SALES_VALUE': 'sum', 'QUANTITY': 'sum',
                            'RETAIL_DISC': 'sum', 'COUPON_DISC': 'sum', 'COUPON_MATCH_DISC': 'sum'}
    }
    PERFORMANCE_CONFIG = {'use_polars': True, 'fallback_to_pandas': True}
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# Helper functions
def _get_fill_columns(df_columns):
    """Get columns that should be zero-filled."""
    base_cols = ['SALES_VALUE', 'QUANTITY']
    optional_cols = ['RETAIL_DISC', 'COUPON_DISC', 'COUPON_MATCH_DISC']
    return base_cols + [col for col in optional_cols if col in df_columns]


def _get_agg_rules(df_columns):
    """Get aggregation rules for existing columns."""
    base_rules = AGGREGATION_CONFIG.get('aggregation_rules', {'SALES_VALUE': 'sum', 'QUANTITY': 'sum'})
    optional_cols = ['RETAIL_DISC', 'COUPON_DISC', 'COUPON_MATCH_DISC']
    rules = base_rules.copy()
    for col in optional_cols:
        if col in df_columns and col not in rules:
            rules[col] = 'sum'
    return {k: v for k, v in rules.items() if k in df_columns}


def aggregate_to_weekly_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate transactions to weekly level using Polars (2-5x faster than pandas)."""
    groupby_keys = AGGREGATION_CONFIG['groupby_keys']

    # Build aggregation expressions
    agg_exprs = []
    for col, agg_func in AGGREGATION_CONFIG['aggregation_rules'].items():
        if col in df.columns:
            if agg_func == 'sum':
                agg_exprs.append(pl.col(col).sum().alias(col))
            elif agg_func == 'mean':
                agg_exprs.append(pl.col(col).mean().alias(col))
            elif agg_func == 'count':
                agg_exprs.append(pl.col(col).count().alias(col))

    df_agg = df.group_by(groupby_keys).agg(agg_exprs)
    logger.info(f"WS0-Polars: Aggregated {len(df):,} -> {len(df_agg):,} weekly records")
    return df_agg


def create_master_grid_polars(df_agg: pl.DataFrame) -> pl.DataFrame:
    """Create complete grid with zero-filling using Polars (5-10x faster than pandas)."""
    # Get dimensions and create grid
    all_products = df_agg.select('PRODUCT_ID').unique().sort()
    all_stores = df_agg.select('STORE_ID').unique().sort()
    all_weeks = df_agg.select('WEEK_NO').unique().sort()

    n_products, n_stores, n_weeks = len(all_products), len(all_stores), len(all_weeks)
    logger.info(f"WS0-Polars: Grid: {n_products:,}×{n_stores:,}×{n_weeks:,} = {n_products*n_stores*n_weeks:,} combinations")

    # Create grid using cross joins
    product_store_grid = all_products.join(all_stores, how='cross')
    full_grid = product_store_grid.join(all_weeks, how='cross').select(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])

    # Left join with aggregated data and fill nulls
    fill_cols = [pl.col(col).fill_null(0) for col in _get_fill_columns(df_agg.columns)]

    master_df = (
        full_grid
        .join(df_agg, on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'], how='left')
        .with_columns(fill_cols)
        .sort(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])
    )

    filled_rows = len(master_df) - len(df_agg)
    logger.info(f"WS0-Polars: Grid complete: {len(master_df):,} rows ({filled_rows:,} zero-filled)")
    return master_df


def prepare_master_dataframe_polars(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Main orchestrator using Polars optimization (6-15x faster than pandas)."""
    if not POLARS_AVAILABLE:
        return prepare_master_dataframe_pandas(raw_transactions)

    # Convert to Polars and process
    df_pl = pl.from_pandas(raw_transactions)
    df_weekly = aggregate_to_weekly_polars(df_pl)
    master_df = create_master_grid_polars(df_weekly)

    # Convert back to pandas for compatibility
    master_df_pd = master_df.to_pandas()

    # Verify time ordering
    is_sorted = master_df_pd.groupby(['PRODUCT_ID', 'STORE_ID'])['WEEK_NO'].apply(
        lambda x: (x.diff().dropna() >= 0).all()
    ).all()

    if not is_sorted:
        master_df_pd = master_df_pd.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)

    logger.info(f"WS0-Polars: Complete: {master_df_pd.shape[0]:,} rows, {master_df_pd.shape[1]} columns")
    return master_df_pd


# Pandas implementation (fallback)
def aggregate_to_weekly_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to weekly level using pandas."""
    groupby_keys = AGGREGATION_CONFIG.get('groupby_keys', ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])

    # Verify required columns
    required_cols = groupby_keys + ['SALES_VALUE', 'QUANTITY']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Build aggregation rules
    agg_rules = _get_agg_rules(df.columns)

    df_agg = df.groupby(groupby_keys, as_index=False).agg(agg_rules)
    logger.info(f"WS0-Pandas: Aggregated {len(df):,} -> {len(df_agg):,} weekly records")
    return df_agg


def create_master_grid_pandas(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Create complete grid with zero-filling using pandas."""
    # Get dimensions
    all_products = df_agg['PRODUCT_ID'].unique()
    all_stores = df_agg['STORE_ID'].unique()
    all_weeks = np.sort(df_agg['WEEK_NO'].unique())

    logger.info(f"WS0-Pandas: Grid: {len(all_products):,}×{len(all_stores):,}×{len(all_weeks):,} = {len(all_products)*len(all_stores)*len(all_weeks):,} combinations")

    # Create complete grid
    grid_index = pd.MultiIndex.from_product([all_products, all_stores, all_weeks], names=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])
    master_grid = pd.DataFrame(index=grid_index).reset_index()

    # Left join aggregated data
    master_df = pd.merge(master_grid, df_agg, on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'], how='left')

    # Zero-fill missing values
    fill_cols = _get_fill_columns(master_df.columns)
    master_df[fill_cols] = master_df[fill_cols].fillna(0)

    # Sort by time dimension
    master_df = master_df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)

    filled_rows = len(master_df) - len(df_agg)
    logger.info(f"WS0-Pandas: Grid complete: {len(master_df):,} rows ({filled_rows:,} zero-filled)")
    return master_df


def prepare_master_dataframe_pandas(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Main orchestrator using pandas (fallback implementation)."""
    # Process pipeline
    df_weekly = aggregate_to_weekly_pandas(raw_transactions)
    master_df = create_master_grid_pandas(df_weekly)

    # Verify time ordering
    is_sorted = master_df.groupby(['PRODUCT_ID', 'STORE_ID'])['WEEK_NO'].apply(
        lambda x: (x.diff().dropna() >= 0).all()
    ).all()

    if not is_sorted:
        master_df = master_df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)

    logger.info(f"WS0-Pandas: Complete: {master_df.shape[0]:,} rows, {master_df.shape[1]} columns")
    return master_df


# Main entry point - auto-selects best implementation
def prepare_master_dataframe_optimized(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Smart wrapper choosing fastest available implementation (Polars > Pandas)."""
    use_polars = PERFORMANCE_CONFIG.get('use_polars', True) and POLARS_AVAILABLE

    if use_polars:
        try:
            return prepare_master_dataframe_polars(raw_transactions)
        except Exception as e:
            if not PERFORMANCE_CONFIG.get('fallback_to_pandas', True):
                raise
            logger.warning(f"WS0: Polars failed: {e}. Using pandas fallback.")

    return prepare_master_dataframe_pandas(raw_transactions)


# Backward compatibility exports
prepare_master_dataframe = prepare_master_dataframe_optimized
enrich_aggregation_features = prepare_master_dataframe_optimized
aggregate_to_weekly = aggregate_to_weekly_pandas
