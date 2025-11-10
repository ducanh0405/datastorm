"""
WS0: Aggregation & Grid Creation (Polars + Pandas)
===================================================
Auto-selects fastest implementation. Returns pandas DataFrame for compatibility.
"""
import logging

import numpy as np
import pandas as pd

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
    # Force garbage collection before processing
    import gc
    gc.collect()

    logger.info(f"WS0-Pandas: Processing {len(raw_transactions):,} transactions")

    # Process pipeline
    df_weekly = aggregate_to_weekly_pandas(raw_transactions)

    # Free up memory
    del raw_transactions
    gc.collect()

    logger.info(f"WS0-Pandas: Weekly aggregation complete: {len(df_weekly):,} rows")

    # Use OPTIMIZED grid to avoid memory issues with full data
    master_df = create_optimized_master_grid(df_weekly)

    # Free up memory
    del df_weekly
    gc.collect()

    # Verify time ordering
    is_sorted = master_df.groupby(['PRODUCT_ID', 'STORE_ID'])['WEEK_NO'].apply(
        lambda x: (x.diff().dropna() >= 0).all()
    ).all()

    if not is_sorted:
        master_df = master_df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)

    logger.info(f"WS0-Pandas: Complete: {master_df.shape[0]:,} rows, {master_df.shape[1]} columns")
    return master_df


def prepare_master_dataframe_optimized(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Optimized master dataframe preparation using ACTIVE pairs only.

    This eliminates data sparsity by only creating time series for product-store
    combinations that have at least one transaction in the dataset.

    FORCE OPTIMIZATION: Always uses pandas + optimized grid for memory safety.
    """
    logger.info("WS0-Optimized: Starting OPTIMIZED master dataframe preparation (ACTIVE pairs only + FORCE pandas)")

    # Force garbage collection before processing
    import gc
    gc.collect()

    logger.info(f"WS0-Optimized: Processing {len(raw_transactions):,} transactions (FORCE pandas mode)")

    # FORCE pandas implementation for memory safety
    return prepare_master_dataframe_pandas(raw_transactions)


# Legacy main entry point - auto-selects best implementation
def prepare_master_dataframe(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Smart wrapper choosing memory-efficient implementation for large datasets."""

    # Auto-detect large datasets and force pandas for memory efficiency
    is_large_dataset = (
        len(raw_transactions) > 1_000_000 or  # > 1M rows
        PERFORMANCE_CONFIG.get('force_pandas_for_large_data', False)
    )

    use_polars = (
        PERFORMANCE_CONFIG.get('use_polars', True) and
        POLARS_AVAILABLE and
        not is_large_dataset  # Force pandas for large datasets
    )

    if use_polars:
        try:
            logger.info("WS0: Using Polars for performance (small dataset detected)")
            return prepare_master_dataframe_polars(raw_transactions)
        except Exception as e:
            if not PERFORMANCE_CONFIG.get('fallback_to_pandas', True):
                raise
            logger.warning(f"WS0: Polars failed: {e}. Using pandas fallback.")

    logger.info("WS0: Using Pandas for memory efficiency (large dataset or forced)")
    return prepare_master_dataframe_pandas(raw_transactions)


def create_optimized_master_grid(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Create optimized master grid for ACTIVE product-store pairs only.

    This eliminates sparsity by only creating time series for product-store
    combinations that have at least one transaction in the dataset.

    Args:
        df_weekly: Weekly aggregated transaction data

    Returns:
        DataFrame with time series for active product-store pairs only
    """
    # Get active product-store pairs (those with at least 1 transaction)
    active_pairs = df_weekly[['PRODUCT_ID', 'STORE_ID']].drop_duplicates()
    n_active_pairs = len(active_pairs)

    # Get all weeks in the dataset (not just 1-104)
    all_weeks = pd.Series(sorted(df_weekly['WEEK_NO'].unique()), name='WEEK_NO')
    n_weeks = len(all_weeks)

    total_optimized = n_active_pairs * n_weeks
    total_full = len(df_weekly['PRODUCT_ID'].unique()) * len(df_weekly['STORE_ID'].unique()) * 104

    logger.info(f"WS0-Optimized: ACTIVE pairs only: {n_active_pairs:,} pairs × {n_weeks:,} weeks = {total_optimized:,} rows")
    logger.info(f"WS0-Optimized: Memory reduction: {((total_full - total_optimized) / total_full * 100):.1f}%")

    # Create optimized grid using active pairs only
    optimized_grid = pd.DataFrame([
        (row.PRODUCT_ID, row.STORE_ID, week)
        for _, row in active_pairs.iterrows()
        for week in all_weeks
    ], columns=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])

    # Left join with weekly data
    master_df = optimized_grid.merge(
        df_weekly,
        on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
        how='left'
    )

    # Zero-fill missing sales data
    fill_cols = _get_fill_columns(df_weekly.columns)
    master_df[fill_cols] = master_df[fill_cols].fillna(0)

    # Sort by time dimension for time-series features
    master_df = master_df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)

    filled_rows = len(master_df) - len(df_weekly)
    logger.info(f"WS0-Optimized: Grid complete: {len(master_df):,} rows ({filled_rows:,} zero-filled, {(filled_rows/len(master_df)*100):.1f}%)")

    return master_df


def create_master_grid_pandas_chunked(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """Create master grid using chunked processing for memory efficiency."""

    # Get unique values
    all_products = df_weekly['PRODUCT_ID'].unique()
    all_stores = df_weekly['STORE_ID'].unique()
    all_weeks = pd.Series(range(1, 105), name='WEEK_NO')  # Weeks 1-104

    n_products, n_stores, n_weeks = len(all_products), len(all_stores), len(all_weeks)
    total_combinations = n_products * n_stores * n_weeks

    logger.info(f"WS0-Pandas-Chunked: Grid: {n_products:,}×{n_stores:,}×{n_weeks:,} = {total_combinations:,} combinations")

    # Estimate memory usage and determine chunk size
    chunk_size_mb = PERFORMANCE_CONFIG.get('chunk_size_mb', 50)
    estimated_row_size = 100  # bytes per row estimate
    rows_per_chunk = (chunk_size_mb * 1024 * 1024) // estimated_row_size

    logger.info(f"WS0-Pandas-Chunked: Processing in chunks of ~{rows_per_chunk:,} rows")

    # Create chunks of product-store combinations
    product_store_combinations = pd.MultiIndex.from_product(
        [all_products, all_stores],
        names=['PRODUCT_ID', 'STORE_ID']
    )

    chunks = []
    chunk_size = min(len(product_store_combinations) // 4 + 1, 10000)  # Adaptive chunking

    for i in range(0, len(product_store_combinations), chunk_size):
        chunk_combinations = product_store_combinations[i:i+chunk_size]

        # Create grid for this chunk
        chunk_grid = pd.DataFrame(
            [(pid, sid, wk) for (pid, sid) in chunk_combinations for wk in all_weeks],
            columns=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']
        )

        # Left join with weekly data
        chunk_result = chunk_grid.merge(
            df_weekly,
            on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
            how='left'
        )

        # Fill nulls
        fill_cols = _get_fill_columns(df_weekly.columns)
        for col in fill_cols:
            chunk_result[col] = chunk_result[col].fillna(0)

        chunks.append(chunk_result)

        # Garbage collection between chunks
        gc.collect()

        logger.info(f"WS0-Pandas-Chunked: Processed chunk {len(chunks)}/{(len(product_store_combinations) // chunk_size) + 1}")

    # Combine all chunks
    master_df = pd.concat(chunks, ignore_index=True)

    # Sort for time ordering
    master_df = master_df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)

    # Final garbage collection
    del chunks, product_store_combinations
    gc.collect()

    filled_rows = len(master_df) - len(df_weekly)
    logger.info(f"WS0-Pandas-Chunked: Grid complete: {len(master_df):,} rows ({filled_rows:,} zero-filled)")
    return master_df


# Backward compatibility exports
prepare_master_dataframe = prepare_master_dataframe_optimized
enrich_aggregation_features = prepare_master_dataframe_optimized
aggregate_to_weekly = aggregate_to_weekly_pandas
