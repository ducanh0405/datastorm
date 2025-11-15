"""
WS0: Aggregation & Grid Creation (Polars + Pandas + Config-Driven)
==================================================================
Auto-selects fastest implementation with dataset-driven configuration.

Main Functions:
- create_master_grid(): Config-driven master grid creation for any dataset
- prepare_master_dataframe(): Smart wrapper choosing best implementation
- prepare_master_dataframe_polars(): Polars optimized (6-15x faster)
- prepare_master_dataframe_pandas(): Pandas fallback with memory optimization

Supports both FreshRetail (hourly, with stockout) and Dunnhumby (weekly, no stockout).
"""
import gc
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
    from src.config import AGGREGATION_CONFIG, PERFORMANCE_CONFIG, get_dataset_config, setup_logging
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
    # Fallback dataset config for Dunnhumby
    def get_dataset_config():
        return {
            'temporal_unit': 'week',
            'time_column': 'WEEK_NO',
            'groupby_keys': ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
            'required_columns': ['PRODUCT_ID', 'STORE_ID', 'WEEK_NO', 'SALES_VALUE'],
            'lag_periods': [1, 4, 8, 12],
            'rolling_windows': [4, 8, 12],
            'has_stockout': False,
            'has_weather': False,
            'has_intraday_patterns': False,
            'file_format': 'csv',
        }
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# Helper functions
def _get_fill_columns(df_columns):
    """Get columns that should be zero-filled."""
    from src.config import get_dataset_config
    config = get_dataset_config()

    # Use target column from config
    target_col = config['target_column']
    base_cols = [target_col]

    # Add QUANTITY if it exists
    if 'QUANTITY' in df_columns:
        base_cols.append('QUANTITY')

    optional_cols = ['RETAIL_DISC', 'COUPON_DISC', 'COUPON_MATCH_DISC']
    return base_cols + [col for col in optional_cols if col in df_columns]


def _get_agg_rules(df_columns):
    """Get aggregation rules for existing columns."""
    from src.config import get_dataset_config
    config = get_dataset_config()

    # Use target column from config
    target_col = config['target_column']

    # Base rules using config target column
    base_rules = {target_col: 'sum'}
    if 'QUANTITY' in df_columns:
        base_rules['QUANTITY'] = 'sum'

    optional_cols = ['RETAIL_DISC', 'COUPON_DISC', 'COUPON_MATCH_DISC']
    rules = base_rules.copy()
    for col in optional_cols:
        if col in df_columns and col not in rules:
            rules[col] = 'sum'
    return {k: v for k, v in rules.items() if k in df_columns}


def create_master_grid(df: pd.DataFrame, dataset_config: dict = None) -> pd.DataFrame:
    """
    Create master grid using dataset configuration.
    Handles different datasets (FreshRetail vs Dunnhumby) with appropriate zero-filling logic.

    Args:
        df: Raw transaction data
        dataset_config: Dataset configuration dict. If None, uses get_dataset_config()

    Returns:
        DataFrame with master grid (complete time series for all combinations)
    """
    config = dataset_config or get_dataset_config()

    groupby_keys = config['groupby_keys']
    time_col = config['time_column']
    has_stockout = config['has_stockout']
    temporal_unit = config['temporal_unit']

    logger.info(f"WS0-Config: Creating master grid for {temporal_unit}-level data")
    logger.info(f"WS0-Config: Groupby keys: {groupby_keys}")
    logger.info(f"WS0-Config: Time column: {time_col}")
    logger.info(f"WS0-Config: Has stockout: {has_stockout}")

    # Aggregation logic using config
    agg_rules = _get_agg_rules(df.columns)
    logger.info(f"WS0-Config: Aggregation rules: {agg_rules}")

    agg_df = df.groupby(groupby_keys, as_index=False).agg(agg_rules)
    logger.info(f"WS0-Config: Aggregated {len(df):,} -> {len(agg_df):,} {temporal_unit}ly records")

    # Handle stockout vs non-stockout datasets differently
    if has_stockout:
        # For FreshRetail: Keep zeros with stockout flag (don't filter zeros)
        # Use sales_quantity instead of SALES_VALUE for FreshRetail
        if 'sales_quantity' in agg_df.columns:
            filter_col = 'sales_quantity'
        else:
            filter_col = 'SALES_VALUE'
        logger.info(f"WS0-Config: Stockout dataset - keeping all records (including zeros)")
        # No filtering for stockout datasets - keep all time periods
    else:
        # For Dunnhumby: Filter zeros normally (traditional retail behavior)
        filter_col = 'SALES_VALUE'
        agg_df = agg_df[agg_df[filter_col] > 0]
        logger.info(f"WS0-Config: Non-stockout dataset - filtered zeros: {len(agg_df)} records remain")

    # Get unique entity combinations that actually exist in the data
    # This is much more memory-efficient than creating all possible combinations
    entity_keys = groupby_keys[:-1]  # All except time column
    unique_entity_combos = agg_df[entity_keys].drop_duplicates()
    
    # Get time range based on dataset
    if temporal_unit == 'hour':
        # For FreshRetail: use actual hours in data (not fixed range)
        time_values = sorted(agg_df[time_col].unique())
    else:  # week
        # For Dunnhumby: use weeks 1-104
        time_values = list(range(1, 105))

    logger.info(f"WS0-Config: Unique entity combinations: {len(unique_entity_combos):,}")
    logger.info(f"WS0-Config: Time periods: {len(time_values)} {temporal_unit}s")
    
    # Calculate total grid size
    total_grid_size = len(unique_entity_combos) * len(time_values)
    logger.info(f"WS0-Config: Total grid size: {total_grid_size:,} rows")
    
    # Check memory requirements and warn if too large
    estimated_memory_gb = (total_grid_size * 8) / (1024**3)  # Rough estimate in GB
    if estimated_memory_gb > 2.0:
        logger.warning(f"WS0-Config: Large grid detected! Estimated memory: {estimated_memory_gb:.2f} GB")
        logger.warning(f"WS0-Config: Consider enabling MEMORY_OPTIMIZATION in config.py")
    
    # Memory-efficient grid creation: only for existing entity combinations
    # Use chunking if grid is very large
    from src.config import MEMORY_OPTIMIZATION
    use_chunking = MEMORY_OPTIMIZATION.get('use_chunking', True) and total_grid_size > 1000000
    
    if use_chunking:
        logger.info("WS0-Config: Using chunked grid creation for memory efficiency")
        chunk_size = MEMORY_OPTIMIZATION.get('chunk_size', 100000)
        grid_chunks = []
        
        # Process entity combinations in chunks
        for i in range(0, len(unique_entity_combos), chunk_size // len(time_values)):
            chunk_entities = unique_entity_combos.iloc[i:i + (chunk_size // len(time_values))]
            if len(chunk_entities) == 0:
                break
                
            time_df = pd.DataFrame({time_col: time_values})
            try:
                chunk_grid = chunk_entities.merge(time_df, how='cross')
            except TypeError:
                chunk_entities = chunk_entities.copy()
                chunk_entities['_key'] = 1
                time_df['_key'] = 1
                chunk_grid = chunk_entities.merge(time_df, on='_key').drop('_key', axis=1)
            
            grid_chunks.append(chunk_grid)
            logger.info(f"WS0-Config: Processed chunk {len(grid_chunks)}: {len(chunk_grid):,} rows")
        
        grid_df = pd.concat(grid_chunks, ignore_index=True)
        logger.info(f"WS0-Config: Combined {len(grid_chunks)} chunks: {len(grid_df):,} total rows")
    else:
        # Standard approach for smaller grids
        time_df = pd.DataFrame({time_col: time_values})
        
        # Create grid using cross join (memory efficient)
        # For pandas < 2.0, use assign with key trick
        try:
            # Try pandas 2.0+ cross merge
            grid_df = unique_entity_combos.merge(time_df, how='cross')
        except TypeError:
            # Fallback for older pandas: use assign with key
            unique_entity_combos = unique_entity_combos.copy()
            unique_entity_combos['_key'] = 1
            time_df['_key'] = 1
            grid_df = unique_entity_combos.merge(time_df, on='_key').drop('_key', axis=1)

    # Left join with aggregated data (use merge in chunks if very large)
    if use_chunking and len(grid_df) > 1000000:
        logger.info("WS0-Config: Merging aggregated data in chunks")
        merge_chunks = []
        chunk_size = MEMORY_OPTIMIZATION.get('chunk_size', 100000)
        
        for i in range(0, len(grid_df), chunk_size):
            chunk_grid = grid_df.iloc[i:i + chunk_size]
            chunk_merged = chunk_grid.merge(agg_df, on=groupby_keys, how='left')
            merge_chunks.append(chunk_merged)
            logger.info(f"WS0-Config: Merged chunk {len(merge_chunks)}: {len(chunk_merged):,} rows")
        
        master_df = pd.concat(merge_chunks, ignore_index=True)
    else:
        master_df = grid_df.merge(agg_df, on=groupby_keys, how='left')

    # Zero-fill missing values
    fill_cols = _get_fill_columns(agg_df.columns)
    master_df[fill_cols] = master_df[fill_cols].fillna(0)

    # Sort for time ordering
    master_df = master_df.sort_values(groupby_keys).reset_index(drop=True)

    filled_rows = len(master_df) - len(agg_df)
    logger.info(f"WS0-Config: Grid complete: {len(master_df):,} rows ({filled_rows:,} zero-filled)")

    return master_df


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
        from src.config import get_dataset_config
        config = get_dataset_config()
        sort_cols = config['groupby_keys']
        master_df_pd = master_df_pd.sort_values(sort_cols).reset_index(drop=True)
        logger.info(f"WS0-Polars: Re-sorted by: {sort_cols}")

    logger.info(f"WS0-Polars: Complete: {master_df_pd.shape[0]:,} rows, {master_df_pd.shape[1]} columns")
    return master_df_pd


# Pandas implementation (fallback)
def aggregate_to_weekly_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to weekly/hours level using pandas."""
    from src.config import get_dataset_config
    config = get_dataset_config()
    groupby_keys = config['groupby_keys']

    # Verify required columns based on dataset
    sales_col = config['required_columns'][3] if len(config['required_columns']) > 3 else 'SALES_VALUE'
    required_cols = groupby_keys + [sales_col]
    # QUANTITY is optional for FreshRetail
    if 'QUANTITY' in df.columns:
        required_cols.append('QUANTITY')

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
    from src.config import get_dataset_config
    config = get_dataset_config()

    # Get dimensions based on config
    groupby_keys = config['groupby_keys']
    time_col = config['time_column']

    all_entities = {}
    for key in groupby_keys[:-1]:  # All except time column
        all_entities[key] = sorted(df_agg[key].unique())

    # Get time values based on temporal unit
    if config['temporal_unit'] == 'hour':
        time_values = sorted(df_agg[time_col].unique())
    else:  # week
        time_values = list(range(1, 105))  # Weeks 1-104

    entity_counts = [len(all_entities[key]) for key in all_entities.keys()]
    total_combinations = np.prod(entity_counts) * len(time_values)

    logger.info(f"WS0-Pandas: Grid: {'×'.join([f'{count:,}' for count in entity_counts])}×{len(time_values):,} = {total_combinations:,} combinations")

    # Create complete grid using dynamic entities
    entity_values = [all_entities[key] for key in all_entities.keys()]
    entity_names = list(all_entities.keys()) + [time_col]

    grid_combinations = []
    for entity_combo in pd.MultiIndex.from_product(entity_values, names=list(all_entities.keys())):
        for time_val in time_values:
            row = dict(zip(entity_names, entity_combo + (time_val,)))
            grid_combinations.append(row)

    master_grid = pd.DataFrame(grid_combinations)

    # Left join aggregated data
    master_df = pd.merge(master_grid, df_agg, on=groupby_keys, how='left')

    # Zero-fill missing values
    fill_cols = _get_fill_columns(master_df.columns)
    master_df[fill_cols] = master_df[fill_cols].fillna(0)

    # Dynamic sorting based on dataset
    from src.config import get_dataset_config
    config = get_dataset_config()
    sort_cols = config['groupby_keys']
    master_df = master_df.sort_values(sort_cols).reset_index(drop=True)
    logger.info(f"WS0-Pandas: Sorted by: {sort_cols}")

    filled_rows = len(master_df) - len(df_agg)
    logger.info(f"WS0-Pandas: Grid complete: {len(master_df):,} rows ({filled_rows:,} zero-filled)")
    return master_df


def prepare_master_dataframe_pandas(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    """Main orchestrator using pandas (fallback implementation)."""
    # Force garbage collection before processing
    import gc
    gc.collect()

    logger.info(f"WS0-Pandas: Processing {len(raw_transactions):,} transactions")

    # Handle column name conversion for different datasets
    from src.config import get_dataset_config
    config = get_dataset_config()

    # For FreshRetail: convert 'dt' to 'hour_timestamp' if needed
    if config['name'] == 'FreshRetailNet-50K' and 'dt' in raw_transactions.columns:
        raw_transactions = raw_transactions.copy()
        raw_transactions['hour_timestamp'] = pd.to_datetime(raw_transactions['dt'])
        logger.info("WS0-Pandas: Converted 'dt' to 'hour_timestamp' for FreshRetail")

    # For FreshRetail: convert 'sale_amount' to 'sales_quantity' if needed
    if 'sale_amount' in raw_transactions.columns and 'sales_quantity' not in raw_transactions.columns:
        raw_transactions = raw_transactions.copy()
        raw_transactions['sales_quantity'] = raw_transactions['sale_amount']
        logger.info("WS0-Pandas: Converted 'sale_amount' to 'sales_quantity' for FreshRetail")

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

    # Verify time ordering using config
    config = get_dataset_config()
    time_col = config['time_column']
    group_cols = config['groupby_keys'][:-1]  # All except time column

    # Check if time column is sorted within each group
    is_sorted = master_df.groupby(group_cols)[time_col].apply(
        lambda x: x.is_monotonic_increasing
    ).all()

    if not is_sorted:
        sort_cols = config['groupby_keys']
        master_df = master_df.sort_values(sort_cols).reset_index(drop=True)
        logger.info(f"WS0-Pandas: Re-sorted by: {sort_cols}")

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
        df_weekly: Weekly/hourly aggregated transaction data

    Returns:
        DataFrame with time series for active product-store pairs only
    """
    from src.config import get_dataset_config
    config = get_dataset_config()

    # Get column names from config
    groupby_keys = config['groupby_keys']
    product_col, store_col, time_col = groupby_keys

    # Get active product-store pairs (those with at least 1 transaction)
    active_pairs = df_weekly[[product_col, store_col]].drop_duplicates()
    n_active_pairs = len(active_pairs)

    # Get all time periods in the dataset
    all_periods = pd.Series(sorted(df_weekly[time_col].unique()), name=time_col)
    n_periods = len(all_periods)

    # Calculate memory savings (only meaningful for weekly data)
    if config['temporal_unit'] == 'week':
        total_full = len(df_weekly[product_col].unique()) * len(df_weekly[store_col].unique()) * 104
        total_optimized = n_active_pairs * n_periods
        memory_reduction = ((total_full - total_optimized) / total_full * 100)
        logger.info(f"WS0-Optimized: ACTIVE pairs only: {n_active_pairs:,} pairs × {n_periods:,} periods = {total_optimized:,} rows")
        logger.info(f"WS0-Optimized: Memory reduction: {memory_reduction:.1f}%")
    else:
        logger.info(f"WS0-Optimized: ACTIVE pairs only: {n_active_pairs:,} pairs × {n_periods:,} periods")

    # Create optimized grid using active pairs only
    optimized_grid = pd.DataFrame([
        (getattr(row, product_col), getattr(row, store_col), period)
        for _, row in active_pairs.iterrows()
        for period in all_periods
    ], columns=[product_col, store_col, time_col])

    # Left join with weekly data
    master_df = optimized_grid.merge(
        df_weekly,
        on=[product_col, store_col, time_col],
        how='left'
    )

    # Zero-fill missing sales data
    fill_cols = _get_fill_columns(df_weekly.columns)
    master_df[fill_cols] = master_df[fill_cols].fillna(0)

    # Sort by time dimension for time-series features
    from src.config import get_dataset_config
    config = get_dataset_config()
    sort_cols = config['groupby_keys']
    master_df = master_df.sort_values(sort_cols).reset_index(drop=True)
    logger.info(f"WS0-Optimized: Sorted by: {sort_cols}")

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
    from src.config import get_dataset_config
    config = get_dataset_config()
    sort_cols = config['groupby_keys']
    master_df = master_df.sort_values(sort_cols).reset_index(drop=True)
    logger.info(f"WS0-Pandas-Chunked: Sorted by: {sort_cols}")

    # Final garbage collection
    del chunks, product_store_combinations
    gc.collect()

    filled_rows = len(master_df) - len(df_weekly)
    logger.info(f"WS0-Pandas-Chunked: Grid complete: {len(master_df):,} rows ({filled_rows:,} zero-filled)")
    return master_df


# Backward compatibility exports
enrich_aggregation_features = prepare_master_dataframe_optimized
aggregate_to_weekly = aggregate_to_weekly_pandas

# New config-driven exports
create_master_grid_config = create_master_grid  # Alias for clarity
