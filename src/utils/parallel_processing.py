"""
Parallel Processing Utilities for Feature Engineering
====================================================
Provides parallel processing capabilities for pandas operations,
especially for groupby operations that can be parallelized.

Usage:
    from src.utils.parallel_processing import parallel_groupby_apply
    
    result_df = parallel_groupby_apply(
        df, 
        group_cols=['PRODUCT_ID', 'STORE_ID'],
        func=process_group,
        n_jobs=12
    )
"""
import logging
from functools import partial
from typing import Callable, List, Tuple, Any, Dict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def parallel_groupby_apply(
    df: pd.DataFrame,
    group_cols: List[str],
    func: Callable,
    n_jobs: int = -1,
    verbose: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Apply function to each group in parallel using joblib.
    
    This is much faster than pandas groupby().apply() for CPU-intensive operations
    because it distributes groups across multiple CPU cores.
    
    Args:
        df: DataFrame to process
        group_cols: Columns to group by (e.g., ['PRODUCT_ID', 'STORE_ID'])
        func: Function to apply to each group
               Signature: func(group_df: pd.DataFrame, **kwargs) -> pd.DataFrame
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)
        verbose: Verbosity level for joblib (0 = silent, 10 = very verbose)
        **kwargs: Additional arguments to pass to func
    
    Returns:
        DataFrame with function applied to each group (concatenated)
    
    Example:
        def process_group(group_df, target_col='SALES_VALUE'):
            # Process single group
            group_df['new_feature'] = group_df[target_col].mean()
            return group_df
        
        result = parallel_groupby_apply(
            df, 
            group_cols=['PRODUCT_ID', 'STORE_ID'],
            func=process_group,
            n_jobs=12,
            target_col='SALES_VALUE'
        )
    """
    if n_jobs == 1:
        # Sequential processing (for debugging)
        logger.info(f"Sequential processing {len(df):,} rows in groups...")
        grouped = df.groupby(group_cols)
        results = [func(group_df, **kwargs) for _, group_df in grouped]
        result_df = pd.concat(results, ignore_index=True)
        logger.info(f"Sequential processing complete: {len(result_df):,} rows")
        return result_df
    
    # Get groups
    grouped = df.groupby(group_cols)
    groups = list(grouped)
    n_groups = len(groups)
    
    logger.info(f"Parallel processing {n_groups:,} groups with {n_jobs} jobs...")
    
    # Process groups in parallel
    # Try 'multiprocessing' first (faster for CPU-intensive tasks)
    # Fallback to 'threading' if multiprocessing fails (e.g., on Windows with nested functions)
    backend = 'multiprocessing'
    try:
        results = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(func)(group_df.copy(), **kwargs) 
            for _, group_df in groups
        )
    except Exception as e:
        logger.warning(f"Multiprocessing backend failed ({e}), trying threading backend...")
        backend = 'threading'
        results = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(func)(group_df.copy(), **kwargs) 
            for _, group_df in groups
        )
    
    # Combine results
    if not results:
        logger.warning("No results returned from parallel processing!")
        return df
    
    result_df = pd.concat(results, ignore_index=True)
    
    logger.info(f"Parallel processing complete: {len(result_df):,} rows from {n_groups:,} groups")
    return result_df


def parallel_chunk_apply(
    df: pd.DataFrame,
    func: Callable,
    chunk_size: int = 10000,
    n_jobs: int = -1,
    verbose: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Split DataFrame into chunks and process in parallel.
    
    Useful when processing large DataFrames that don't need groupby operations.
    
    Args:
        df: DataFrame to process
        func: Function to apply to each chunk
               Signature: func(chunk_df: pd.DataFrame, **kwargs) -> pd.DataFrame
        chunk_size: Number of rows per chunk
        n_jobs: Number of parallel jobs
        verbose: Verbosity level for joblib
        **kwargs: Additional arguments to pass to func
    
    Returns:
        Combined DataFrame from all chunks
    """
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    logger.info(f"Processing {len(df):,} rows in {n_chunks} chunks with {n_jobs} jobs...")
    
    chunks = [df.iloc[i:i+chunk_size].copy() for i in range(0, len(df), chunk_size)]
    
    # Try multiprocessing first, fallback to threading if needed
    backend = 'multiprocessing'
    try:
        results = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(func)(chunk, **kwargs) for chunk in chunks
        )
    except Exception as e:
        logger.warning(f"Multiprocessing backend failed ({e}), trying threading backend...")
        backend = 'threading'
        results = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(func)(chunk, **kwargs) for chunk in chunks
        )
    
    if not results:
        logger.warning("No results returned from chunk processing!")
        return df
    
    result_df = pd.concat(results, ignore_index=True)
    logger.info(f"Chunk processing complete: {len(result_df):,} rows")
    return result_df


def parallel_column_apply(
    df: pd.DataFrame,
    func: Callable,
    columns: List[str] = None,
    n_jobs: int = -1,
    verbose: int = 1,
    **kwargs
) -> pd.DataFrame:
    """
    Apply function to multiple columns in parallel.
    
    Useful when creating multiple independent features.
    
    Args:
        df: DataFrame to process
        func: Function to apply to each column
               Signature: func(df: pd.DataFrame, col: str, **kwargs) -> pd.Series
        columns: List of columns to process (if None, process all numeric columns)
        n_jobs: Number of parallel jobs
        verbose: Verbosity level
        **kwargs: Additional arguments to pass to func
    
    Returns:
        DataFrame with new columns added
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Parallel processing {len(columns)} columns with {n_jobs} jobs...")
    
    # Try multiprocessing first, fallback to threading if needed
    backend = 'multiprocessing'
    try:
        results = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(func)(df, col, **kwargs) for col in columns
        )
    except Exception as e:
        logger.warning(f"Multiprocessing backend failed ({e}), trying threading backend...")
        backend = 'threading'
        results = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)(
            delayed(func)(df, col, **kwargs) for col in columns
        )
    
    # Add results as new columns
    result_df = df.copy()
    for col, result_series in zip(columns, results):
        if isinstance(result_series, pd.Series):
            result_df[f"{col}_processed"] = result_series
    
    logger.info(f"Column processing complete: {len(result_df):,} rows")
    return result_df

