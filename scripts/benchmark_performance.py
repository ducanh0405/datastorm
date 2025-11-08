#!/usr/bin/env python3
"""
Performance Benchmark Script
============================
Compares pandas vs Polars performance for key operations.

Usage:
    python scripts/benchmark_performance.py
    python scripts/benchmark_performance.py --use-sample  # Use 1% sample data
"""
import time
import pandas as pd
import logging
from pathlib import Path
import argparse
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIRS, OUTPUT_FILES
from src.pipelines._01_load_data import load_competition_data

# Try to import Polars
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerformanceBenchmark:
    """Benchmark pandas vs Polars performance."""

    def __init__(self, use_sample: bool = True):
        self.use_sample = use_sample
        self.results = {}

        if use_sample:
            self.data_dir = DATA_DIRS['poc_data']
            logger.info("Using POC sample data (1%) for benchmarking")
        else:
            self.data_dir = DATA_DIRS['raw_data'] if DATA_DIRS['raw_data'].exists() else DATA_DIRS['poc_data']
            logger.info(f"Using full data from: {self.data_dir}")

    def load_data_pandas(self) -> pd.DataFrame:
        """Load data using pandas."""
        start_time = time.time()
        data = load_competition_data(self.data_dir)
        if 'transaction_data' not in data:
            raise ValueError("No transaction_data found")

        df = data['transaction_data']
        load_time = time.time() - start_time
        logger.info(f"Load time: {load_time:.2f}s")
        return df

    def load_data_polars(self) -> pl.DataFrame:
        """Load data using Polars."""
        start_time = time.time()

        files = [f for f in self.data_dir.iterdir() if f.is_file() and f.suffix in ['.csv', '.parquet']]
        transaction_file = None

        for file in files:
            if 'transaction' in file.stem.lower():
                transaction_file = file
                break

        if not transaction_file:
            raise ValueError("No transaction file found")

        if transaction_file.suffix == '.csv':
            df = pl.read_csv(transaction_file)
        else:
            df = pl.read_parquet(transaction_file)

        load_time = time.time() - start_time
        logger.info(f"Load time: {load_time:.2f}s")
        return df

    def benchmark_aggregation_pandas(self, df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Benchmark pandas aggregation."""
        start_time = time.time()

        # Simulate WS0 aggregation
        df_agg = df.groupby(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).agg({
            'SALES_VALUE': 'sum',
            'QUANTITY': 'sum',
            'RETAIL_DISC': 'sum',
            'COUPON_DISC': 'sum',
        }).reset_index()

        agg_time = time.time() - start_time
        logger.info(f"Aggregation time: {agg_time:.2f}s")
        return df_agg, agg_time

    def benchmark_aggregation_polars(self, df: pl.DataFrame) -> tuple[pl.DataFrame, float]:
        """Benchmark Polars aggregation."""
        start_time = time.time()

        # Polars aggregation
        df_agg = df.group_by(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).agg([
            pl.col('SALES_VALUE').sum().alias('SALES_VALUE'),
            pl.col('QUANTITY').sum().alias('QUANTITY'),
            pl.col('RETAIL_DISC').sum().alias('RETAIL_DISC'),
            pl.col('COUPON_DISC').sum().alias('COUPON_DISC'),
        ])

        agg_time = time.time() - start_time
        logger.info(f"Aggregation time: {agg_time:.2f}s")
        return df_agg, agg_time

    def benchmark_grid_creation_pandas(self, df_agg: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """Benchmark pandas grid creation."""
        start_time = time.time()

        # Get unique values
        all_products = df_agg['PRODUCT_ID'].unique()
        all_stores = df_agg['STORE_ID'].unique()
        all_weeks = sorted(df_agg['WEEK_NO'].unique())

        # Create grid using MultiIndex
        from itertools import product
        grid_index = pd.MultiIndex.from_product([all_products, all_stores, all_weeks],
                                              names=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])
        grid_df = pd.DataFrame(index=grid_index).reset_index()

        # Left join with aggregated data
        master_df = pd.merge(grid_df, df_agg,
                           on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'],
                           how='left')

        # Fill nulls with 0
        fill_cols = ['SALES_VALUE', 'QUANTITY', 'RETAIL_DISC', 'COUPON_DISC']
        for col in fill_cols:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(0)

        # Sort
        master_df = master_df.sort_values(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO']).reset_index(drop=True)

        grid_time = time.time() - start_time
        logger.info(f"Grid creation time: {grid_time:.2f}s")
        return master_df, grid_time

    def benchmark_grid_creation_polars(self, df_agg: pl.DataFrame) -> tuple[pl.DataFrame, float]:
        """Benchmark Polars grid creation."""
        start_time = time.time()

        # Get unique values
        all_products = df_agg.select('PRODUCT_ID').unique()
        all_stores = df_agg.select('STORE_ID').unique()
        all_weeks = df_agg.select('WEEK_NO').unique().sort('WEEK_NO')

        # Create grid using cross joins
        product_store_grid = all_products.join(all_stores, how='cross')
        full_grid = product_store_grid.join(all_weeks, how='cross').select(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])

        # Left join with aggregated data and fill nulls
        master_df = (
            full_grid
            .join(df_agg, on=['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'], how='left')
            .with_columns([
                pl.col('SALES_VALUE').fill_null(0),
                pl.col('QUANTITY').fill_null(0),
                pl.col('RETAIL_DISC').fill_null(0),
                pl.col('COUPON_DISC').fill_null(0),
            ])
            .sort(['PRODUCT_ID', 'STORE_ID', 'WEEK_NO'])
        )

        grid_time = time.time() - start_time
        logger.info(f"Grid creation time: {grid_time:.2f}s")
        return master_df, grid_time

    def run_benchmark(self):
        """Run complete benchmark suite."""
        logger.info("=" * 80)
        logger.info("STARTING PERFORMANCE BENCHMARK")
        logger.info("=" * 80)

        try:
            # Load data
            logger.info("\n[DATA LOADING]")
            logger.info("-" * 40)

            df_pandas = self.load_data_pandas()
            self.results['data_loading'] = {'rows': len(df_pandas), 'cols': len(df_pandas.columns)}

            if POLARS_AVAILABLE:
                df_polars = self.load_data_polars()
                self.results['data_loading_polars'] = {'rows': len(df_polars), 'cols': len(df_polars.columns)}
            else:
                logger.warning("⚠️ Polars not available, skipping Polars benchmarks")
                return

            # Benchmark aggregation
            logger.info("\n🔄 AGGREGATION BENCHMARK")
            logger.info("-" * 40)

            df_agg_pandas, agg_time_pandas = self.benchmark_aggregation_pandas(df_pandas)
            df_agg_polars, agg_time_polars = self.benchmark_aggregation_polars(df_polars)

            self.results['aggregation'] = {
                'pandas_time': agg_time_pandas,
                'polars_time': agg_time_polars,
                'speedup': agg_time_pandas / agg_time_polars if agg_time_polars > 0 else float('inf'),
                'rows_out': len(df_agg_pandas)
            }

            # Benchmark grid creation
            logger.info("\n🎯 GRID CREATION BENCHMARK")
            logger.info("-" * 40)

            master_pandas, grid_time_pandas = self.benchmark_grid_creation_pandas(df_agg_pandas)
            master_polars, grid_time_polars = self.benchmark_grid_creation_polars(df_agg_polars)

            self.results['grid_creation'] = {
                'pandas_time': grid_time_pandas,
                'polars_time': grid_time_polars,
                'speedup': grid_time_pandas / grid_time_polars if grid_time_polars > 0 else float('inf'),
                'rows_out': len(master_pandas)
            }

            # Calculate total speedup
            total_pandas = agg_time_pandas + grid_time_pandas
            total_polars = agg_time_polars + grid_time_polars
            total_speedup = total_pandas / total_polars if total_polars > 0 else float('inf')

            self.results['total'] = {
                'pandas_time': total_pandas,
                'polars_time': total_polars,
                'speedup': total_speedup
            }

            self.print_results()

        except Exception as e:
            logger.error(f"BENCHMARK FAILED: {e}", exc_info=True)

    def print_results(self):
        """Print benchmark results."""
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 80)

        print(f"\n[DATA LOADING]")
        if 'data_loading' in self.results:
            dl = self.results['data_loading']
            print(f"   Rows: {dl['rows']:,}, Columns: {dl['cols']}")

        if 'aggregation' in self.results:
            agg = self.results['aggregation']
            print(f"\n[AGGREGATION]")
            print(f"  Pandas: {agg['pandas_time']:.2f}s")
            print(f"  Polars: {agg['polars_time']:.2f}s")
            print(f"  Speedup: {agg['speedup']:.1f}x")

        if 'grid_creation' in self.results:
            grid = self.results['grid_creation']
            print(f"\n[GRID CREATION]")
            print(f"  Pandas: {grid['pandas_time']:.2f}s")
            print(f"  Polars: {grid['polars_time']:.2f}s")
            print(f"  Speedup: {grid['speedup']:.1f}x")
            print(f"   Output rows: {grid['rows_out']:,}")

        if 'total' in self.results:
            total = self.results['total']
            print(f"\n[TOTAL WS0 PERFORMANCE]")
            print(f"  Pandas: {total['pandas_time']:.2f}s")
            print(f"  Polars: {total['polars_time']:.2f}s")
            print(f"  Speedup: {total['speedup']:.1f}x")

        print(f"\n[RECOMMENDATIONS]")
        if 'total' in self.results and self.results['total']['speedup'] > 2:
            print("   SUCCESS: Use Polars for production! Significant performance improvement.")
        elif POLARS_AVAILABLE:
            print("   INFO: Moderate speedup. Consider Polars for large datasets.")
        else:
            print("   WARNING: Polars not available. Consider installing: pip install polars")

        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Performance Benchmark: pandas vs Polars')
    parser.add_argument('--use-sample', action='store_true',
                       help='Use 1%% sample data instead of full dataset')
    parser.add_argument('--full-data', action='store_true',
                       help='Force use full data (may be slow)')

    args = parser.parse_args()

    # Determine data size
    use_sample = args.use_sample or not args.full_data

    benchmark = PerformanceBenchmark(use_sample=use_sample)
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()


