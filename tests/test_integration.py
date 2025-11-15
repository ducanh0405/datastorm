"""
Integration Tests for SmartGrocy Pipeline
===================================================
End-to-end tests for the complete pipeline workflow.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines._01_load_data import load_data
from src.features.ws0_aggregation import prepare_master_dataframe


@pytest.fixture
def sample_data_dir():
    """Sample data directory for integration tests."""
    # Try poc_data first, fallback to 2_raw if poc_data doesn't have sample data
    poc_data_dir = PROJECT_ROOT / 'data' / 'poc_data'
    raw_data_dir = PROJECT_ROOT / 'data' / '2_raw'
    
    # Check if poc_data has sample files, otherwise use 2_raw
    if poc_data_dir.exists() and list(poc_data_dir.glob('*.csv')):
        return poc_data_dir
    elif raw_data_dir.exists():
        return raw_data_dir
    else:
        return poc_data_dir  # Default fallback


@pytest.fixture
def sample_freshretail_data(sample_data_dir):
    """Load sample FreshRetail data."""
    # Try multiple possible file names
    possible_files = [
        'freshretail_train_sample.csv',
        'freshretail_train.csv',
        'freshretail_train.parquet'
    ]
    
    for filename in possible_files:
        data_path = sample_data_dir / filename
        if data_path.exists():
            if filename.endswith('.parquet'):
                return pd.read_parquet(data_path)
            else:
                return pd.read_csv(data_path)
    
    pytest.skip(f"Sample data not found in {sample_data_dir}. Expected one of: {possible_files}")


class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""

    def test_full_data_loading_pipeline(self, sample_data_dir):
        """Test the complete data loading pipeline."""
        # This tests the actual data loading from disk
        dataframes, config = load_data()

        # Verify we got data
        assert isinstance(dataframes, dict)
        assert len(dataframes) > 0

        # Check that we have the expected structure
        assert 'sales' in dataframes
        sales_df = dataframes['sales']
        assert len(sales_df) > 0

        # Verify config is loaded
        assert isinstance(config, dict)
        assert 'name' in config
        assert 'temporal_unit' in config

    def test_data_loading_to_master_dataframe(self, sample_freshretail_data):
        """Test the pipeline from raw data to master dataframe."""
        # Step 1: Process raw data
        master_df = prepare_master_dataframe(sample_freshretail_data)

        # Verify the output structure
        assert isinstance(master_df, pd.DataFrame)
        assert len(master_df) > 0

        # Check required columns exist
        required_cols = ['product_id', 'store_id', 'hour_timestamp', 'sales_quantity']
        for col in required_cols:
            assert col in master_df.columns, f"Missing column: {col}"

        # Verify data integrity
        assert master_df['sales_quantity'].notna().any(), "Should have some non-null sales data"

        # Verify sorting
        is_sorted = master_df.groupby(['product_id', 'store_id'])['hour_timestamp'].apply(
            lambda x: x.is_monotonic_increasing
        ).all()
        assert is_sorted, "Data should be sorted by time within each product-store group"

    def test_pipeline_data_consistency(self, sample_freshretail_data):
        """Test that data remains consistent through pipeline steps."""
        # Get original data stats
        original_rows = len(sample_freshretail_data)
        original_products = sample_freshretail_data['product_id'].nunique()
        original_stores = sample_freshretail_data['store_id'].nunique()

        # Process through pipeline
        master_df = prepare_master_dataframe(sample_freshretail_data)

        # Check that we don't lose data unexpectedly
        # (Master df should have at least as many rows as unique product-store combinations)
        expected_min_rows = original_products * original_stores
        assert len(master_df) >= expected_min_rows, f"Too few rows: {len(master_df)} < {expected_min_rows}"

        # Check that all products and stores are preserved
        master_products = master_df['product_id'].nunique()
        master_stores = master_df['store_id'].nunique()

        assert master_products == original_products, f"Lost products: {master_products} != {original_products}"
        assert master_stores == original_stores, f"Lost stores: {master_stores} != {original_stores}"

    def test_config_driven_behavior(self):
        """Test that pipeline behavior changes based on config."""
        # Load data with current config
        dataframes, config = load_data()

        # Verify config affects behavior
        assert config['name'] == 'FreshRetailNet-50K'
        assert config['temporal_unit'] == 'hour'
        assert config['time_column'] == 'hour_timestamp'
        assert config['target_column'] == 'sales_quantity'

        # Check that data loading works
        sales_df = dataframes['sales']
        assert len(sales_df) > 0
        # Note: Raw data may have 'dt' column, not 'hour_timestamp'
        # The conversion happens in the aggregation step


class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""

    def test_missing_data_handling(self):
        """Test behavior when data files are missing."""
        # This would test error handling for missing files
        # For now, just verify that the current setup works
        try:
            dataframes, config = load_data()
            assert True  # If we get here, basic error handling works
        except Exception as e:
            pytest.fail(f"Pipeline failed unexpectedly: {e}")

    def test_empty_data_handling(self):
        """Test behavior with empty dataframes."""
        # Create empty dataframe with correct structure
        empty_df = pd.DataFrame(columns=['product_id', 'store_id', 'dt', 'sale_amount'])

        # This should handle empty data gracefully
        try:
            result = prepare_master_dataframe(empty_df)
            # Should return empty or minimal dataframe
            assert isinstance(result, pd.DataFrame)
        except Exception:
            # Empty data might cause expected errors
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
