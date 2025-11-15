"""
Smoke Tests for DataStorm Pipeline
===================================
Quick validation tests to ensure pipeline runs end-to-end without errors.
Uses sample data from data/poc_data/ or data/2_raw/ for fast execution.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.ws0_aggregation import prepare_master_dataframe  # noqa: E402
from src.features.ws2_timeseries_features import add_lag_rolling_features  # noqa: E402
from src.pipelines._01_load_data import load_data  # noqa: E402


@pytest.fixture
def sample_data_dir():
    """Sample data directory for smoke tests."""
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


@pytest.mark.smoke
def test_data_loader(sample_data_dir):
    """Test that data loader can read FreshRetail sample data."""
    # Test loading freshretail data
    dataframes, config = load_data()

    assert isinstance(dataframes, dict)
    assert 'sales' in dataframes  # FreshRetail data is loaded as 'sales'
    assert len(dataframes['sales']) > 0

    # Check required columns exist
    df = dataframes['sales']
    required_cols = ['product_id', 'store_id', 'dt', 'sale_amount']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"


@pytest.mark.smoke
def test_ws0_aggregation(sample_freshretail_data):
    """Test WS0: Aggregation & Master Grid Creation for FreshRetail."""
    master_df = prepare_master_dataframe(sample_freshretail_data)

    # Verify aggregation for FreshRetail format
    assert 'product_id' in master_df.columns
    assert 'store_id' in master_df.columns
    assert 'hour_timestamp' in master_df.columns  # FreshRetail uses hour_timestamp
    assert 'sales_quantity' in master_df.columns  # FreshRetail target column

    # Verify we have aggregated data
    assert len(master_df) > 0

    # Verify sorting by time
    is_sorted = master_df.groupby(['product_id', 'store_id'])['hour_timestamp'].apply(
        lambda x: x.is_monotonic_increasing
    ).all()
    assert is_sorted, "master_df not properly sorted by hour_timestamp"


@pytest.mark.smoke
def test_ws2_timeseries_features(sample_freshretail_data):
    """Test WS2: Leak-Safe Time-Series Features for FreshRetail (Config-Driven)."""
    from src.config import get_dataset_config
    
    # First run WS0
    master_df = prepare_master_dataframe(sample_freshretail_data)
    
    # Get dataset config to verify WS2 is config-driven
    try:
        config = get_dataset_config('freshretail')
        assert 'lag_periods' in config, "WS2 should be config-driven with lag_periods"
        assert 'rolling_windows' in config, "WS2 should be config-driven with rolling_windows"
    except (KeyError, ImportError):
        pytest.skip("Dataset config not available - WS2 config-driven test skipped")
    
    # Test WS2 config-driven function
    try:
        from src.features.ws2_timeseries_features import add_timeseries_features_config
        
        # Run WS2 with config-driven approach
        enriched_df = add_timeseries_features_config(master_df, config)
        
        # Verify lag features exist (FreshRetail naming)
        expected_lag_cols = [f'sales_quantity_lag_{lag}' for lag in config.get('lag_periods', [1, 24, 48, 168])]
        found_lags = [col for col in expected_lag_cols[:2] if col in enriched_df.columns]  # Check first 2 lags
        assert len(found_lags) > 0, f"Missing expected lag features. Expected: {expected_lag_cols[:2]}, Found: {list(enriched_df.columns)}"
        
        # Verify rolling features exist (if configured)
        if config.get('rolling_windows'):
            rolling_cols = [col for col in enriched_df.columns if 'rolling_mean' in col or 'rolling_std' in col]
            assert len(rolling_cols) > 0, "Should have rolling features"
        
        # Verify calendar features (if intraday patterns enabled)
        if config.get('has_intraday_patterns', False):
            assert 'hour_of_day' in enriched_df.columns or 'hour_sin' in enriched_df.columns, \
                "FreshRetail should have intraday pattern features"
            assert 'day_of_week' in enriched_df.columns or 'dow_sin' in enriched_df.columns, \
                "Should have day of week features"
        
            # Verify no leakage: lag_1 should be 0 for first period (NaN filled with 0 is correct behavior)
            # Note: WS2 fills NaN with 0 at the end, which is correct for time-series features
            # The important thing is that lag features don't use future data
            first_periods = enriched_df.groupby(['product_id', 'store_id']).head(1)
            lag_cols = [col for col in enriched_df.columns if '_lag_1' in col]
            if lag_cols:
                for lag_col in lag_cols[:1]:  # Check first lag column
                    if lag_col in first_periods.columns:
                        # Lag features for first period should be 0 (NaN filled) or NaN, never use future data
                        # Since WS2 fills NaN with 0, we check that values are 0 or NaN (not using future)
                        assert (first_periods[lag_col] == 0).all() or first_periods[lag_col].isna().all(), \
                            f"Lag feature {lag_col} should be 0 or NaN for first period (no future data leak)"
        
    except ImportError as e:
        pytest.skip(f"WS2 config-driven function not available: {e}")
    except Exception as e:
        pytest.fail(f"WS2 test failed: {e}")


@pytest.mark.smoke
def test_time_based_split():
    """Test that time-based split logic works correctly."""
    # Create sample data with known time range (using FreshRetail format)
    df = pd.DataFrame({
        'hour_timestamp': list(range(1, 101)),
        'sales_quantity': list(range(100)),
        'feature1': list(range(100))
    })

    # 80th percentile cutoff
    cutoff = df['hour_timestamp'].quantile(0.8)

    train_mask = df['hour_timestamp'] < cutoff
    test_mask = df['hour_timestamp'] >= cutoff

    train = df[train_mask]
    test = df[test_mask]

    # Verify split ratio
    assert len(train) / len(df) == pytest.approx(0.8, abs=0.05)

    # Verify no time overlap
    assert train['hour_timestamp'].max() < test['hour_timestamp'].min(), "Time leakage: train/test overlap!"

    # Verify no shuffle (time ordering preserved)
    assert train['hour_timestamp'].is_monotonic_increasing
    assert test['hour_timestamp'].is_monotonic_increasing


@pytest.mark.smoke
def test_quantile_model_config():
    """Test that quantile model configuration is correct."""
    import lightgbm as lgb

    # Test Q50 model
    model_q50 = lgb.LGBMRegressor(
        objective='quantile',
        alpha=0.50,
        n_estimators=10,
        random_state=42
    )

    # Verify objective is set correctly
    assert model_q50.objective == 'quantile'
    assert model_q50.alpha == 0.50


@pytest.mark.smoke
def test_freshretail_sample_data(sample_freshretail_data):
    """Test that sample data has correct structure."""
    df = sample_freshretail_data

    # Check data types and ranges
    assert df['sale_amount'].dtype in ['float64', 'float32']
    assert df['product_id'].dtype in ['int64', 'int32']
    assert df['store_id'].dtype in ['int64', 'int32']
    # city_id can be float64 (e.g., when loaded from parquet with NaN handling)
    assert df['city_id'].dtype in ['int64', 'int32', 'float64', 'float32']

    # Check value ranges
    assert df['sale_amount'].min() >= 0, "Sale amounts should be non-negative"
    assert df['product_id'].min() >= 0, "Product IDs should be non-negative"
    assert df['store_id'].min() >= 0, "Store IDs should be non-negative"

    # Check datetime column
    assert 'dt' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df['dt'])), "dt should be datetime"


@pytest.mark.smoke
def test_directory_structure():
    """Test that required directories exist."""
    required_dirs = [
        PROJECT_ROOT / 'data' / '2_raw',        # Raw data directory
        PROJECT_ROOT / 'data' / '3_processed',  # Processed data directory
        PROJECT_ROOT / 'models',                # Models directory
        PROJECT_ROOT / 'reports' / 'metrics',   # Metrics directory
        PROJECT_ROOT / 'tests',                 # Tests directory
    ]

    for directory in required_dirs:
        assert directory.exists(), f"Required directory missing: {directory}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke"])
